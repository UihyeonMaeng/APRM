"""Default APRM trainer with matched image/anatomy augmentation and validation."""

from __future__ import annotations

import json
from pathlib import Path
from time import time

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join
from torch import autocast, nn

from aprm.anatomy.routing_preprocessing import ensure_routing_sidecars
from aprm.training.trainers.base_trainer import BaseTrainer
from aprm.training.trainers.aprm_modules import (
    APRM,
    AnatomyDataLoader,
    BALANCE_LOSS_WEIGHT,
    EXPERT_TOP_K,
    NUM_ANATOMICAL_REGIONS,
    NUM_EXPERTS,
    OUTSIDE_THRESHOLD,
    infer_routing_dataset_class,
    routing_to_label_map,
)
from aprm.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from aprm.utilities.get_network_from_plans import get_network_from_plans
from aprm.utilities.helpers import dummy_context


class APRMTrainer(BaseTrainer):
    """Single-device, full-resolution 3D U-Net with APRM at every encoder scale."""

    routing_num_classes = NUM_ANATOMICAL_REGIONS
    n_experts = NUM_EXPERTS
    top_k = EXPERT_TOP_K
    tau_out = OUTSIDE_THRESHOLD
    balance_loss_weight = BALANCE_LOSS_WEIGHT
    grad_clip_norm = 12.0
    routing_stats_enabled = True

    def _do_i_compile(self):
        # Dynamic gate/expert selection is intentionally executed in eager mode.
        return False

    def initialize(self):
        if self.is_ddp:
            raise NotImplementedError("This APRM release supports single-device training only")
        if self.is_cascaded or len(self.configuration_manager.patch_size) != 3:
            raise ValueError("The APRM release requires non-cascaded 3D patches")
        if self.label_manager.has_regions:
            raise ValueError("APRM expects categorical segmentation labels, not overlapping region targets")
        super().initialize()
        generated = ensure_routing_sidecars(
            dataset_name_or_id=self.plans_manager.dataset_name,
            configuration=self.configuration_name,
            plans_identifier=self.plans_manager.plans_name,
            anatomy_split="Tr",
            overwrite=False,
            num_processes=0,
        )
        self.print_to_log_file(f"Checked {len(generated)} anatomical routing sidecars")
        self.dataset_class = infer_routing_dataset_class(self.preprocessed_dataset_folder)
        self.routing_stats_folder = Path(self.output_folder) / "routing_stats"
        self._routing_epoch_accumulators = {"train": {}, "val": {}}
        self.logger.update_config(
            {
                "aprm": {
                    "anatomical_gates": self.routing_num_classes,
                    "experts_per_stage": self.n_experts,
                    "expert_top_k": self.top_k,
                    "outside_threshold": self.tau_out,
                    "balance_loss_weight": self.balance_loss_weight,
                    "routed_encoder_stages": list(self.network.encoder.routed_stage_indices),
                    "expert_expansion": 4,
                    "expert_dropout": 0.1,
                    "router_noise": "Gaussian during training only",
                }
            }
        )
        self.print_to_log_file(
            f"APRM: all encoder stages {list(self.network.encoder.routed_stage_indices)}, "
            f"{self.n_experts} experts/stage, top-{self.top_k}, tau_out={self.tau_out}"
        )

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        backbone = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return APRM(backbone, image_input_channels=num_input_channels)

    def _unwrap_network(self):
        return self.network

    def _get_aux_loss(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        loss = self.network.get_aux_loss()
        if loss is None:
            raise RuntimeError("APRM balancing loss is unavailable before a network forward pass")
        return loss

    @staticmethod
    def _scalar_numpy(value: torch.Tensor):
        return value.detach().float().cpu().numpy()

    def _capture_routing_stats(self, split: str):
        if not self.routing_stats_enabled:
            return
        if not hasattr(self, "_routing_epoch_accumulators"):
            self._routing_epoch_accumulators = {"train": {}, "val": {}}
        for stage, stats in self.network.get_routing_stats().items():
            accum = self._routing_epoch_accumulators[split].setdefault(
                stage,
                {
                    "sample_count": 0,
                    "token_count": 0,
                    "gate_counts": np.zeros(NUM_ANATOMICAL_REGIONS, dtype=np.int64),
                    "dense_probability_sum": np.zeros(NUM_EXPERTS),
                    "sparse_weight_sum": np.zeros(NUM_EXPERTS),
                    "top1_token_counts": np.zeros(NUM_EXPERTS),
                    "active_expert_counts": np.zeros(NUM_EXPERTS, dtype=np.int64),
                    "occupancy_sum": np.zeros(NUM_ANATOMICAL_REGIONS),
                },
            )
            gates = stats["selected_gates"].detach().cpu().numpy()
            num_tokens = len(gates) * stats["num_tokens_per_sample"]
            accum["sample_count"] += len(gates)
            accum["token_count"] += num_tokens
            accum["gate_counts"] += np.bincount(gates, minlength=NUM_ANATOMICAL_REGIONS)
            accum["dense_probability_sum"] += stats["mean_probabilities"].float().cpu().numpy() * num_tokens
            accum["sparse_weight_sum"] += stats["mean_expert_weights"].float().cpu().numpy() * num_tokens
            accum["top1_token_counts"] += stats["token_expert_ratios"].float().cpu().numpy() * num_tokens
            accum["active_expert_counts"] += stats["active_expert_counts"].cpu().numpy()
            accum["occupancy_sum"] += stats["routing_vector"].float().sum(dim=0).cpu().numpy()

    def _write_epoch_routing_stats(self, epoch: int):
        if not self.routing_stats_enabled:
            return
        summary = {"epoch": int(epoch), "stats": {}}
        for split, stages in self._routing_epoch_accumulators.items():
            summary["stats"][split] = {}
            for stage, accum in stages.items():
                samples = max(1, accum["sample_count"])
                tokens = max(1, accum["token_count"])
                summary["stats"][split][stage] = {
                    "sample_count": accum["sample_count"],
                    "token_count": accum["token_count"],
                    "gate_counts": accum["gate_counts"].tolist(),
                    "gate_ratios": (accum["gate_counts"] / samples).tolist(),
                    "mean_dense_probability": (accum["dense_probability_sum"] / tokens).tolist(),
                    "mean_sparse_weight": (accum["sparse_weight_sum"] / tokens).tolist(),
                    "top1_token_ratios": (accum["top1_token_counts"] / tokens).tolist(),
                    "active_expert_counts": accum["active_expert_counts"].tolist(),
                    "mean_anatomical_occupancy": (accum["occupancy_sum"] / samples).tolist(),
                }
        self.routing_stats_folder.mkdir(parents=True, exist_ok=True)
        with (self.routing_stats_folder / "routing_stats_epoch.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary) + "\n")

    def on_epoch_start(self):
        super().on_epoch_start()
        self._routing_epoch_accumulators = {"train": {}, "val": {}}

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        train_loss = float(np.round(float(self.logger.get_value("train_losses", step=-1)), decimals=4))
        val_loss = float(np.round(float(self.logger.get_value("val_losses", step=-1)), decimals=4))
        pseudo_dice = [round(float(i), 4) for i in self.logger.get_value("dice_per_class_or_region", step=-1)]
        epoch_time = round(
            float(
                self.logger.get_value("epoch_end_timestamps", step=-1)
                - self.logger.get_value("epoch_start_timestamps", step=-1)
            ),
            2,
        )

        self.print_to_log_file("train_loss", train_loss)
        self.print_to_log_file("val_loss", val_loss)
        self.print_to_log_file("Pseudo dice", pseudo_dice)
        self.print_to_log_file(f"Epoch time: {epoch_time:.2f} s")

        self._write_epoch_routing_stats(self.current_epoch)

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        current_ema = float(self.logger.get_value("ema_fg_dice", step=-1))
        if self._best_ema is None or current_ema > float(self._best_ema):
            self._best_ema = current_ema
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {self._best_ema:.4f}")
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_routing_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=None,
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = AnatomyDataLoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        dl_val = AnatomyDataLoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def _network_forward(self, batch: dict, data: torch.Tensor):
        if "routing_vector" not in batch:
            raise ValueError("APRM training batches require an anatomical occupancy vector")
        routing = batch["routing_vector"].to(self.device, non_blocking=True)
        return self.network(data, routing)

    def _unpack_loaded_case_for_inference(self, loaded):
        if len(loaded) != 5:
            raise ValueError("APRM validation requires a dataset with anatomical routing sidecars")
        return loaded

    def _augment_inference_input(self, data, seg_prev, routing):
        data = super()._augment_inference_input(data, seg_prev, routing)
        labels = routing_to_label_map(routing)
        if labels is None:
            raise FileNotFoundError("Missing anatomical routing sidecar during APRM validation")
        if tuple(labels.shape[1:]) != tuple(data.shape[1:]):
            raise ValueError("Validation image and anatomy must have identical spatial shapes")
        return np.concatenate((data, labels.astype(data.dtype, copy=False)), axis=0)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [item.to(self.device, non_blocking=True) for item in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self._network_forward(batch, data)
            self._capture_routing_stats("train")
            seg_loss = self.loss(output, target)
            aux_loss = self._get_aux_loss(data.device, seg_loss.dtype)
            total_loss = seg_loss + self.balance_loss_weight * aux_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite APRM training loss; check images, targets, and learning rate")
        skipped = False
        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            skipped = not bool(torch.isfinite(grad_norm))
            if skipped:
                self.print_to_log_file("AMP skipped an optimizer step because gradients overflowed")
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm, error_if_nonfinite=True)
            self.optimizer.step()
        return {
            "loss": self._scalar_numpy(total_loss),
            "seg_loss": self._scalar_numpy(seg_loss),
            "aux_loss": self._scalar_numpy(aux_loss),
            "skipped_nonfinite_update": np.asarray(int(skipped), dtype=np.int64),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [i.to(self.device, non_blocking=True) for i in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self._network_forward(batch, data)
            self._capture_routing_stats("val")
            seg_loss = self.loss(output, target)
            aux_loss = self._get_aux_loss(device=data.device, dtype=seg_loss.dtype)
            l = seg_loss + self.balance_loss_weight * aux_loss

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]
        if not torch.isfinite(l) or not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite APRM validation loss or prediction")

        axes = [0] + list(range(2, output.ndim))
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float16)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        mask = None
        if self.label_manager.has_ignore_label:
            mask = (target != self.label_manager.ignore_label).float()
            target = target.clone()
            target[target == self.label_manager.ignore_label] = 0

        from aprm.training.loss.dice import get_tp_fp_fn_tn

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]

        return {
            "loss": self._scalar_numpy(l),
            "seg_loss": self._scalar_numpy(seg_loss),
            "aux_loss": self._scalar_numpy(aux_loss),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
