"""APRM data handling and all-encoder anatomical routing.

The released configuration is fixed: seven anatomical gates, eight shared
experts per encoder stage, top-2 expert routing, and outside threshold 0.9.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import isfile, join
from threadpoolctl import threadpool_limits
from torch import nn

import blosc2

from aprm.training.dataloading.data_loader import APRMDataLoader
from aprm.training.dataloading.dataset import APRMDatasetBlosc2, APRMDatasetNumpy

NUM_ANATOMICAL_REGIONS = 7
NUM_EXPERTS = 8
EXPERT_TOP_K = 2
OUTSIDE_THRESHOLD = 0.9
BALANCE_LOSS_WEIGHT = 1e-2


def load_routing_map(folder: str, identifier: str) -> Optional[np.ndarray]:
    """Load a categorical anatomical sidecar without guessing a missing prior."""
    npy_file = join(folder, identifier + "_routing.npy")
    if isfile(npy_file):
        return np.load(npy_file, mmap_mode="r", allow_pickle=False)
    npz_file = join(folder, identifier + "_routing.npz")
    if isfile(npz_file):
        with np.load(npz_file, allow_pickle=False) as archive:
            for key in ("routing", "data", "seg", "arr_0"):
                if key in archive.files:
                    return archive[key]
            if len(archive.files) == 1:
                return archive[archive.files[0]]
            raise ValueError(f"Ambiguous or empty routing archive: {npz_file}")
    b2nd_file = join(folder, identifier + "_routing.b2nd")
    if isfile(b2nd_file):
        mmap_kwargs = {} if os.name == "nt" else {"mmap_mode": "r"}
        return blosc2.open(urlpath=b2nd_file, mode="r", dparams={"nthreads": 1}, **mmap_kwargs)
    return None


def routing_to_label_map(routing: Optional[np.ndarray], num_classes: int = NUM_ANATOMICAL_REGIONS):
    """Return a validated categorical map of shape ``(1, D, H, W)``.

    Integer labels, channel-first distributions, and channel-last distributions
    are accepted. Missing maps are returned as None for the caller to report.
    Invalid IDs are rejected instead of silently clipping to another anatomy.
    """
    if num_classes != NUM_ANATOMICAL_REGIONS:
        raise ValueError("The released APRM configuration requires seven anatomical classes")
    if routing is None:
        return None
    if not isinstance(routing, np.ndarray):
        routing = routing[:]
    routing = np.asarray(routing)
    if routing.ndim == 3:
        labels = routing
    elif routing.ndim == 4 and routing.shape[0] == 1:
        labels = routing[0]
    elif routing.ndim == 4 and routing.shape[0] == num_classes:
        if not np.isfinite(routing).all() or np.any(routing < 0):
            raise ValueError("Anatomical channel values must be finite and nonnegative")
        labels = routing.argmax(axis=0)
    elif routing.ndim == 4 and routing.shape[-1] == 1:
        labels = routing[..., 0]
    elif routing.ndim == 4 and routing.shape[-1] == num_classes:
        if not np.isfinite(routing).all() or np.any(routing < 0):
            raise ValueError("Anatomical channel values must be finite and nonnegative")
        labels = routing.argmax(axis=-1)
    else:
        raise ValueError(f"Expected a 3D categorical anatomy map or 7-channel map, got {routing.shape}")
    if labels.size == 0 or not np.isfinite(labels).all():
        raise ValueError("Anatomical label map must be nonempty and finite")
    rounded = np.rint(labels)
    if not np.array_equal(labels, rounded) or rounded.min() < 0 or rounded.max() >= num_classes:
        raise ValueError("Anatomical labels must be integers in [0, 6]")
    return rounded.astype(np.int16, copy=False)[None]


def routing_to_channels(routing: Optional[np.ndarray], num_classes: int = NUM_ANATOMICAL_REGIONS):
    """Convert a validated anatomical label map to channel-first one-hot data."""
    labels = routing_to_label_map(routing, num_classes)
    if labels is None:
        return None
    return np.moveaxis(np.eye(num_classes, dtype=np.float32)[labels[0]], -1, 0)


class RoutedDatasetNumpy(APRMDatasetNumpy):
    def load_case(self, identifier):
        data, seg, seg_prev, properties = super().load_case(identifier)
        return data, seg, seg_prev, properties, load_routing_map(self.source_folder, identifier)


class RoutedDatasetBlosc2(APRMDatasetBlosc2):
    def load_case(self, identifier):
        data, seg, seg_prev, properties = super().load_case(identifier)
        return data, seg, seg_prev, properties, load_routing_map(self.source_folder, identifier)


def infer_routing_dataset_class(folder: str):
    files = [name for name in os.listdir(folder) if "_routing." not in name]
    if any(name.endswith(".b2nd") for name in files):
        return RoutedDatasetBlosc2
    if any(name.endswith((".npz", ".npy")) for name in files):
        return RoutedDatasetNumpy
    raise RuntimeError(f"Unable to infer dataset format in {folder}")


class AnatomyDataLoader(APRMDataLoader):
    """Apply identical spatial transforms to images, targets, and anatomy.

    A single categorical anatomy channel passes through the segmentation
    transforms. It is removed from the target before loss computation, and the
    transformed patch occupancy is passed separately to the model.
    """

    routing_num_classes = NUM_ANATOMICAL_REGIONS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.patch_size_was_2d:
            raise ValueError("APRM requires 3D patches")

    def determine_shapes(self):
        data, seg, seg_prev, _, _ = self._data.load_case(self._data.identifiers[0])
        if seg_prev is not None:
            raise ValueError("APRM supports the non-cascaded 3D configuration only")
        spatial_shape = self.final_patch_size[1:] if self.patch_size_was_2d else self.final_patch_size
        return ((self.batch_size, data.shape[0], *spatial_shape), (self.batch_size, seg.shape[0], *spatial_shape))

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = torch.empty(self.data_shape, dtype=torch.float32)
        seg_all = None
        routing_vectors = torch.empty((self.batch_size, NUM_ANATOMICAL_REGIONS), dtype=torch.float32)
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for batch_index, identifier in enumerate(selected_keys):
                data, seg, seg_prev, properties, routing = self._data.load_case(identifier)
                if seg_prev is not None:
                    raise ValueError("APRM supports the non-cascaded 3D configuration only")
                labels = routing_to_label_map(routing)
                if labels is None:
                    raise FileNotFoundError(f"Missing anatomical routing sidecar for {identifier}")
                shape = data.shape[1:]
                if tuple(labels.shape[1:]) != tuple(shape):
                    raise ValueError(f"Anatomy/image grid mismatch for {identifier}: {labels.shape[1:]} vs {shape}")
                lower, upper = self.get_bbox(shape, self.get_do_oversample(batch_index), properties["class_locations"])
                bbox = [[low, high] for low, high in zip(lower, upper)]
                image = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                target = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1)).to(torch.int16)
                anatomy = torch.from_numpy(crop_and_pad_nd(labels, bbox, 0)).to(torch.int16)
                target_channels = target.shape[0]
                joint_target = torch.cat((target, anatomy), dim=0)
                if self.transforms is not None:
                    transformed = self.transforms(image=image, segmentation=joint_target)
                    image, joint_target = transformed["image"], transformed["segmentation"]
                if isinstance(joint_target, list):
                    anatomy = joint_target[0][target_channels : target_channels + 1]
                    target = [sample[:target_channels] for sample in joint_target]
                else:
                    anatomy = joint_target[target_channels : target_channels + 1]
                    target = joint_target[:target_channels]
                # Segmentation transforms can pad with -1; this is outside for anatomy.
                if not torch.isfinite(anatomy).all():
                    raise ValueError(f"Non-finite transformed anatomy for {identifier}")
                anatomy = anatomy.float().round().long().clamp_min_(0)
                if (anatomy >= NUM_ANATOMICAL_REGIONS).any():
                    raise ValueError(f"Invalid transformed anatomical label for {identifier}")
                counts = torch.bincount(anatomy.reshape(-1), minlength=NUM_ANATOMICAL_REGIONS).float()
                routing_vectors[batch_index] = counts / anatomy.numel()
                data_all[batch_index] = image
                if isinstance(target, list):
                    if seg_all is None:
                        seg_all = [
                            torch.empty((self.batch_size, *sample.shape), dtype=sample.dtype) for sample in target
                        ]
                    for index, sample in enumerate(target):
                        seg_all[index][batch_index] = sample
                else:
                    if seg_all is None:
                        seg_all = torch.empty((self.batch_size, *target.shape), dtype=target.dtype)
                    seg_all[batch_index] = target
        return {"data": data_all, "target": seg_all, "keys": selected_keys, "routing_vector": routing_vectors}


def prepare_routing_vector(routing: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Compute patch occupancy once, on the original input patch grid.

    Accept ``(B, 7)`` occupancy, ``(B, 1, D, H, W)`` categorical labels, or
    ``(B, 7, D, H, W)`` anatomy channels. All-zero channel voxels introduced by
    sliding-window padding count as outside (class 0), as in training.
    """
    if routing is None or routing.numel() == 0:
        raise ValueError("APRM requires an anatomical prior for every image patch")
    routing = routing.to(device=device, dtype=torch.float32)
    if routing.shape[0] != batch_size or not torch.isfinite(routing).all():
        raise ValueError("Anatomy batch size must match the image and values must be finite")
    if routing.ndim == 2 and routing.shape[1] == NUM_ANATOMICAL_REGIONS:
        occupancy = routing
        if (occupancy < 0).any() or (occupancy.sum(dim=1) <= 0).any():
            raise ValueError("Occupancy vectors must be nonnegative with positive total mass")
    elif routing.ndim == 5 and routing.shape[1] == 1:
        labels = routing[:, 0]
        if ((labels != labels.round()) | (labels < 0) | (labels >= NUM_ANATOMICAL_REGIONS)).any():
            raise ValueError("Anatomical labels must be integers in [0, 6]")
        labels = labels.long()
        occupancy = torch.stack(
            [torch.bincount(sample.reshape(-1), minlength=NUM_ANATOMICAL_REGIONS) for sample in labels]
        ).float()
    elif routing.ndim == 5 and routing.shape[1] == NUM_ANATOMICAL_REGIONS:
        if (routing < 0).any():
            raise ValueError("Anatomical channel values must be nonnegative")
        mass = routing.sum(dim=1, keepdim=True)
        normalized = routing / mass.clamp_min(1e-8)
        outside_padding = (mass == 0).squeeze(1)
        normalized[:, 0] = normalized[:, 0] + outside_padding.to(normalized.dtype)
        occupancy = normalized.mean(dim=(2, 3, 4))
    else:
        raise ValueError(f"Unsupported anatomy shape {tuple(routing.shape)}; expected Bx7 or Bx(1/7)xDxHxW")
    return occupancy / occupancy.sum(dim=1, keepdim=True)


def select_anatomical_gate(occupancy: torch.Tensor) -> torch.Tensor:
    """Equation (4): outside if q_out >= 0.9; otherwise largest anatomical ROI."""
    roi_gate = occupancy[:, 1:].argmax(dim=1) + 1
    return torch.where(occupancy[:, 0] >= OUTSIDE_THRESHOLD, torch.zeros_like(roi_gate), roi_gate)


def compute_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    """Equations (13)-(14), averaging token probabilities within a stage."""
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("Expert probabilities must be a nonempty token-by-expert matrix")
    mean_probability = probabilities.float().mean(dim=0)
    return probabilities.shape[1] * mean_probability.square().sum()


class TokenExpert(nn.Module):
    """Residual expert adapter: C -> 4C -> C, ReLU, and dropout 0.1."""

    def __init__(self, n_embed: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(inplace=True),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    """Equations (7)-(9); Gaussian routing noise is used only in training."""

    def __init__(self, n_embed: int):
        super().__init__()
        self.topkroute_linear = nn.Linear(n_embed, NUM_EXPERTS)
        self.noise_linear = nn.Linear(n_embed, NUM_EXPERTS)
        for layer in (self.topkroute_linear, self.noise_linear):
            nn.init.kaiming_uniform_(layer.weight, a=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, tokens: torch.Tensor):
        logits = self.topkroute_linear(tokens)
        if self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(self.noise_linear(tokens))
        top_logits, indices = logits.topk(EXPERT_TOP_K, dim=-1)
        sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        sparse_logits.scatter_(-1, indices, top_logits)
        return F.softmax(sparse_logits, dim=-1), indices, F.softmax(logits, dim=-1)


class APRMRoutedStage(nn.Module):
    """One encoder stage followed by an anatomy-conditioned shared expert pool."""

    def __init__(self, stage_module: nn.Module, out_channels: int):
        super().__init__()
        self.stage = stage_module
        self.out_channels = int(out_channels)
        self.routers = nn.ModuleList([NoisyTopkRouter(out_channels) for _ in range(NUM_ANATOMICAL_REGIONS)])
        self.experts = nn.ModuleList([TokenExpert(out_channels) for _ in range(NUM_EXPERTS)])
        self.collect_token_maps = False

    def set_collect_token_maps(self, enabled: bool = True):
        self.collect_token_maps = bool(enabled)

    def forward(self, x: torch.Tensor, occupancy: torch.Tensor):
        stage_out = self.stage(x)
        tokens = stage_out.flatten(2).transpose(1, 2).contiguous()
        batch_size, n_tokens, channels = tokens.shape
        if channels != self.out_channels:
            raise RuntimeError(f"APRM stage has {channels} feature channels, expected {self.out_channels}")
        selected_gates = select_anatomical_gate(occupancy)
        flat_tokens = tokens.reshape(-1, channels)
        flat_gates = selected_gates[:, None].expand(batch_size, n_tokens).reshape(-1)
        weights = flat_tokens.new_zeros((flat_tokens.shape[0], NUM_EXPERTS))
        probabilities = torch.zeros_like(weights)
        top_indices = torch.zeros((flat_tokens.shape[0], EXPERT_TOP_K), device=x.device, dtype=torch.long)
        for gate_index, router in enumerate(self.routers):
            mask = flat_gates == gate_index
            if mask.any():
                sparse, indices, dense = router(flat_tokens[mask])
                weights[mask] = sparse.to(weights.dtype)
                probabilities[mask] = dense.to(probabilities.dtype)
                top_indices[mask] = indices
        delta = torch.zeros_like(flat_tokens)
        for expert_index, expert in enumerate(self.experts):
            mask = (top_indices == expert_index).any(dim=1)
            if mask.any():
                expert_output = expert(flat_tokens[mask])
                delta[mask] += expert_output * weights[mask, expert_index, None]
        output = stage_out + delta.view(batch_size, n_tokens, channels).transpose(1, 2).reshape_as(stage_out)
        balance_loss = compute_balance_loss(probabilities)

        # Top-1 diagnostic maps are distinct from actual top-2 expert activation.
        top1_experts = weights.argmax(dim=1)
        top1_ratios = torch.bincount(top1_experts, minlength=NUM_EXPERTS).float() / flat_tokens.shape[0]
        active_counts = torch.bincount(top_indices.reshape(-1), minlength=NUM_EXPERTS)
        sample_weights = weights.view(batch_size, n_tokens, NUM_EXPERTS).mean(dim=1)
        stats = {
            "selected_gates": selected_gates.detach(),
            "selected_experts": sample_weights.argmax(dim=1).detach(),
            "expert_weights": sample_weights.detach(),
            "mean_expert_weights": weights.mean(dim=0).detach(),
            "mean_probabilities": probabilities.mean(dim=0).detach(),
            "token_expert_ratios": top1_ratios.detach(),
            "active_expert_counts": active_counts.detach(),
            "routing_vector": occupancy.detach(),
            "mean_anatomy": occupancy.mean(dim=0).detach(),
            "has_routing_ratio": occupancy.new_ones(()),
            "background_fraction": occupancy[:, 0].detach(),
            "max_roi_fraction": occupancy[:, 1:].max(dim=1).values.detach(),
            "is_out_gate": (selected_gates == 0).detach(),
            "out_gate_ratio": (selected_gates == 0).float().mean().detach(),
            "num_gates": NUM_ANATOMICAL_REGIONS,
            "num_experts": NUM_EXPERTS,
            "num_tokens_per_sample": n_tokens,
            "token_map_shape": list(stage_out.shape[2:]),
            "tau_out": OUTSIDE_THRESHOLD,
        }
        if self.collect_token_maps:
            shape = (batch_size, *stage_out.shape[2:])
            stats["token_selected_expert_map"] = top1_experts.view(shape).to(torch.int16).detach().cpu()
            stats["token_expert_confidence_map"] = (
                weights.max(dim=1).values.view(shape).to(torch.float16).detach().cpu()
            )
        return output, balance_loss, stats


class APRMEncoder(nn.Module):
    """Route every encoder stage, including stage 0 and the bottleneck."""

    def __init__(self, original_encoder: nn.Module):
        super().__init__()
        self.stem = getattr(original_encoder, "stem", None)
        self.return_skips = True
        for name in (
            "output_channels",
            "strides",
            "conv_op",
            "norm_op",
            "norm_op_kwargs",
            "nonlin",
            "nonlin_kwargs",
            "dropout_op",
            "dropout_op_kwargs",
            "conv_bias",
            "kernel_sizes",
        ):
            setattr(self, name, getattr(original_encoder, name))
        self.routed_stage_indices = tuple(range(len(original_encoder.stages)))
        self.stages = nn.ModuleList(
            [
                APRMRoutedStage(stage, int(self.output_channels[index]))
                for index, stage in enumerate(original_encoder.stages)
            ]
        )
        self._last_aux_loss = None
        self._last_routing_stats: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor, occupancy: torch.Tensor):
        if self.stem is not None:
            x = self.stem(x)
        skips, losses, stats = [], [], {}
        for index, stage in enumerate(self.stages):
            x, loss, stage_stats = stage(x, occupancy)
            skips.append(x)
            losses.append(loss)
            stats[f"stage_{index}"] = stage_stats
        self._last_aux_loss = torch.stack(losses).mean()
        self._last_routing_stats = stats
        return skips


class APRMDecoder(nn.Module):
    """Original U-Net decoder without an additional reference to its encoder."""

    def __init__(self, original_decoder: nn.Module):
        super().__init__()
        self.deep_supervision = original_decoder.deep_supervision
        self.transpconvs = original_decoder.transpconvs
        self.stages = original_decoder.stages
        self.seg_layers = original_decoder.seg_layers

    def forward(self, skips: List[torch.Tensor]):
        x = skips[-1]
        outputs = []
        for index, stage in enumerate(self.stages):
            x = self.transpconvs[index](x)
            x = stage(torch.cat((x, skips[-(index + 2)]), dim=1))
            if self.deep_supervision or index == len(self.stages) - 1:
                outputs.append(self.seg_layers[index](x))
        return outputs[::-1] if self.deep_supervision else outputs[-1]


class APRM(nn.Module):
    """APRM segmentation model with fixed routing at every encoder stage.

    Training supplies CT channels and a Bx7 occupancy vector. Sliding-window
    inference may append a categorical anatomy channel or seven anatomy channels
    to the image; the image encoder never consumes those anatomical channels.
    """

    routing_num_classes = NUM_ANATOMICAL_REGIONS

    def __init__(self, base_network: nn.Module, image_input_channels: int):
        super().__init__()
        self.image_input_channels = int(image_input_channels)
        self.encoder = APRMEncoder(base_network.encoder)
        self.decoder = APRMDecoder(base_network.decoder)
        self._last_aux_loss = None
        self._last_routing_stats = {}
        self._collect_routing_history = False
        self._routing_history = []

    def set_collect_routing_history(self, enabled: bool = True):
        self._collect_routing_history = bool(enabled)

    def reset_routing_history(self):
        self._routing_history = []

    def consume_routing_history(self):
        history, self._routing_history = self._routing_history, []
        return history

    @staticmethod
    def _detach_stats(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: APRM._detach_stats(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [APRM._detach_stats(item) for item in value]
        return value

    def forward(self, x: torch.Tensor, routing_vector: Optional[torch.Tensor] = None):
        if x.ndim != 5:
            raise ValueError("APRM requires image tensors of shape BxCxDxHxW")
        if routing_vector is None:
            n_extra = x.shape[1] - self.image_input_channels
            if n_extra not in (1, NUM_ANATOMICAL_REGIONS):
                raise ValueError("APRM inference requires one categorical or seven anatomical channels")
            routing_vector = x[:, self.image_input_channels :]
        elif x.shape[1] != self.image_input_channels:
            raise ValueError("With explicit anatomy, the image tensor must contain only image channels")
        if routing_vector.ndim == 5 and tuple(routing_vector.shape[2:]) != tuple(x.shape[2:]):
            raise ValueError("Anatomical and image patches must use the same spatial grid")
        occupancy = prepare_routing_vector(routing_vector, x.shape[0], x.device)
        output = self.decoder(self.encoder(x[:, : self.image_input_channels], occupancy))
        self._last_aux_loss = self.encoder._last_aux_loss
        self._last_routing_stats = self.encoder._last_routing_stats
        if self._collect_routing_history:
            self._routing_history.append(self._detach_stats(self._last_routing_stats))
        return output

    def get_aux_loss(self):
        return self._last_aux_loss

    def get_routing_stats(self):
        return self._last_routing_stats
