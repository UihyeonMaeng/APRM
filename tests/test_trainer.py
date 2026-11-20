"""One tiny CPU training integration test with isolated APRM data roots."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_default_trainer_augmented_step_resume_and_sliding_validation(tmp_path):
    # Paths are read at module import time. A child interpreter isolates them from
    # the other tests without reloading global modules or changing user settings.
    env = os.environ.copy()
    for key, name in (("APRM_raw", "raw"), ("APRM_preprocessed", "preprocessed"), ("APRM_results", "results")):
        env[key] = str(tmp_path / name)
    env.update(
        APRM_n_proc_DA="0",
        APRM_wandb_enabled="0",
        WORLD_SIZE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    project = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
    script = textwrap.dedent("""
        import json
        import os
        from pathlib import Path
        import numpy as np
        import torch
        from aprm.run.run_training import get_trainer_from_args
        from aprm.training.dataloading.dataset import APRMDatasetNumpy

        torch.set_num_threads(1)
        torch.manual_seed(13)
        np.random.seed(13)
        name = "Dataset001_Integration"
        raw = Path(os.environ["APRM_raw"]) / name
        pre = Path(os.environ["APRM_preprocessed"]) / name
        folder = pre / "APRMPlans_3d_fullres"
        (raw / "anatomyTr").mkdir(parents=True)
        folder.mkdir(parents=True)
        arch = {
            "n_stages": 3, "features_per_stage": [4, 8, 16], "conv_op": "torch.nn.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * 3, "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
            "n_conv_per_stage": [1, 1, 1], "n_conv_per_stage_decoder": [1, 1],
            "conv_bias": True, "norm_op": "torch.nn.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None, "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
        }
        config = {
            "data_identifier": "APRMPlans_3d_fullres", "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 2, "patch_size": [8, 8, 8], "median_image_size_in_voxels": [12, 12, 12],
            "spacing": [1., 1., 1.], "normalization_schemes": ["ZScoreNormalization"],
            "use_mask_for_norm": [False], "batch_dice": False,
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": arch, "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            },
        }
        for role in ("data", "seg", "probabilities"):
            config["resampling_fn_" + role] = "resample_data_or_seg_to_shape"
            config["resampling_fn_" + role + "_kwargs"] = {
                "is_seg": role == "seg", "order": 0 if role == "seg" else 1,
                "order_z": 0, "force_separate_z": False,
            }
        plans = {
            "dataset_name": name, "plans_name": "APRMPlans", "image_reader_writer": "SimpleITKIO",
            "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
            "foreground_intensity_properties_per_channel": {"0": {}},
            "configurations": {"3d_fullres": config},
        }
        dataset = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "tumor": 1},
                   "file_ending": ".nii.gz", "numTraining": 2}
        for path, payload in ((pre / "APRMPlans.json", plans), (pre / "dataset.json", dataset),
                              (raw / "dataset.json", dataset),
                              (pre / "splits_final.json", [{"train": ["train"], "val": ["val"]}])):
            path.write_text(json.dumps(payload), encoding="utf-8")
        for case in ("train", "val"):
            image = np.random.normal(size=(1, 12, 12, 12)).astype(np.float32)
            target = np.zeros((1, 12, 12, 12), dtype=np.int16)
            target[0, 4:8, 4:8, 4:8] = 1
            anatomy = np.zeros_like(target)
            anatomy[0, :6], anatomy[0, 6:] = 1, 6
            APRMDatasetNumpy.save_case(image, target, {"class_locations": {1: np.argwhere(target == 1)}},
                                      str(folder / case))
            np.save(folder / (case + "_routing.npy"), anatomy)

        trainer = get_trainer_from_args(name, "3d_fullres", 0, device=torch.device("cpu"))
        trainer.initialize()
        assert trainer.network.encoder.routed_stage_indices == (0, 1, 2)
        trainer.num_epochs = 1
        trainer.on_epoch_start()
        trainer.on_train_epoch_start()
        train_loader, val_loader = trainer.get_dataloaders()
        batch = next(train_loader)
        assert batch["data"].shape == (2, 1, 8, 8, 8)
        assert [tuple(t.shape) for t in batch["target"]] == [(2, 1, 8, 8, 8), (2, 1, 4, 4, 4)]
        torch.testing.assert_close(batch["routing_vector"].sum(1), torch.ones(2))
        before = next(trainer.network.parameters()).detach().clone()
        train_result = trainer.train_step(batch)
        assert not torch.equal(before, next(trainer.network.parameters()).detach())
        assert np.isclose(train_result["loss"], train_result["seg_loss"] + .01 * train_result["aux_loss"])
        assert int(train_result["skipped_nonfinite_update"]) == 0
        trainer.on_train_epoch_end([train_result])
        trainer.on_validation_epoch_start()
        with torch.no_grad():
            val_result = trainer.validation_step(next(val_loader))
        assert np.isfinite(val_result["loss"])
        assert val_result["tp_hard"].shape == (1,)
        trainer.on_validation_epoch_end([val_result])
        # Plot styling is outside this test; exercise actual epoch bookkeeping,
        # routing JSONL statistics, and best-checkpoint persistence.
        trainer.logger.plot_progress_png = lambda _: None
        trainer.on_epoch_end()
        checkpoint = Path(trainer.output_folder) / "checkpoint_best.pth"
        assert checkpoint.is_file()
        assert (Path(trainer.output_folder) / "routing_stats/routing_stats_epoch.jsonl").is_file()
        trained_weights = {key: value.detach().clone() for key, value in trainer.network.state_dict().items()}
        with torch.no_grad():
            next(trainer.network.parameters()).zero_()
        trainer.load_checkpoint(str(checkpoint))
        assert trainer.current_epoch == 1
        for key, value in trainer.network.state_dict().items():
            torch.testing.assert_close(value, trained_weights[key])
        trainer.set_deep_supervision_enabled(False)
        loaded = trainer.dataset_class(str(folder), ["val"]).load_case("val")
        data = trainer._augment_inference_input(loaded[0], None, loaded[-1])
        assert data.shape[0] == 2  # Image plus one categorical anatomical channel.
        predictor = trainer._create_validation_predictor()
        # Multi-window merging and mirroring have their own predictor tests.
        # Here one tile checks the trainer's categorical-prior interface.
        predictor.use_mirroring = False
        data = data[:, :8, :8, :8]
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(data))
        assert logits.shape == (2, 8, 8, 8)
        assert torch.isfinite(logits).all()
        print("TRAINER_INTEGRATION_OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=str(project), text=True, capture_output=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "TRAINER_INTEGRATION_OK" in result.stdout
