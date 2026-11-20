"""Train, resume, or validate the default APRM model."""

import argparse
import os
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json

from aprm.paths import APRM_preprocessed, APRM_results
from aprm.training.trainers.aprm_trainer import APRMTrainer
from aprm.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def get_trainer_from_args(
    dataset_name_or_id,
    configuration,
    fold,
    trainer_name="APRMTrainer",
    plans_identifier="APRMPlans",
    continue_training=False,
    device=torch.device("cuda"),
):
    if trainer_name != "APRMTrainer" or configuration != "3d_fullres":
        raise ValueError("This release supports APRMTrainer with 3d_fullres only.")
    if not APRM_preprocessed or not APRM_results:
        raise EnvironmentError("Set APRM_preprocessed and APRM_results before training.")
    dataset = maybe_convert_to_dataset_name(dataset_name_or_id)
    folder = Path(APRM_preprocessed) / dataset
    plans_file = folder / f"{plans_identifier}.json"
    if not plans_file.is_file():
        raise FileNotFoundError(f"Run APRM_plan_and_preprocess first: {plans_file}")
    plans = load_json(str(plans_file))
    plans["continue_training"] = continue_training
    return APRMTrainer(plans, configuration, fold, load_json(str(folder / "dataset.json")), device=device)


def run_training(
    dataset_name_or_id,
    configuration,
    fold,
    trainer_class_name="APRMTrainer",
    plans_identifier="APRMPlans",
    num_gpus=1,
    export_validation_probabilities=False,
    continue_training=False,
    only_run_validation=False,
    val_with_best=False,
    device=torch.device("cuda"),
):
    if num_gpus != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This APRM release supports one device per training run; DDP is not enabled.")
    if continue_training and (only_run_validation or val_with_best):
        raise ValueError("Choose resume training or validation, not both.")
    if str(fold) != "all":
        try:
            fold = int(fold)
        except ValueError as error:
            raise ValueError("fold must be a non-negative integer or 'all'.") from error
        if fold < 0:
            raise ValueError("fold must be a non-negative integer or 'all'.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a matching PyTorch build or select -device cpu.")
    trainer = get_trainer_from_args(
        dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier, continue_training, device
    )
    output = Path(trainer.output_folder)
    if continue_training:
        checkpoint = next(
            (output / name for name in ("checkpoint_latest.pth", "checkpoint_final.pth") if (output / name).is_file()),
            None,
        )
        if checkpoint is None:
            raise FileNotFoundError(f"No resumable checkpoint found in {output}")
        trainer.load_checkpoint(str(checkpoint))
    elif only_run_validation or val_with_best:
        checkpoint = output / ("checkpoint_best.pth" if val_with_best else "checkpoint_final.pth")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Validation checkpoint not found: {checkpoint}")
        trainer.load_checkpoint(str(checkpoint))
    elif (output / "checkpoint_final.pth").exists() or (output / "checkpoint_latest.pth").exists():
        raise FileExistsError(f"Training output already contains a checkpoint: {output}. Use --c to resume.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if not (only_run_validation or val_with_best):
        trainer.run_training()
    trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    parser = argparse.ArgumentParser(description="Train the fixed all-encoder APRM lung CT model.")
    parser.add_argument("dataset_name_or_id", help="Dataset name or numeric ID")
    parser.add_argument("configuration", choices=["3d_fullres"])
    parser.add_argument("fold", help="Fold index or 'all'; see documentation/usage.md")
    parser.add_argument("-tr", default="APRMTrainer", choices=["APRMTrainer"])
    parser.add_argument("-p", default="APRMPlans", help="Plans identifier")
    parser.add_argument("-num_gpus", type=int, default=1, choices=[1], help="One device per run")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--c", action="store_true", help="Resume the latest/final checkpoint")
    action.add_argument("--val", action="store_true", help="Validate checkpoint_final.pth only")
    action.add_argument("--val_best", action="store_true", help="Validate checkpoint_best.pth only")
    parser.add_argument("--npz", action="store_true", help="Save validation probabilities")
    parser.add_argument("-device", choices=["cuda", "cpu", "mps"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    run_training(
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        args.tr,
        args.p,
        args.num_gpus,
        args.npz,
        args.c,
        args.val,
        args.val_best,
        torch.device(args.device),
    )
