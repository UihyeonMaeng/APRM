"""Command-line entry points for the default APRM preprocessing pipeline."""

import argparse

from aprm.configuration import default_num_processes
from aprm.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess


def _dataset_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-d", nargs="+", type=int, required=True, help="Numeric dataset IDs")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _planning_arguments(parser):
    parser.add_argument("-gpu_memory_target", type=float, default=None, help="Backbone planning memory target in GB")
    parser.add_argument(
        "-overwrite_target_spacing",
        nargs=3,
        type=float,
        default=None,
        help="Target spacing in the image reader's axis order (typically z, y, x)",
    )
    parser.add_argument("-overwrite_plans_name", default=None, help="Output plans identifier (default: APRMPlans)")


def _plan(args):
    if args.gpu_memory_target is not None and args.gpu_memory_target <= 0:
        raise ValueError("The GPU memory target must be positive.")
    if args.overwrite_target_spacing and min(args.overwrite_target_spacing) <= 0:
        raise ValueError("Target spacing values must be positive.")
    return plan_experiments(
        args.d,
        "ExperimentPlanner",
        args.gpu_memory_target,
        "DefaultPreprocessor",
        args.overwrite_target_spacing,
        args.overwrite_plans_name,
    )


def extract_fingerprint_entry():
    parser = _dataset_parser("Extract APRM dataset fingerprints.")
    parser.add_argument("-np", type=int, default=default_num_processes)
    parser.add_argument("--verify_dataset_integrity", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Regenerate an existing fingerprint")
    args = parser.parse_args()
    extract_fingerprints(
        args.d, "DatasetFingerprintExtractor", args.np, args.verify_dataset_integrity, args.clean, args.verbose
    )


def plan_experiment_entry():
    parser = _dataset_parser("Plan the default full-resolution APRM backbone.")
    _planning_arguments(parser)
    args = parser.parse_args()
    _plan(args)


def preprocess_entry():
    parser = _dataset_parser("Preprocess images and targets for APRM.")
    parser.add_argument("-c", nargs="+", default=["3d_fullres"], choices=["3d_fullres"])
    parser.add_argument("-plans_name", default="APRMPlans")
    parser.add_argument("-np", type=int, default=4, help="Preprocessing workers")
    args = parser.parse_args()
    if args.np < 1:
        parser.error("-np must be at least one")
    preprocess(args.d, args.plans_name, args.c, [args.np], args.verbose)


def plan_and_preprocess_entry():
    parser = _dataset_parser("Analyze, plan, and preprocess a 3D APRM dataset.")
    _planning_arguments(parser)
    parser.add_argument("-c", nargs="+", default=["3d_fullres"], choices=["3d_fullres"])
    parser.add_argument("-npfp", type=int, default=default_num_processes, help="Fingerprint workers")
    parser.add_argument("-np", type=int, default=4, help="Preprocessing workers")
    parser.add_argument("--verify_dataset_integrity", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Regenerate an existing fingerprint")
    parser.add_argument("--no_pp", action="store_true", help="Only create fingerprints and plans")
    args = parser.parse_args()
    if args.np < 1 or args.npfp < 1:
        parser.error("Worker counts must be at least one")
    extract_fingerprints(
        args.d, "DatasetFingerprintExtractor", args.npfp, args.verify_dataset_integrity, args.clean, args.verbose
    )
    plans_identifier = _plan(args)
    if not args.no_pp:
        preprocess(args.d, plans_identifier, args.c, [args.np], args.verbose)
