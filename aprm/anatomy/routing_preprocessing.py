from __future__ import annotations

import argparse
import multiprocessing
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, load_pickle, maybe_mkdir_p

from aprm.paths import APRM_preprocessed, APRM_raw
from aprm.anatomy.alignment import load_aligned_anatomy
from aprm.training.dataloading.dataset import infer_dataset_class
from aprm.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from aprm.utilities.plans_handling.plans_handler import PlansManager


EXPECTED_APRM_MAPPING = {
    "background": 0,
    "RUL": 1,
    "RML": 2,
    "RLL": 3,
    "LUL": 4,
    "LLL": 5,
    "CENTRAL": 6,
}


@dataclass(frozen=True)
class RoutingPrepContext:
    dataset_name: str
    plans_identifier: str
    configuration: str
    anatomy_split: str
    output_dtype: str = "int16"

    @property
    def raw_dataset_folder(self) -> str:
        return join(APRM_raw, self.dataset_name)

    @property
    def anatomy_dir(self) -> str:
        return join(self.raw_dataset_folder, f"anatomy{self.anatomy_split}")

    @property
    def preprocessed_base(self) -> str:
        return join(APRM_preprocessed, self.dataset_name)

    @property
    def plans_file(self) -> str:
        return join(self.preprocessed_base, self.plans_identifier + ".json")

    @property
    def dataset_json_file(self) -> str:
        return join(self.raw_dataset_folder, "dataset.json")


def _validate_mapping(label_mapping_file: str) -> None:
    if not os.path.isfile(label_mapping_file):
        return
    mapping = load_json(label_mapping_file)
    for key, value in EXPECTED_APRM_MAPPING.items():
        if key not in mapping or int(mapping[key]) != int(value):
            raise RuntimeError(
                f"APRM label mapping mismatch in {label_mapping_file}. Expected {EXPECTED_APRM_MAPPING}, got {mapping}"
            )


def _convert_one_case(identifier: str, context: RoutingPrepContext) -> str:
    plans = load_json(context.plans_file)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(context.configuration)
    dataset_json = load_json(context.dataset_json_file)
    preprocessed_folder = join(context.preprocessed_base, configuration_manager.data_identifier)
    dataset_class = infer_dataset_class(preprocessed_folder)
    dataset = dataset_class(preprocessed_folder, [identifier])
    data, _, _, properties = dataset.load_case(identifier)
    spatial_shape = tuple(int(i) for i in data.shape[1:])

    anatomy_file = join(context.anatomy_dir, identifier + ".nii.gz")
    seg_resampled = load_aligned_anatomy(anatomy_file, properties, spatial_shape, plans_manager, configuration_manager)
    save_path = join(preprocessed_folder, identifier + "_routing.npy")
    np.save(save_path, seg_resampled.astype(np.int16, copy=False))
    return save_path


def prepare_routing_sidecars(
    dataset_name_or_id: str | int,
    configuration: str = "3d_fullres",
    plans_identifier: str = "APRMPlans",
    anatomy_split: str = "Tr",
    overwrite: bool = False,
    num_processes: int = 0,
    identifiers: Sequence[str] | None = None,
) -> list[str]:
    if APRM_raw is None:
        raise EnvironmentError("APRM_raw is not set")
    if APRM_preprocessed is None:
        raise EnvironmentError("APRM_preprocessed is not set")

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    context = RoutingPrepContext(
        dataset_name=dataset_name,
        plans_identifier=plans_identifier,
        configuration=configuration,
        anatomy_split=anatomy_split,
    )
    if not os.path.isfile(context.plans_file):
        raise FileNotFoundError(f"Plans file not found: {context.plans_file}. Run APRM_plan_and_preprocess first.")
    if not os.path.isdir(context.anatomy_dir):
        raise FileNotFoundError(
            f"Anatomy labelmap folder not found: {context.anatomy_dir}. Run APRM_prepare_anatomy first."
        )
    _validate_mapping(join(context.anatomy_dir, "label_mapping.json"))

    plans = load_json(context.plans_file)
    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration)
    preprocessed_folder = join(context.preprocessed_base, configuration_manager.data_identifier)
    if not os.path.isdir(preprocessed_folder):
        raise FileNotFoundError(
            f"Preprocessed folder not found: {preprocessed_folder}. Run APRM_plan_and_preprocess first."
        )

    dataset_class = infer_dataset_class(preprocessed_folder)
    all_identifiers = dataset_class.get_identifiers(preprocessed_folder)
    if identifiers is None:
        identifiers = all_identifiers
    else:
        unknown = set(identifiers) - set(all_identifiers)
        if unknown:
            raise ValueError(f"Unknown case identifiers: {sorted(unknown)}")

    to_process: list[str] = []
    for identifier in identifiers:
        save_path = join(preprocessed_folder, identifier + "_routing.npy")
        if overwrite or not os.path.isfile(save_path):
            to_process.append(identifier)

    if len(to_process) == 0:
        return [join(preprocessed_folder, i + "_routing.npy") for i in identifiers]

    maybe_mkdir_p(preprocessed_folder)

    worker = partial(_convert_one_case, context=context)
    if num_processes is None or num_processes <= 1:
        generated = [worker(i) for i in to_process]
    else:
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            generated = pool.map(worker, to_process)

    return generated


def ensure_routing_sidecars(
    dataset_name_or_id: str | int,
    configuration: str = "3d_fullres",
    plans_identifier: str = "APRMPlans",
    anatomy_split: str = "Tr",
    overwrite: bool = False,
    num_processes: int = 0,
) -> list[str]:
    return prepare_routing_sidecars(
        dataset_name_or_id=dataset_name_or_id,
        configuration=configuration,
        plans_identifier=plans_identifier,
        anatomy_split=anatomy_split,
        overwrite=overwrite,
        num_processes=num_processes,
        identifiers=None,
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert APRM_raw/<dataset>/anatomyTr labelmaps into <identifier>_routing.npy sidecars "
        "inside the APRM_preprocessed configuration folder."
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="Dataset name or numeric ID, for example Dataset020_NSCLC or 20"
    )
    parser.add_argument("-c", "--configuration", default="3d_fullres")
    parser.add_argument("-p", "--plans", default="APRMPlans")
    parser.add_argument("--anatomy_split", default="Tr", choices=["Tr"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--np", type=int, default=0, help="Number of worker processes. 0 or 1 means single process.")
    parser.add_argument("--identifiers", nargs="*", default=None, help="Optional subset of case identifiers to convert")
    return parser


def prepare_routing_sidecars_entry() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    generated = prepare_routing_sidecars(
        dataset_name_or_id=args.dataset,
        configuration=args.configuration,
        plans_identifier=args.plans,
        anatomy_split=args.anatomy_split,
        overwrite=args.overwrite,
        num_processes=args.np,
        identifiers=args.identifiers,
    )
    print(f"[DONE] prepared {len(generated)} routing sidecars")
