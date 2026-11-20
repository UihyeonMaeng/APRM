"""Anatomy-aware preprocessing iterators for APRM inference.

The image is normalized and resampled first. Anatomical labels are then transformed
using that image's transpose, crop and target grid and appended as seven channels.
The routing channels never participate in CT intensity normalization.
"""

import multiprocessing
import os
import queue
import traceback
from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from aprm.anatomy.alignment import align_anatomy, append_anatomy_channels
from aprm.anatomy.labels import NUM_ANATOMY_CLASSES
from aprm.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from aprm.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def _check_cases(images, previous_stage, output_files, anatomy, anatomy_properties=None):
    """Validate complete inputs before starting preprocessing workers."""
    count = len(images)
    if anatomy is None or len(anatomy) != count or any(a is None for a in anatomy):
        raise ValueError(
            "Provide one anatomical prior per image. File prediction requires anatomy_dir; "
            "NumPy prediction requires anatomy and anatomy_properties."
        )
    if previous_stage is not None:
        if len(previous_stage) != count:
            raise ValueError("Previous-stage inputs must match the number of images.")
        if any(s is not None for s in previous_stage):
            raise ValueError("This APRM release supports 3d_fullres only, without a cascade.")
    if output_files is not None and len(output_files) != count:
        raise ValueError("Output filenames must match the number of images.")
    if anatomy_properties is not None and len(anatomy_properties) != count:
        raise ValueError("Anatomy geometry metadata must match the number of images.")


def _as_tensor(data):
    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))


def _preprocess_file_case(
    files, anatomy_filename, output_file, preprocessor, plans_manager, configuration_manager, dataset_json
):
    data, _, properties = preprocessor.run_case(files, None, plans_manager, configuration_manager, dataset_json)
    data = append_anatomy_channels(data, anatomy_filename, properties, plans_manager, configuration_manager)
    return {"data": _as_tensor(data), "data_properties": properties, "ofile": output_file}


def _preprocess_numpy_case(
    image,
    properties,
    anatomy,
    anatomy_properties,
    output_file,
    preprocessor,
    plans_manager,
    configuration_manager,
    dataset_json,
):
    if anatomy is None or anatomy_properties is None:
        raise ValueError("NumPy inference requires anatomy and anatomy_properties on the original image grid.")
    if not isinstance(image, np.ndarray) or image.ndim != 4:
        raise ValueError("Input images must be NumPy arrays with shape (channels, depth, height, width).")
    if image.shape[0] != len(dataset_json["channel_names"]):
        raise ValueError("Pass only the original CT image channels; anatomical channels are appended internally.")
    data, _, properties = preprocessor.run_case_npy(
        image, None, deepcopy(properties), plans_manager, configuration_manager, dataset_json
    )
    aligned = align_anatomy(
        anatomy, anatomy_properties, properties, data.shape[1:], plans_manager, configuration_manager
    )
    one_hot = (aligned[None] == np.arange(NUM_ANATOMY_CLASSES)[:, None, None, None]).astype(np.float32)
    data = np.concatenate((data, one_hot), axis=0)
    return {"data": _as_tensor(data), "data_properties": properties, "ofile": output_file}


def _preprocessing_worker(
    cases,
    mode,
    plans_manager,
    dataset_json,
    configuration_manager,
    output_queue,
    error_queue,
    done_event,
    abort_event,
    verbose,
):
    try:
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        for case in cases:
            if abort_event.is_set():
                return
            if mode == "files":
                item = _preprocess_file_case(*case, preprocessor, plans_manager, configuration_manager, dataset_json)
            else:
                item = _preprocess_numpy_case(*case, preprocessor, plans_manager, configuration_manager, dataset_json)
            # Pipe-backed NumPy transfer avoids tensor storage handles and their
            # extra resource-sharing server. The parent restores the same tensor.
            item["data"] = item["data"].numpy()
            while not abort_event.is_set():
                try:
                    output_queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
    except Exception:
        error_queue.put(traceback.format_exc())
        abort_event.set()
    finally:
        done_event.set()


def _preprocessing_iterator(
    cases, mode, plans_manager, dataset_json, configuration_manager, num_processes, pin_memory, verbose
):
    if not cases:
        return
    if num_processes < 1:
        raise ValueError("num_processes must be positive. Use predict_from_files_sequential for no multiprocessing.")
    context = multiprocessing.get_context("spawn")
    num_processes = min(num_processes, len(cases))
    abort_event = context.Event()
    error_queue = context.Queue()
    processes, queues, done_events = [], [], []
    try:
        for index in range(num_processes):
            output_queue = context.Queue(maxsize=1)
            done = context.Event()
            process = context.Process(
                target=_preprocessing_worker,
                args=(
                    cases[index::num_processes],
                    mode,
                    plans_manager,
                    dataset_json,
                    configuration_manager,
                    output_queue,
                    error_queue,
                    done,
                    abort_event,
                    verbose,
                ),
                daemon=True,
            )
            process.start()
            processes.append(process)
            queues.append(output_queue)
            done_events.append(done)

        # Round-robin consumption preserves case/output order without relying on Queue.empty().
        for case_index in range(len(cases)):
            worker_index = case_index % num_processes
            while True:
                if abort_event.is_set():
                    detail = error_queue.get(timeout=1)
                    raise RuntimeError(f"Anatomical/image preprocessing failed:\n{detail}")
                try:
                    item = queues[worker_index].get(timeout=0.1)
                    break
                except queue.Empty:
                    if any(not p.is_alive() and not event.is_set() for p, event in zip(processes, done_events)):
                        raise RuntimeError(
                            "A preprocessing worker exited unexpectedly. Reduce the number "
                            "of preprocessing processes if memory was exhausted."
                        )
                    if done_events[worker_index].is_set() and not processes[worker_index].is_alive():
                        if abort_event.is_set():
                            continue
                        raise RuntimeError("A preprocessing worker finished without returning its case.")
            item["data"] = _as_tensor(item["data"])
            if pin_memory:
                item["data"] = item["data"].pin_memory()
            yield item
    finally:
        abort_event.set()
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
        for pipe_queue in [*queues, error_queue]:
            pipe_queue.close()
            pipe_queue.join_thread()


def preprocessing_iterator_fromfiles(
    list_of_lists: List[List[str]],
    list_of_segs_from_prev_stage_files: Union[None, List[str]],
    output_filenames_truncated: Union[None, List[str]],
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_processes: int,
    pin_memory: bool = False,
    verbose: bool = False,
    anatomy_filenames: List[str] = None,
):
    _check_cases(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated, anatomy_filenames)
    missing = [name for name in anatomy_filenames if not os.path.isfile(name)]
    if missing:
        raise FileNotFoundError("Missing anatomical prior(s): " + ", ".join(map(str, missing)))
    output_files = output_filenames_truncated or [None] * len(list_of_lists)
    cases = list(zip(list_of_lists, anatomy_filenames, output_files))
    yield from _preprocessing_iterator(
        cases, "files", plans_manager, dataset_json, configuration_manager, num_processes, pin_memory, verbose
    )


class PreprocessAdapter(DataLoader):
    def __init__(
        self,
        list_of_lists: List[List[str]],
        list_of_segs_from_prev_stage_files: Union[None, List[str]],
        preprocessor: DefaultPreprocessor,
        output_filenames_truncated: Union[None, List[str]],
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_threads_in_multithreaded: int = 1,
        anatomy_filenames: List[str] = None,
    ):
        _check_cases(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated, anatomy_filenames)
        self.preprocessor = preprocessor
        self.plans_manager, self.configuration_manager, self.dataset_json = (
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        output_files = output_filenames_truncated or [None] * len(list_of_lists)
        super().__init__(
            list(zip(list_of_lists, anatomy_filenames, output_files)),
            1,
            num_threads_in_multithreaded,
            seed_for_shuffle=1,
            return_incomplete=True,
            shuffle=False,
            infinite=False,
            sampling_probabilities=None,
        )
        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        return _preprocess_file_case(
            *self._data[self.get_indices()[0]],
            self.preprocessor,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )


class PreprocessAdapterFromNpy(DataLoader):
    def __init__(
        self,
        list_of_images: List[np.ndarray],
        list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
        list_of_image_properties: List[dict],
        truncated_ofnames: Union[List[str], None],
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_threads_in_multithreaded: int = 1,
        verbose: bool = False,
        list_of_anatomy: List[np.ndarray] = None,
        list_of_anatomy_properties: List[dict] = None,
    ):
        _check_cases(
            list_of_images, list_of_segs_from_prev_stage, truncated_ofnames, list_of_anatomy, list_of_anatomy_properties
        )
        if list_of_anatomy_properties is None or any(p is None for p in list_of_anatomy_properties):
            raise ValueError("Provide anatomy_properties for each raw anatomical label map.")
        if len(list_of_image_properties) != len(list_of_images):
            raise ValueError("Image metadata must match the number of images.")
        self.preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        self.plans_manager, self.configuration_manager, self.dataset_json = (
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        output_files = truncated_ofnames or [None] * len(list_of_images)
        cases = list(
            zip(list_of_images, list_of_image_properties, list_of_anatomy, list_of_anatomy_properties, output_files)
        )
        super().__init__(
            cases,
            1,
            num_threads_in_multithreaded,
            seed_for_shuffle=1,
            return_incomplete=True,
            shuffle=False,
            infinite=False,
            sampling_probabilities=None,
        )
        self.indices = list(range(len(list_of_images)))

    def generate_train_batch(self):
        return _preprocess_numpy_case(
            *self._data[self.get_indices()[0]],
            self.preprocessor,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )


def preprocessing_iterator_fromnpy(
    list_of_images: List[np.ndarray],
    list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
    list_of_image_properties: List[dict],
    truncated_ofnames: Union[List[str], None],
    plans_manager: PlansManager,
    dataset_json: dict,
    configuration_manager: ConfigurationManager,
    num_processes: int,
    pin_memory: bool = False,
    verbose: bool = False,
    list_of_anatomy: List[np.ndarray] = None,
    list_of_anatomy_properties: List[dict] = None,
):
    _check_cases(
        list_of_images, list_of_segs_from_prev_stage, truncated_ofnames, list_of_anatomy, list_of_anatomy_properties
    )
    if list_of_anatomy_properties is None or any(p is None for p in list_of_anatomy_properties):
        raise ValueError("Provide anatomy_properties for each raw anatomical label map.")
    if len(list_of_image_properties) != len(list_of_images):
        raise ValueError("Image metadata must match the number of images.")
    output_files = truncated_ofnames or [None] * len(list_of_images)
    cases = list(
        zip(list_of_images, list_of_image_properties, list_of_anatomy, list_of_anatomy_properties, output_files)
    )
    yield from _preprocessing_iterator(
        cases, "numpy", plans_manager, dataset_json, configuration_manager, num_processes, pin_memory, verbose
    )
