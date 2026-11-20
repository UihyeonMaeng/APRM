import itertools
import multiprocessing
import os
from copy import deepcopy
from queue import Queue, Full
from threading import Thread, Event
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import (
    load_json,
    join,
    isfile,
    maybe_mkdir_p,
    isdir,
    subdirs,
    save_json,
)
from torch import nn
from torch._dynamo import OptimizedModule
from tqdm import tqdm

import aprm
from aprm.configuration import default_num_processes
from aprm.anatomy.alignment import append_anatomy_channels
from aprm.anatomy.labels import NUM_ANATOMY_CLASSES
from aprm.inference.data_iterators import (
    PreprocessAdapterFromNpy,
    preprocessing_iterator_fromfiles,
    preprocessing_iterator_fromnpy,
)
from aprm.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from aprm.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from aprm.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from aprm.utilities.find_class_by_name import recursive_find_python_class
from aprm.utilities.helpers import empty_cache, dummy_context
from aprm.utilities.json_export import recursive_fix_for_json_export
from aprm.utilities.label_handling.label_handling import determine_num_input_channels
from aprm.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from aprm.utilities.utils import create_lists_from_splitted_dataset_folder


class APRMPredictor(object):
    def __init__(
        self,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_device: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True,
    ):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        (
            self.plans_manager,
            self.configuration_manager,
            self.list_of_parameters,
            self.network,
            self.dataset_json,
            self.trainer_name,
            self.allowed_mirroring_axes,
            self.label_manager,
        ) = None, None, None, None, None, None, None, None

        if not 0 < tile_step_size <= 1:
            raise ValueError("tile_step_size must be in (0, 1].")
        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        else:
            print(f"perform_everything_on_device=True is only supported for cuda devices! Setting this to False")
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = APRMPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        if not use_folds:
            raise FileNotFoundError("No trained folds were found in the model folder.")
        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != "all" else f
            checkpoint = torch.load(
                join(model_training_output_dir, f"fold_{f}", checkpoint_name),
                map_location=torch.device("cpu"),
                weights_only=False,
            )
            if i == 0:
                trainer_name = checkpoint["trainer_name"]
                configuration_name = checkpoint["init_args"]["configuration"]
                inference_allowed_mirroring_axes = (
                    checkpoint["inference_allowed_mirroring_axes"]
                    if "inference_allowed_mirroring_axes" in checkpoint.keys()
                    else None
                )

            if (
                checkpoint["trainer_name"] != trainer_name
                or checkpoint["init_args"]["configuration"] != configuration_name
            ):
                raise ValueError("All folds must use the same APRM trainer and configuration.")
            parameters.append(checkpoint["network_weights"])

        if trainer_name != "APRMTrainer" or configuration_name != "3d_fullres":
            raise ValueError(
                "This release loads APRMTrainer checkpoints for 3d_fullres only. "
                "Legacy checkpoints require explicit migration and verification."
            )
        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(
            join(aprm.__path__[0], "training", "trainers"), trainer_name, "aprm.training.trainers"
        )
        if trainer_class is None:
            raise RuntimeError(
                f"Unable to locate trainer class {trainer_name} in aprm.training.trainers. "
                f"Please place it there (in any .py file)!"
            )
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False,
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.network = network

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        # Dynamic sparse dispatch and per-stage routing statistics require eager execution.
        if os.environ.get("APRM_compile", "").lower() in ("true", "1", "t"):
            print("APRM uses eager execution; APRM_compile is ignored for inference.")

    def manual_initialization(
        self,
        network: nn.Module,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        parameters: Optional[List[dict]],
        dataset_json: dict,
        trainer_name: str,
        inference_allowed_mirroring_axes: Optional[Tuple[int, ...]],
    ):
        """
        Initialize from an APRM training network for final validation.
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if isinstance(self.network, OptimizedModule):
            self.network = self.network._orig_mod

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print("use_folds is None, attempting to auto detect available folds")
        fold_folders = subdirs(model_training_output_dir, prefix="fold_", join=False)
        fold_folders = [i for i in fold_folders if i != "fold_all"]
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split("_")[-1]) for i in fold_folders]
        print(f"found the following folds: {use_folds}")
        return sorted(use_folds)

    def _manage_input_and_output_lists(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
        folder_with_segs_from_prev_stage: str = None,
        overwrite: bool = True,
        part_id: int = 0,
        num_parts: int = 1,
        save_probabilities: bool = False,
    ):
        if num_parts < 1 or not 0 <= part_id < num_parts:
            raise ValueError("Require num_parts >= 1 and 0 <= part_id < num_parts.")
        if folder_with_segs_from_prev_stage is not None:
            raise ValueError("This APRM release supports 3d_fullres without cascade inputs.")
        if isinstance(list_of_lists_or_source_folder, (str, os.PathLike)):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
                str(list_of_lists_or_source_folder), self.dataset_json["file_ending"]
            )
        inputs = [[os.fspath(filename) for filename in files] for files in list_of_lists_or_source_folder]
        if isinstance(output_folder_or_list_of_truncated_output_files, list):
            if len(output_folder_or_list_of_truncated_output_files) != len(inputs):
                raise ValueError("The output list must contain one name per input case before partitioning.")
        inputs = inputs[part_id::num_parts]
        caseids = [self._case_id(files) for files in inputs]
        if len(set(caseids)) != len(caseids):
            raise ValueError("Input case identifiers must be unique.")
        if isinstance(output_folder_or_list_of_truncated_output_files, (str, os.PathLike)):
            outputs = [join(str(output_folder_or_list_of_truncated_output_files), case) for case in caseids]
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            outputs = [os.fspath(name) for name in output_folder_or_list_of_truncated_output_files[part_id::num_parts]]
        else:
            outputs = None
        if outputs is not None and len(set(outputs)) != len(outputs):
            raise ValueError("Output filenames must be unique.")
        if not overwrite and outputs is not None:
            keep = [
                index
                for index, name in enumerate(outputs)
                if not (
                    isfile(name + self.dataset_json["file_ending"])
                    and (not save_probabilities or isfile(name + ".npz"))
                )
            ]
            inputs, outputs = [inputs[index] for index in keep], [outputs[index] for index in keep]
        return inputs, outputs, [None] * len(inputs)

    def _case_id(self, files: List[str]) -> str:
        if not files:
            raise ValueError("Each case must contain its original image channels.")
        if len(files) != len(self.dataset_json["channel_names"]):
            raise ValueError("Image channel count does not match dataset.json.")
        ending = self.dataset_json["file_ending"]
        case_id = None
        for index, filename in enumerate(files):
            name = os.path.basename(filename)
            suffix = f"_{index:04d}{ending}"
            if not name.endswith(suffix) or len(name) <= len(suffix):
                raise ValueError(f"Expected case_{index:04d}{ending}, got {name}.")
            current_id = name[: -len(suffix)]
            if case_id is not None and current_id != case_id:
                raise ValueError("Image channels within one case have different case identifiers.")
            case_id = current_id
        return case_id

    def _resolve_anatomy_files(self, input_lists: List[List[str]], anatomy_dir: str):
        if anatomy_dir is None:
            raise ValueError(
                "APRM prediction requires anatomy_dir containing case.nii.gz priors. "
                "Use --anatomy_dir with APRM_predict."
            )
        if not os.path.isdir(anatomy_dir):
            raise FileNotFoundError(f"Anatomical prior directory does not exist: {anatomy_dir}")
        # Resolve from original source IDs after filtering; output aliases are unrelated to prior names.
        files = [
            os.path.join(anatomy_dir, self._case_id(images) + self.dataset_json["file_ending"])
            for images in input_lists
        ]
        missing = [filename for filename in files if not os.path.isfile(filename)]
        if missing:
            raise FileNotFoundError("Missing anatomical prior(s): " + ", ".join(missing))
        return files

    def _save_prediction_metadata(self, outputs, arguments):
        if outputs is None:
            return
        directories = sorted({os.path.dirname(os.path.abspath(name)) for name in outputs})
        arguments = deepcopy(arguments)
        recursive_fix_for_json_export(arguments)
        for directory in directories:
            maybe_mkdir_p(directory)
            save_json(arguments, join(directory, "predict_from_raw_data_args.json"))
            save_json(self.dataset_json, join(directory, "dataset.json"), sort_keys=False)
            save_json(self.plans_manager.plans, join(directory, "plans.json"), sort_keys=False)

    def predict_from_files(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        num_processes_preprocessing: int = default_num_processes,
        num_processes_segmentation_export: int = default_num_processes,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
        anatomy_dir: str = None,
    ):
        """Predict original CT files with one co-registered anatomical prior per case.

        For an image ``case_0000.nii.gz``, anatomy_dir must contain ``case.nii.gz``
        with labels 0..6. Geometry is checked before the prior is resampled.
        """
        if num_processes_preprocessing < 1 or num_processes_segmentation_export < 1:
            raise ValueError(
                "Worker counts must be positive. For sequential prediction use "
                "predict_from_files_sequential (CLI: -npp 0 -nps 0)."
            )
        inputs, outputs, previous = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
        )
        anatomy_files = self._resolve_anatomy_files(inputs, anatomy_dir)
        if not inputs:
            return []
        self._save_prediction_metadata(
            outputs,
            {
                "input_files": inputs,
                "output_files": outputs,
                "anatomy_files": anatomy_files,
                "tile_step_size": self.tile_step_size,
                "use_mirroring": self.use_mirroring,
                "save_probabilities": save_probabilities,
                "num_parts": num_parts,
                "part_id": part_id,
            },
        )
        iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            inputs, previous, outputs, num_processes_preprocessing, anatomy_files
        )
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(
        self,
        input_list_of_lists: List[List[str]],
        seg_from_prev_stage_files: Union[List[str], None],
        output_filenames_truncated: Union[List[str], None],
        num_processes: int,
        anatomy_filenames: List[str] = None,
    ):
        return preprocessing_iterator_fromfiles(
            input_list_of_lists,
            seg_from_prev_stage_files,
            output_filenames_truncated,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == "cuda",
            self.verbose_preprocessing,
            anatomy_filenames=anatomy_filenames,
        )

    def get_data_iterator_from_raw_npy_data(
        self,
        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
        segs_from_prev_stage_or_list_of_segs_from_prev_stage,
        properties_or_list_of_properties: Union[dict, List[dict]],
        truncated_ofname: Union[str, List[str], None],
        num_processes: int = 3,
        anatomy_or_list_of_anatomy=None,
        anatomy_properties_or_list_of_properties=None,
    ):
        """Preprocess raw arrays and their anatomical maps on the original image grid.

        Supply image and anatomy properties from the same reader specified in plans.json.
        Anatomical arrays have shape (D,H,W) or (1,D,H,W), with categorical labels 0..6.
        """
        if anatomy_or_list_of_anatomy is None or anatomy_properties_or_list_of_properties is None:
            raise ValueError(
                "Raw NumPy prediction requires anatomy_or_list_of_anatomy and "
                "anatomy_properties_or_list_of_properties. For files use predict_from_files "
                "with anatomy_dir."
            )
        images = image_or_list_of_images if isinstance(image_or_list_of_images, list) else [image_or_list_of_images]
        props = (
            properties_or_list_of_properties
            if isinstance(properties_or_list_of_properties, list)
            else [properties_or_list_of_properties]
        )
        anatomy = (
            anatomy_or_list_of_anatomy if isinstance(anatomy_or_list_of_anatomy, list) else [anatomy_or_list_of_anatomy]
        )
        anatomy_props = (
            anatomy_properties_or_list_of_properties
            if isinstance(anatomy_properties_or_list_of_properties, list)
            else [anatomy_properties_or_list_of_properties]
        )
        previous = segs_from_prev_stage_or_list_of_segs_from_prev_stage
        if isinstance(previous, np.ndarray):
            previous = [previous]
        output_files = [truncated_ofname] if isinstance(truncated_ofname, str) else truncated_ofname
        return preprocessing_iterator_fromnpy(
            images,
            previous,
            props,
            output_files,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == "cuda",
            self.verbose_preprocessing,
            list_of_anatomy=anatomy,
            list_of_anatomy_properties=anatomy_props,
        )

    def predict_from_list_of_npy_arrays(
        self,
        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
        segs_from_prev_stage_or_list_of_segs_from_prev_stage,
        properties_or_list_of_properties: Union[dict, List[dict]],
        truncated_ofname: Union[str, List[str], None],
        num_processes: int = 3,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = default_num_processes,
        anatomy_or_list_of_anatomy=None,
        anatomy_properties_or_list_of_properties=None,
    ):
        iterator = self.get_data_iterator_from_raw_npy_data(
            image_or_list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            num_processes,
            anatomy_or_list_of_anatomy,
            anatomy_properties_or_list_of_properties,
        )
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(
        self,
        data_iterator,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = default_num_processes,
    ):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        if num_processes_segmentation_export < 1:
            raise ValueError("num_processes_segmentation_export must be positive.")
        try:
            with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
                worker_list = [i for i in export_pool._pool]
                r = []
                for preprocessed in data_iterator:
                    data = preprocessed["data"]
                    if isinstance(data, str):
                        data = torch.from_numpy(np.load(data, allow_pickle=False))

                    ofile = preprocessed["ofile"]
                    if ofile is not None:
                        maybe_mkdir_p(os.path.dirname(os.path.abspath(ofile)))
                    if ofile is not None:
                        print(f"\nPredicting {os.path.basename(ofile)}:")
                    else:
                        print(f"\nPredicting image of shape {data.shape}:")

                    print(f"perform_everything_on_device: {self.perform_everything_on_device}")

                    properties = preprocessed["data_properties"]

                    # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                    # npy files
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                    while not proceed:
                        sleep(0.1)
                        proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                    # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                    prediction = self.predict_logits_from_preprocessed_data(data).cpu().detach().numpy()

                    if ofile is not None:
                        print("sending off prediction to background worker for resampling and export")
                        r.append(
                            export_pool.starmap_async(
                                export_prediction_from_logits,
                                (
                                    (
                                        prediction,
                                        properties,
                                        self.configuration_manager,
                                        self.plans_manager,
                                        self.dataset_json,
                                        ofile,
                                        save_probabilities,
                                    ),
                                ),
                            )
                        )
                    else:
                        print("sending off prediction to background worker for resampling")
                        r.append(
                            export_pool.starmap_async(
                                convert_predicted_logits_to_segmentation_with_correct_shape,
                                (
                                    (
                                        prediction,
                                        self.plans_manager,
                                        self.configuration_manager,
                                        self.label_manager,
                                        properties,
                                        save_probabilities,
                                    ),
                                ),
                            )
                        )
                    if ofile is not None:
                        print(f"done with {os.path.basename(ofile)}")
                    else:
                        print(f"\nDone with image of shape {data.shape}:")
                ret = [i.get()[0] for i in r]

            return ret
        finally:
            if hasattr(data_iterator, "close"):
                data_iterator.close()
            if isinstance(data_iterator, MultiThreadedAugmenter):
                data_iterator._finish()
            compute_gaussian.cache_clear()
            empty_cache(self.device)

    def predict_single_npy_array(
        self,
        input_image: np.ndarray,
        image_properties: dict,
        segmentation_previous_stage: np.ndarray = None,
        output_file_truncated: str = None,
        save_or_return_probabilities: bool = False,
        anatomy: np.ndarray = None,
        anatomy_properties: dict = None,
    ):
        """Predict one raw image and anatomical map using their original physical grid.

        Use the image reader named in plans.json for both inputs and pass both returned
        property dictionaries. Do not pre-normalize CT or append anatomy channels yourself.
        """
        if anatomy is None or anatomy_properties is None:
            raise ValueError(
                "NumPy prediction requires anatomy and anatomy_properties. "
                "For image files use predict_from_files(..., anatomy_dir=...)."
            )
        adapter = PreprocessAdapterFromNpy(
            [input_image],
            [segmentation_previous_stage],
            [image_properties],
            [output_file_truncated],
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_threads_in_multithreaded=1,
            verbose=self.verbose_preprocessing,
            list_of_anatomy=[anatomy],
            list_of_anatomy_properties=[anatomy_properties],
        )
        case = next(adapter)
        logits = self.predict_logits_from_preprocessed_data(case["data"]).cpu()
        if output_file_truncated is not None:
            maybe_mkdir_p(os.path.dirname(os.path.abspath(output_file_truncated)))
            return export_prediction_from_logits(
                logits,
                case["data_properties"],
                self.configuration_manager,
                self.plans_manager,
                self.dataset_json,
                output_file_truncated,
                save_or_return_probabilities,
            )
        return convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            self.plans_manager,
            self.configuration_manager,
            self.label_manager,
            case["data_properties"],
            return_probabilities=save_or_return_probabilities,
        )

    @torch.inference_mode()
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """Predict a preprocessed CT with categorical or one-hot anatomy.

        Append either one categorical label channel (0..6) or seven one-hot channels.
        The network removes these routing channels before the image encoder.
        Returned logits remain on the preprocessed grid; export restores image geometry.
        """
        image_channels = len(self.dataset_json["channel_names"])
        expected_channels = (image_channels + 1, image_channels + NUM_ANATOMY_CLASSES)
        if data.ndim != 4 or data.shape[0] not in expected_channels:
            raise ValueError(
                f"Expected (C+1,D,H,W) or (C+7,D,H,W), with {expected_channels} total channels. "
                "Append categorical anatomy or use an anatomy-aware raw input API."
            )
        n_threads = torch.get_num_threads()
        torch.set_num_threads(min(default_num_processes, n_threads))
        try:
            if not self.list_of_parameters:
                # During training validation, manual_initialization supplies the live network.
                return self.predict_sliding_window_return_logits(data).cpu()
            prediction = None
            for parameters in self.list_of_parameters:
                self.network.load_state_dict(parameters)
                fold_prediction = self.predict_sliding_window_return_logits(data).cpu()
                prediction = fold_prediction if prediction is None else prediction + fold_prediction
            prediction /= len(self.list_of_parameters)
            return prediction
        finally:
            torch.set_num_threads(n_threads)

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(image_size) - 1, (
                "if tile_size has less entries than image_size, "
                "len(tile_size) "
                "must be one shorter than len(image_size) "
                "(only dimension "
                "discrepancy of 1 allowed)."
            )
            steps = compute_steps_for_sliding_window(
                image_size[1:], self.configuration_manager.patch_size, self.tile_step_size
            )
            if self.verbose:
                print(
                    f"n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is"
                    f" {image_size}, tile_size {self.configuration_manager.patch_size}, "
                    f"tile_step_size {self.tile_step_size}\nsteps:\n{steps}"
                )
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple(
                                [
                                    slice(None),
                                    d,
                                    *[
                                        slice(si, si + ti)
                                        for si, ti in zip((sx, sy), self.configuration_manager.patch_size)
                                    ],
                                ]
                            )
                        )
        else:
            steps = compute_steps_for_sliding_window(
                image_size, self.configuration_manager.patch_size, self.tile_step_size
            )
            if self.verbose:
                print(
                    f"n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, "
                    f"tile_step_size {self.tile_step_size}\nsteps:\n{steps}"
                )
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple(
                                [
                                    slice(None),
                                    *[
                                        slice(si, si + ti)
                                        for si, ti in zip((sx, sy, sz), self.configuration_manager.patch_size)
                                    ],
                                ]
                            )
                        )
        return slicers

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, "mirror_axes does not match the dimension of the input!"

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= len(axes_combinations) + 1
        return prediction

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        slicers,
        do_on_device: bool = True,
    ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device("cpu")

        stop_event = Event()
        worker = None

        def enqueue(q, item):
            while not stop_event.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return
                except Full:
                    continue

        def producer(d, slh, q):
            try:
                for s in slh:
                    if stop_event.is_set():
                        return
                    enqueue(q, (torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
                enqueue(q, None)
            except Exception as error:
                enqueue(q, error)

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f"move image to device {results_device}")
            data = data.to(results_device)
            queue = Queue(maxsize=2)
            worker = Thread(target=producer, args=(data, slicers, queue), daemon=True)
            worker.start()

            # preallocate arrays
            if self.verbose:
                print(f"preallocating results arrays on device {results_device}")
            predicted_logits = torch.zeros(
                (self.label_manager.num_segmentation_heads, *data.shape[1:]), dtype=torch.half, device=results_device
            )
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(
                    tuple(self.configuration_manager.patch_size),
                    sigma_scale=1.0 / 8,
                    value_scaling_factor=10,
                    device=results_device,
                )
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f"running prediction: {len(slicers)} steps")

            with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                while True:
                    item = queue.get()
                    if item is None:
                        queue.task_done()
                        break
                    if isinstance(item, Exception):
                        queue.task_done()
                        raise item
                    workon, sl = item
                    prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                    if self.use_gaussian:
                        prediction *= gaussian
                    predicted_logits[sl] += prediction
                    n_predictions[sl[1:]] += gaussian
                    queue.task_done()
                    pbar.update()
            queue.join()

            torch.div(predicted_logits, n_predictions, out=predicted_logits)
            # Reject numerical overflow or invalid aggregation.
            if not torch.isfinite(predicted_logits).all():
                raise RuntimeError(
                    "Encountered non-finite values in predicted array. Aborting... If this problem persists, "
                    "reduce value_scaling_factor in compute_gaussian or increase the dtype of "
                    "predicted_logits to fp32"
                )
        except Exception:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise
        finally:
            stop_event.set()
            if worker is not None:
                worker.join(timeout=1)
        return predicted_logits

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # CUDA autocast avoids unsupported/slow mixed-precision paths on other devices.
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            assert input_image.ndim == 4, "input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)"

            if self.verbose:
                print(f"Input shape: {input_image.shape}")
                print("step_size:", self.tile_step_size)
                print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(
                input_image, self.configuration_manager.patch_size, "constant", {"value": 0}, True, None
            )

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            if self.perform_everything_on_device and self.device.type == "cuda":
                # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                try:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(
                        data, slicers, self.perform_everything_on_device
                    )
                except torch.cuda.OutOfMemoryError:
                    print(
                        "Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU"
                    )
                    empty_cache(self.device)
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
            else:
                predicted_logits = self._internal_predict_sliding_window_return_logits(
                    data, slicers, self.perform_everything_on_device
                )

            empty_cache(self.device)
            # revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def predict_from_files_sequential(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        folder_with_segs_from_prev_stage: str = None,
        anatomy_dir: str = None,
        num_parts: int = 1,
        part_id: int = 0,
    ):
        """Predict files and aligned anatomy without preprocessing/export subprocesses."""
        inputs, outputs, _ = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
        )
        anatomy_files = self._resolve_anatomy_files(inputs, anatomy_dir)
        if not inputs:
            return []
        self._save_prediction_metadata(
            outputs,
            {
                "input_files": inputs,
                "output_files": outputs,
                "anatomy_files": anatomy_files,
                "tile_step_size": self.tile_step_size,
                "use_mirroring": self.use_mirroring,
                "save_probabilities": save_probabilities,
                "num_parts": num_parts,
                "part_id": part_id,
            },
        )
        outputs = outputs if outputs is not None else [None] * len(inputs)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        results = []
        try:
            for images, output_file, anatomy_file in zip(inputs, outputs, anatomy_files):
                data, _, properties = preprocessor.run_case(
                    images, None, self.plans_manager, self.configuration_manager, self.dataset_json
                )
                data = append_anatomy_channels(
                    data, anatomy_file, properties, self.plans_manager, self.configuration_manager
                )
                logits = self.predict_logits_from_preprocessed_data(
                    torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
                ).cpu()
                if output_file is not None:
                    export_prediction_from_logits(
                        logits,
                        properties,
                        self.configuration_manager,
                        self.plans_manager,
                        self.dataset_json,
                        output_file,
                        save_probabilities,
                    )
                    results.append(None)
                else:
                    results.append(
                        convert_predicted_logits_to_segmentation_with_correct_shape(
                            logits,
                            self.plans_manager,
                            self.configuration_manager,
                            self.label_manager,
                            properties,
                            save_probabilities,
                        )
                    )
            return results
        finally:
            compute_gaussian.cache_clear()
            empty_cache(self.device)


def _environment_default(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _prediction_parser(model_folder: bool):
    import argparse

    parser = argparse.ArgumentParser(description="Predict 3D lung tumors with APRM and anatomical priors.")
    parser.add_argument("-i", required=True, help="Image folder containing case_0000.nii.gz files.")
    parser.add_argument("-o", required=True, help="Output segmentation folder.")
    parser.add_argument("--anatomy_dir", required=True, help="Co-registered 0..6 anatomical maps named case.nii.gz.")
    if model_folder:
        parser.add_argument("-m", required=True, help="Trained model folder containing fold_X subfolders.")
    else:
        parser.add_argument("-d", required=True, help="Dataset name or numeric ID.")
        parser.add_argument("-p", default="APRMPlans", help="Plans identifier (default: APRMPlans).")
        parser.add_argument("-tr", choices=["APRMTrainer"], default="APRMTrainer")
        parser.add_argument("-c", choices=["3d_fullres"], default="3d_fullres")
    parser.add_argument("-f", nargs="+", default=["0"], help="Trained fold(s), or all. Default: 0.")
    parser.add_argument("-chk", default="checkpoint_final.pth", help="Checkpoint filename.")
    parser.add_argument("-step_size", type=float, default=0.5, help="Sliding-window step fraction (default: 0.5).")
    parser.add_argument("--disable_tta", action="store_true", help="Disable mirroring during inference.")
    parser.add_argument("--save_probabilities", action="store_true", help="Also export class probabilities.")
    parser.add_argument("--continue_prediction", action="store_true", help="Skip complete existing predictions.")
    parser.add_argument(
        "-npp",
        type=int,
        default=_environment_default("APRM_npp", 3),
        help="Preprocessing processes; use -npp 0 -nps 0 for sequential inference.",
    )
    parser.add_argument(
        "-nps",
        type=int,
        default=_environment_default("APRM_nps", 3),
        help="Export processes; use -npp 0 -nps 0 for sequential inference.",
    )
    parser.add_argument("-num_parts", type=int, default=1, help="Number of independent input partitions.")
    parser.add_argument("-part_id", type=int, default=0, help="Zero-based partition index.")
    parser.add_argument(
        "-device",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Inference device. Use CUDA_VISIBLE_DEVICES to select a GPU.",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    return parser


def _run_prediction_cli(model_folder: bool):
    parser = _prediction_parser(model_folder)
    args = parser.parse_args()
    if args.num_parts < 1 or not 0 <= args.part_id < args.num_parts:
        parser.error("Require num_parts >= 1 and 0 <= part_id < num_parts.")
    if args.npp < 0 or args.nps < 0 or ((args.npp == 0) != (args.nps == 0)):
        parser.error("Use positive worker counts, or set both -npp 0 and -nps 0.")
    try:
        folds = [fold if fold == "all" else int(fold) for fold in args.f]
    except ValueError:
        parser.error("Folds must be integers or all.")
    training_folder = args.m if model_folder else get_output_folder(args.d, args.tr, args.p, args.c)
    if args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    print("APRM: Anatomical Prior-Guided Routing for 3D Lung Tumor Segmentation (ACCV 2026).")
    print("Please cite APRM and its nnU-Net foundation; citation details are in README.md.")
    predictor = APRMPredictor(
        tile_step_size=args.step_size,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=True,
        device=torch.device(args.device),
        verbose=args.verbose,
        verbose_preprocessing=args.verbose,
        allow_tqdm=not args.disable_progress_bar,
    )
    predictor.initialize_from_trained_model_folder(training_folder, folds, args.chk)
    common = dict(
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        anatomy_dir=args.anatomy_dir,
        num_parts=args.num_parts,
        part_id=args.part_id,
    )
    if args.npp == 0:
        predictor.predict_from_files_sequential(args.i, args.o, **common)
    else:
        predictor.predict_from_files(
            args.i, args.o, num_processes_preprocessing=args.npp, num_processes_segmentation_export=args.nps, **common
        )


def predict_entry_point_modelfolder():
    _run_prediction_cli(model_folder=True)


def predict_entry_point():
    _run_prediction_cli(model_folder=False)


if __name__ == "__main__":
    predict_entry_point()
