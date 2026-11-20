"""Small deterministic CPU fixtures; no dataset or pretrained weights required."""

import numpy as np
import pytest
import torch
from torch import nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from aprm.training.trainers.aprm_modules import APRM
from aprm.utilities.plans_handling.plans_handler import PlansManager


@pytest.fixture(scope="session", autouse=True)
def cpu_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_model():
    torch.manual_seed(17)
    backbone = PlainConvUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=[4, 8, 16],
        conv_op=nn.Conv3d,
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        n_conv_per_stage=[1, 1, 1],
        num_classes=2,
        n_conv_per_stage_decoder=[1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )
    return APRM(backbone, image_input_channels=1)


@pytest.fixture
def preprocessing_setup():
    """Real preprocessing objects with a nonidentity axis permutation."""
    plans = PlansManager(
        {
            "dataset_name": "Dataset001_Test",
            "plans_name": "APRMPlans",
            "image_reader_writer": "SimpleITKIO",
            "transpose_forward": [2, 0, 1],
            "transpose_backward": [1, 2, 0],
            "foreground_intensity_properties_per_channel": {"0": {}},
            "configurations": {
                "3d_fullres": {
                    "data_identifier": "APRMPlans_3d_fullres",
                    "preprocessor_name": "DefaultPreprocessor",
                    "spacing": [1.0, 2.0, 1.5],
                    "patch_size": [8, 8, 8],
                    "batch_size": 1,
                    "normalization_schemes": ["ZScoreNormalization"],
                    "use_mask_for_norm": [False],
                    "resampling_fn_data": "resample_data_or_seg_to_shape",
                    "resampling_fn_data_kwargs": {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": False},
                    "resampling_fn_seg": "resample_data_or_seg_to_shape",
                    "resampling_fn_seg_kwargs": {"is_seg": True, "order": 0, "order_z": 0, "force_separate_z": False},
                    "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                    "resampling_fn_probabilities_kwargs": {
                        "is_seg": False,
                        "order": 1,
                        "order_z": 0,
                        "force_separate_z": False,
                    },
                    "architecture": {
                        "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                        "arch_kwargs": {},
                        "_kw_requires_import": [],
                    },
                    "batch_dice": False,
                }
            },
        }
    )
    config = plans.get_configuration("3d_fullres")
    dataset = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "tumor": 1}, "file_ending": ".nii.gz"}
    return plans, config, dataset


@pytest.fixture
def raw_case(tmp_path):
    """An asymmetric image/prior pair with padding, origin, and anisotropic spacing."""
    import SimpleITK as sitk

    image = np.zeros((10, 12, 14), dtype=np.float32)
    image[1:9, 2:10, 3:11] = np.arange(512, dtype=np.float32).reshape(8, 8, 8) + 1
    anatomy = np.zeros(image.shape, dtype=np.uint8)
    anatomy[1:5, 2:10, 3:11] = 1
    anatomy[5:9, 2:10, 3:11] = 6
    images = tmp_path / "images"
    priors = tmp_path / "anatomy"
    images.mkdir()
    priors.mkdir()
    image_file = images / "subject_0000.nii.gz"
    anatomy_file = priors / "subject.nii.gz"
    for values, filename in ((image, image_file), (anatomy, anatomy_file)):
        itk = sitk.GetImageFromArray(values)
        itk.SetSpacing((1.0, 1.5, 2.0))
        itk.SetOrigin((12.0, -7.0, 3.0))
        sitk.WriteImage(itk, str(filename))
    return image_file, anatomy_file
