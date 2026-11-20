"""Regression tests for prior construction, physical grids, and actual raw inference."""

from copy import deepcopy

import nibabel as nib
import numpy as np
import pytest
import torch

from aprm.anatomy.alignment import align_anatomy, validate_geometry
from aprm.anatomy.labels import CENTRAL_ORGANS, LOBE_FILES, REQUIRED_ROIS
from aprm.anatomy.prepare_labelmaps import prepare_anatomy_folder
from aprm.imageio.simpleitk_reader_writer import SimpleITKIO
from aprm.inference.data_iterators import _preprocess_file_case, _preprocess_numpy_case
from aprm.inference.predict_from_raw_data import APRMPredictor, _prediction_parser
from aprm.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor


def write_masks(tmp_path, omit=None, shifted=None):
    images, output, cache = [tmp_path / name for name in ("images", "priors", "cache")]
    images.mkdir()
    masks = cache / "case_0000"
    masks.mkdir(parents=True)
    affine = np.diag([1.0, 1.5, 3.0, 1.0])
    affine[:3, 3] = [12.0, -8.0, 3.0]
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), affine), images / "case_0000.nii.gz")
    for index, roi in enumerate(REQUIRED_ROIS):
        if roi == omit:
            continue
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask.flat[index] = 1
        mask_affine = affine.copy()
        if roi == shifted:
            mask_affine[0, 3] += 10
        nib.save(nib.Nifti1Image(mask, mask_affine), masks / f"{roi}.nii.gz")
    return images, output, cache, affine


def test_central_prior_includes_all_seven_paper_structures(tmp_path):
    assert set(CENTRAL_ORGANS) == {
        "heart",
        "aorta",
        "trachea",
        "esophagus",
        "pulmonary_vein",
        "superior_vena_cava",
        "inferior_vena_cava",
    }
    images, output, cache, affine = write_masks(tmp_path)
    result = prepare_anatomy_folder(images, output, cache, skip_totalseg=True)
    volume = nib.load(result[0])
    labels = np.asarray(volume.dataobj)
    np.testing.assert_array_equal(labels.ravel()[:5], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(labels.ravel()[5:12], [6] * 7)
    assert labels.ravel()[12:].sum() == 0
    np.testing.assert_allclose(volume.affine, affine)
    assert volume.get_data_dtype() == np.dtype(np.uint8)


def test_incomplete_or_displaced_organ_masks_fail_instead_of_partial_prior(tmp_path):
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    images, output, cache, _ = write_masks(missing_root, omit="superior_vena_cava")
    with pytest.raises(FileNotFoundError, match="superior_vena_cava"):
        prepare_anatomy_folder(images, output, cache, skip_totalseg=True)
    assert not (output / "case.nii.gz").exists()
    shifted_root = tmp_path / "shifted"
    shifted_root.mkdir()
    images, output, cache, _ = write_masks(shifted_root, shifted="aorta")
    with pytest.raises(ValueError, match="grid"):
        prepare_anatomy_folder(images, output, cache, skip_totalseg=True)
    assert not (output / "case.nii.gz").exists()


def test_alignment_uses_image_transpose_crop_and_resampling(raw_case, preprocessing_setup):
    image_file, anatomy_file = raw_case
    plans, config, dataset = preprocessing_setup
    config.configuration["spacing"] = [2.0, 4.0, 3.0]  # Downsample every transposed axis by two.
    image, properties = SimpleITKIO().read_images([str(image_file)])
    labels, anatomy_properties = SimpleITKIO().read_seg(str(anatomy_file))
    data, _, processed_properties = DefaultPreprocessor(False).run_case_npy(
        image, None, deepcopy(properties), plans, config, dataset
    )
    aligned = align_anatomy(labels, anatomy_properties, processed_properties, data.shape[1:], plans, config)
    assert data.shape == (1, 4, 4, 4)
    assert processed_properties["bbox_used_for_cropping"] == [[3, 11], [1, 9], [2, 10]]
    # Original Z is the second axis after [2,0,1] permutation.
    np.testing.assert_array_equal(aligned[:, :2], np.ones((4, 2, 4), dtype=np.int16))
    np.testing.assert_array_equal(aligned[:, 2:], np.full((4, 2, 4), 6, dtype=np.int16))


def test_geometry_rejects_origin_orientation_spacing_and_shape(raw_case, preprocessing_setup):
    image_file, anatomy_file = raw_case
    plans, config, dataset = preprocessing_setup
    image, properties = SimpleITKIO().read_images([str(image_file)])
    labels, anatomy_properties = SimpleITKIO().read_seg(str(anatomy_file))
    validate_geometry(properties, anatomy_properties)
    for key, replacement in (
        ("origin", (100.0, -7.0, 3.0)),
        ("direction", (-1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)),
        ("spacing", (2.0, 1.5, 2.0)),
    ):
        bad = deepcopy(anatomy_properties)
        bad["sitk_stuff"][key] = replacement
        with pytest.raises(ValueError, match="mismatch"):
            validate_geometry(properties, bad)
    data, _, processed = DefaultPreprocessor(False).run_case_npy(
        image, None, deepcopy(properties), plans, config, dataset
    )
    with pytest.raises(ValueError, match="shapes differ"):
        align_anatomy(labels[:, :-1], anatomy_properties, processed, data.shape[1:], plans, config)


def test_numpy_and_file_preprocessing_append_same_aligned_prior(raw_case, preprocessing_setup):
    image_file, anatomy_file = raw_case
    plans, config, dataset = preprocessing_setup
    image, properties = SimpleITKIO().read_images([str(image_file)])
    labels, anatomy_properties = SimpleITKIO().read_seg(str(anatomy_file))
    file_result = _preprocess_file_case(
        [str(image_file)], str(anatomy_file), None, DefaultPreprocessor(False), plans, config, dataset
    )
    numpy_result = _preprocess_numpy_case(
        image, properties, labels, anatomy_properties, None, DefaultPreprocessor(False), plans, config, dataset
    )
    torch.testing.assert_close(file_result["data"], numpy_result["data"], rtol=0, atol=0)
    data = file_result["data"]
    assert data.shape == (8, 8, 8, 8)
    torch.testing.assert_close(data[1:].sum(dim=0), torch.ones((8, 8, 8)), rtol=0, atol=0)
    assert torch.unique(data[1:]).tolist() == [0.0, 1.0]
    assert abs(data[0].mean().item()) < 1e-5
    with pytest.raises(ValueError, match="requires anatomy"):
        _preprocess_numpy_case(
            image, properties, None, anatomy_properties, None, DefaultPreprocessor(False), plans, config, dataset
        )


def make_predictor(tiny_model, preprocessing_setup):
    plans, config, dataset = preprocessing_setup
    tiny_model.eval()
    tiny_model.decoder.deep_supervision = False
    predictor = APRMPredictor(
        device=torch.device("cpu"), use_mirroring=False, perform_everything_on_device=False, allow_tqdm=False
    )
    predictor.manual_initialization(tiny_model, plans, config, None, dataset, "APRMTrainer", None)
    return predictor


def test_raw_sequential_and_numpy_predictions_agree(raw_case, preprocessing_setup, tiny_model):
    image_file, anatomy_file = raw_case
    predictor = make_predictor(tiny_model, preprocessing_setup)
    sequential = predictor.predict_from_files_sequential(
        [[str(image_file)]], None, anatomy_dir=str(anatomy_file.parent)
    )
    image, properties = SimpleITKIO().read_images([str(image_file)])
    labels, anatomy_properties = SimpleITKIO().read_seg(str(anatomy_file))
    numpy_output = predictor.predict_single_npy_array(
        image, properties, anatomy=labels, anatomy_properties=anatomy_properties
    )
    assert sequential[0].shape == (10, 12, 14)
    np.testing.assert_array_equal(sequential[0], numpy_output)


def test_raw_worker_and_sequential_predictions_agree(raw_case, preprocessing_setup, tiny_model):
    image_file, anatomy_file = raw_case
    predictor = make_predictor(tiny_model, preprocessing_setup)
    sequential = predictor.predict_from_files_sequential(
        [[str(image_file)]], None, anatomy_dir=str(anatomy_file.parent)
    )
    workers = predictor.predict_from_files(
        [[str(image_file)]],
        None,
        anatomy_dir=str(anatomy_file.parent),
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    np.testing.assert_array_equal(sequential[0], workers[0])


def test_validation_predictor_accepts_categorical_and_channel_priors_equally(preprocessing_setup, tiny_model):
    predictor = make_predictor(tiny_model, preprocessing_setup)
    image = torch.randn((1, 6, 8, 8))  # Smaller than the tile: exercise OUT padding too.
    anatomy = torch.zeros_like(image)
    anatomy[:, :3] = 1
    anatomy[:, 3:] = 6
    one_hot = torch.nn.functional.one_hot(anatomy[0].long(), 7).movedim(-1, 0).float()
    categorical_result = predictor.predict_logits_from_preprocessed_data(torch.cat((image, anatomy), dim=0))
    channel_result = predictor.predict_logits_from_preprocessed_data(torch.cat((image, one_hot), dim=0))
    assert categorical_result.shape == (2, 6, 8, 8)
    torch.testing.assert_close(categorical_result, channel_result, rtol=0, atol=0)


def test_raw_prediction_export_preserves_original_nifti_geometry(raw_case, preprocessing_setup, tiny_model, tmp_path):
    import SimpleITK as sitk

    image_file, anatomy_file = raw_case
    predictor = make_predictor(tiny_model, preprocessing_setup)
    output = tmp_path / "output" / "renamed_prediction"
    predictor.predict_from_files_sequential([[str(image_file)]], [str(output)], anatomy_dir=str(anatomy_file.parent))
    prediction = sitk.ReadImage(str(output) + ".nii.gz")
    image = sitk.ReadImage(str(image_file))
    assert prediction.GetSize() == image.GetSize()
    np.testing.assert_allclose(prediction.GetSpacing(), image.GetSpacing())
    np.testing.assert_allclose(prediction.GetOrigin(), image.GetOrigin())
    np.testing.assert_allclose(prediction.GetDirection(), image.GetDirection())


def test_raw_prediction_requires_prior_before_network_execution(raw_case, preprocessing_setup, tiny_model, monkeypatch):
    image_file, anatomy_file = raw_case
    predictor = make_predictor(tiny_model, preprocessing_setup)

    def unexpected_forward(*args, **kwargs):
        pytest.fail("The model ran without a validated anatomical prior")

    monkeypatch.setattr(predictor, "predict_logits_from_preprocessed_data", unexpected_forward)
    with pytest.raises(ValueError, match="anatomy_dir"):
        predictor.predict_from_files_sequential([[str(image_file)]], None)
    anatomy_file.unlink()
    with pytest.raises(FileNotFoundError, match="subject.nii.gz"):
        predictor.predict_from_files([[str(image_file)]], None, anatomy_dir=str(anatomy_file.parent))


def test_prediction_cli_requires_anatomy_and_exposes_only_default():
    parser = _prediction_parser(False)
    with pytest.raises(SystemExit):
        parser.parse_args(["-i", "images", "-o", "predictions", "-d", "1"])
    args = parser.parse_args(
        ["-i", "images", "-o", "predictions", "-d", "1", "--anatomy_dir", "anatomy", "-npp", "0", "-nps", "0"]
    )
    assert args.tr == "APRMTrainer" and args.c == "3d_fullres"
    assert args.f == ["0"] and args.npp == args.nps == 0
