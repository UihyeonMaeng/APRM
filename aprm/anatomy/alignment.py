"""Transform anatomical priors onto the exact image preprocessing grid."""

from pathlib import Path

import numpy as np

from aprm.anatomy.labels import NUM_ANATOMY_CLASSES


def validate_anatomy_labels(array: np.ndarray) -> np.ndarray:
    """Reject malformed labels instead of silently treating them as outside."""
    array = np.asarray(array)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Anatomy must be a nonempty, finite label map.")
    if np.any(array != np.rint(array)) or array.min() < 0 or array.max() >= NUM_ANATOMY_CLASSES:
        raise ValueError("Anatomy labels must be integers in 0..6; see documentation/usage.md.")
    return array.astype(np.int16, copy=False)


def validate_geometry(image_properties: dict, anatomy_properties: dict) -> None:
    """Require a shared physical grid, including orientation and origin."""
    pairs = [(image_properties["spacing"], anatomy_properties["spacing"], "spacing")]
    for section, keys in (
        ("sitk_stuff", ("spacing", "origin", "direction")),
        ("nibabel_stuff", ("original_affine", "reoriented_affine")),
    ):
        if section in image_properties:
            if section not in anatomy_properties:
                raise ValueError(f"Anatomy is missing {section} geometry metadata.")
            for key in keys:
                if key in image_properties[section]:
                    if key not in anatomy_properties[section]:
                        raise ValueError(f"Anatomy is missing {section}.{key} metadata.")
                    pairs.append((image_properties[section][key], anatomy_properties[section][key], key))
    for image_value, anatomy_value, name in pairs:
        if np.shape(image_value) != np.shape(anatomy_value) or not np.allclose(
            image_value, anatomy_value, rtol=1e-5, atol=1e-4
        ):
            raise ValueError(f"Image/anatomy {name} mismatch. Resample the prior to the original image grid first.")


def align_anatomy(seg, anatomy_properties, image_properties, target_shape, plans_manager, configuration_manager):
    """Apply image transpose, stored crop and label resampling to a raw prior."""
    validate_geometry(image_properties, anatomy_properties)
    seg = validate_anatomy_labels(seg)
    if seg.ndim == 3:
        seg = seg[None]
    if seg.ndim != 4 or seg.shape[0] != 1:
        raise ValueError("Expected a single-channel 3D anatomical label map.")
    seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
    if tuple(seg.shape[1:]) != tuple(image_properties["shape_before_cropping"]):
        raise ValueError("Raw anatomy and image shapes differ before cropping.")
    bbox = image_properties["bbox_used_for_cropping"]
    seg = seg[(slice(None), *[slice(int(lo), int(hi)) for lo, hi in bbox])]
    original_spacing = [image_properties["spacing"][i] for i in plans_manager.transpose_forward]
    target_spacing = list(configuration_manager.spacing)
    if len(target_spacing) != 3:
        raise ValueError("This APRM release requires a 3D configuration.")
    seg = configuration_manager.resampling_fn_seg(seg, target_shape, original_spacing, target_spacing)
    seg = validate_anatomy_labels(seg)
    if tuple(seg.shape) != (1, *tuple(target_shape)):
        raise ValueError(f"Aligned anatomy shape {seg.shape} differs from image grid {target_shape}.")
    return seg[0]


def load_aligned_anatomy(filename, image_properties, target_shape, plans_manager, configuration_manager):
    filename = Path(filename)
    if not filename.is_file():
        raise FileNotFoundError(f"Missing anatomical prior: {filename}")
    reader = plans_manager.image_reader_writer_class()
    seg, anatomy_properties = reader.read_seg(str(filename))
    return align_anatomy(seg, anatomy_properties, image_properties, target_shape, plans_manager, configuration_manager)


def append_anatomy_channels(data, filename, image_properties, plans_manager, configuration_manager):
    seg = load_aligned_anatomy(filename, image_properties, data.shape[1:], plans_manager, configuration_manager)
    channels = (seg[None] == np.arange(NUM_ANATOMY_CLASSES)[:, None, None, None]).astype(np.float32)
    return np.concatenate((data, channels), axis=0)
