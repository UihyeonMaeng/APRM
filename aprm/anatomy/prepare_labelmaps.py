"""Build the fixed seven-class APRM prior from TotalSegmentator CT masks."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from aprm.anatomy.alignment import validate_anatomy_labels
from aprm.anatomy.labels import ANATOMY_LABELS, CENTRAL_ORGANS, LOBE_FILES, REQUIRED_ROIS
from aprm.paths import APRM_raw, APRM_results
from aprm.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def _case_id(image_path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not image_path.name.endswith(suffix):
        raise ValueError(f"Expected <case>_0000.nii.gz, received {image_path.name}")
    return image_path.name[: -len(suffix)]


def _load_mask(path: Path, reference) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required anatomical mask: {path}")
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"Anatomical mask does not share the CT image grid: {path}")
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite anatomical mask: {path}")
    return values > 0


def _run_totalsegmentator(image_path: Path, mask_dir: Path, overwrite: bool) -> None:
    complete = all((mask_dir / f"{roi}.nii.gz").is_file() for roi in REQUIRED_ROIS)
    if complete and not overwrite:
        return
    mask_dir.parent.mkdir(parents=True, exist_ok=True)
    command = ["TotalSegmentator", "-i", str(image_path), "-o", str(mask_dir), "--roi_subset", *REQUIRED_ROIS]
    subprocess.run(command, check=True)
    missing = [roi for roi in REQUIRED_ROIS if not (mask_dir / f"{roi}.nii.gz").is_file()]
    if missing:
        raise RuntimeError(f"TotalSegmentator output is incomplete: {missing}")


def prepare_anatomy_folder(input_dir, output_dir, cache_dir, skip_totalseg=False, overwrite=False):
    """Create one prior per image; missing structures are never silently omitted."""
    input_dir, output_dir, cache_dir = map(Path, (input_dir, output_dir, cache_dir))
    image_files = sorted(input_dir.glob("*_0000.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No *_0000.nii.gz images found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_mapping.json").write_text(json.dumps(ANATOMY_LABELS, indent=2) + "\n")
    for image_path in image_files:
        case = _case_id(image_path)
        output_path = output_dir / f"{case}.nii.gz"
        reference = nib.load(str(image_path))
        if len(reference.shape) != 3:
            raise ValueError(f"APRM requires a single-channel 3D CT image: {image_path}")
        if output_path.exists() and not overwrite:
            existing = nib.load(str(output_path))
            if existing.shape != reference.shape or not np.allclose(
                existing.affine, reference.affine, rtol=1e-5, atol=1e-4
            ):
                raise ValueError(f"Existing prior has different geometry: {output_path}")
            validate_anatomy_labels(np.asarray(existing.dataobj))
            print(f"[SKIP] Valid existing prior: {output_path}")
            continue
        mask_dir = cache_dir / image_path.name[:-7]
        if not skip_totalseg:
            _run_totalsegmentator(image_path, mask_dir, overwrite)
        labelmap = np.zeros(reference.shape, dtype=np.uint8)
        for label, roi in LOBE_FILES.items():
            mask = _load_mask(mask_dir / f"{roi}.nii.gz", reference)
            labelmap[mask & (labelmap == 0)] = ANATOMY_LABELS[label]
        central = np.zeros(reference.shape, dtype=bool)
        for roi in CENTRAL_ORGANS:
            central |= _load_mask(mask_dir / f"{roi}.nii.gz", reference)
        labelmap[central & (labelmap == 0)] = ANATOMY_LABELS["CENTRAL"]
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(labelmap, reference.affine, header), str(output_path))
        print(f"[OK] {output_path}")
    return [str(output_dir / f"{_case_id(image)}.nii.gz") for image in image_files]


def prepare_labelmaps_entry():
    parser = argparse.ArgumentParser(description="Create fixed APRM anatomical priors for lung CT.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-d", "--dataset", help="Dataset name or ID under APRM_raw")
    source.add_argument("-i", "--input", help="Folder with <case>_0000.nii.gz images")
    parser.add_argument("-o", "--output", help="Anatomy output folder (required with -i)")
    parser.add_argument("--cache_dir", help="Directory for TotalSegmentator masks")
    parser.add_argument("--splits", default="Tr", help="Dataset splits: Tr, Ts, or Tr,Ts")
    parser.add_argument("--skip_totalseg", action="store_true", help="Reuse complete masks from the cache")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing priors and masks")
    args = parser.parse_args()
    if args.input:
        if not args.output:
            parser.error("-o/--output is required with -i/--input")
        cache = args.cache_dir or str(Path(args.output).parent / "totalseg")
        prepare_anatomy_folder(args.input, args.output, cache, args.skip_totalseg, args.overwrite)
        return
    if args.output:
        parser.error("Use -i with -o, or let -d choose anatomyTr/anatomyTs automatically.")
    if not APRM_raw:
        raise EnvironmentError("Set APRM_raw before preparing a dataset.")
    dataset = maybe_convert_to_dataset_name(args.dataset)
    raw = Path(APRM_raw) / dataset
    cache = (
        Path(args.cache_dir)
        if args.cache_dir
        else (Path(APRM_results) / dataset / "totalseg" if APRM_results else raw / "totalseg")
    )
    splits = [part.strip() for part in args.splits.replace(" ", ",").split(",") if part.strip()]
    if not splits or any(split not in ("Tr", "Ts") for split in splits):
        parser.error("--splits must contain Tr and/or Ts")
    for split in splits:
        prepare_anatomy_folder(
            raw / f"images{split}", raw / f"anatomy{split}", cache / split, args.skip_totalseg, args.overwrite
        )
