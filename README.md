# APRM

## Anatomical Prior-Guided Routing for 3D Lung Tumor Segmentation

**Uihyeon Maeng\*, Jonghun Kim\*, and Hyunjin Park†**  
Sungkyunkwan University  
**ACCV 2026 · Oral presentation**  
\* Equal contribution · † Corresponding author

APRM (**Anatomical Prior-guided Routing Module**) adapts patch-level feature processing to anatomical context. It uses automatically extracted anatomical priors to select a region-specific router, which activates a small subset of shared expert adapters for each feature token.

This repository provides the **default APRM implementation** and its complete preparation, training, and inference pipeline. The model class is named `APRM`. It builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

## How does APRM work?

1. Extract the five lung lobes and a central region using [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).
2. Select one anatomical gate from the regional occupancy of each image patch.
3. Apply token-wise, noisy top-2 routing over eight shared experts at every encoder stage. The decoder remains shared.

The release uses seven anatomical gates, an outside threshold of **0.9**, and a balancing-loss weight of **0.01**. APRM is applied to **all encoder stages, including the first stage and bottleneck**.

## Getting started

### 1. Installation

Use Python 3.10 or later. Install a PyTorch build compatible with your CUDA environment, then run the following from this repository:

```bash
python -m pip install -e .
```

Install TotalSegmentator separately if anatomical priors have not already been generated. Its pretrained weights are required for prior extraction.

### 2. Set up paths and data

```bash
export APRM_raw="/path/to/APRM_raw"
export APRM_preprocessed="/path/to/APRM_preprocessed"
export APRM_results="/path/to/APRM_results"
```

Place the dataset in `APRM_raw/Dataset020_NSCLC`, using `imagesTr`, `labelsTr`, and `dataset.json`. Images use `<case>_0000.nii.gz`; tumor labels use `<case>.nii.gz`. See [dataset format and usage](documentation/usage.md).

### 3. Prepare anatomical priors and preprocess

```bash
APRM_prepare_anatomy -d Dataset020_NSCLC --splits Tr
APRM_plan_and_preprocess -d 20 --verify_dataset_integrity
APRM_prepare_routing -d Dataset020_NSCLC
```

The last command aligns anatomical maps with the preprocessed image grid. Training also generates missing routing sidecars automatically.

### 4. Train and validate

```bash
APRM_train Dataset020_NSCLC 3d_fullres 0
APRM_train Dataset020_NSCLC 3d_fullres 0 --c
APRM_train Dataset020_NSCLC 3d_fullres 0 --val
```

These commands start training, resume training, and validate the final checkpoint, respectively. This release supports **single-device, full-resolution 3D training**. Use your recorded split file for a fixed study protocol.

### 5. Predict

Prepare an anatomical map for each test image, then provide its folder explicitly:

```bash
APRM_prepare_anatomy -i /path/to/imagesTs -o /path/to/anatomyTs
APRM_predict -i /path/to/imagesTs -o /path/to/predictions \
  --anatomy_dir /path/to/anatomyTs \
  -d Dataset020_NSCLC -c 3d_fullres -f 0
```

Each prior must be named `<case>.nii.gz` and share the original image geometry. Missing or misaligned priors raise an error.

## Release notes

This is an **all-encoder release**. It is not directly compatible with checkpoints from the earlier development configuration. Exact paper-number reproduction requires the original plans, data splits, anatomical maps, and checkpoints, which are not bundled. See [implementation notes](documentation/implementation_notes.md) for the changes and validation scope.

## Citation

```bibtex
@article{maeng2026anatomical,
  title={Anatomical Prior-Guided Routing for 3D Lung Tumor Segmentation},
  author={Maeng, Uihyeon and Kim, Jonghun and Park, Hyunjin},
  journal={arXiv preprint arXiv:2610.39414},
  year={2026}
}
```

A paper link, DOI, and final proceedings metadata can be added here when available. Please also cite nnU-Net and TotalSegmentator when using the corresponding components.

## Acknowledgements and license

We thank the developers of nnU-Net, dynamic-network-architectures, and TotalSegmentator. The original nnU-Net attribution is retained in [NOTICE](NOTICE), and the included source license is [Apache 2.0](LICENSE).
