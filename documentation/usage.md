# Data preparation and usage

## Supported configuration

The public training path is `APRMTrainer`, `APRMPlans`, and `3d_fullres`: a U-Net backbone with APRM after every encoder stage. The intended task is single-channel lung CT tumor segmentation with categorical targets. Baseline, alternative-backbone, and ablation trainers are not included. This release does not implement APRM-Pro or claim general CT/MRI support.

## Dataset layout

Set `APRM_raw`, `APRM_preprocessed`, and `APRM_results` before invoking commands. Store training data under `APRM_raw/DatasetXXX_Name`:

| Path | Content |
| --- | --- |
| `imagesTr/<case>_0000.nii.gz` | CT image |
| `labelsTr/<case>.nii.gz` | Tumor segmentation: background 0, tumor 1 |
| `dataset.json` | Dataset metadata |
| `anatomyTr/<case>.nii.gz` | Anatomical prior generated from the CT |
| `imagesTs/<case>_0000.nii.gz` | Optional held-out images |
| `anatomyTs/<case>.nii.gz` | Optional held-out anatomical priors |

Example `dataset.json` (replace `numTraining` with the actual number of training cases):

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "tumor": 1},
  "numTraining": 319,
  "file_ending": ".nii.gz"
}
```

The number above is an example based on the paper's reported training cohort; no subject-level split is distributed. Tumor labels and anatomical priors are separate inputs. Do not use tumor ground truth to create deployable routing priors.

## Anatomical prior

The map is a mutually exclusive integer label volume on the original CT grid:

| ID | Region |
| --- | --- |
| 0 | Outside the six selected anatomical regions |
| 1 | Right upper lobe (RUL) |
| 2 | Right middle lobe (RML) |
| 3 | Right lower lobe (RLL) |
| 4 | Left upper lobe (LUL) |
| 5 | Left lower lobe (LLL) |
| 6 | CENTRAL |

CENTRAL is the union of **heart, aorta, trachea, esophagus, pulmonary vein, superior vena cava, and inferior vena cava**. Lobe labels take precedence if supplied masks overlap. These are TotalSegmentator v2 CT ROI names, mapped to APRM IDs; do not copy TotalSegmentator's original integer IDs directly.

Install TotalSegmentator according to its [official instructions](https://github.com/wasserth/TotalSegmentator). It can use a separate environment, provided the `TotalSegmentator` command is on `PATH`. Record the version and weights used for each experiment. Automatic extraction requires the pretrained weights; the test suite does not download or execute them.

Dataset mode:

```bash
APRM_prepare_anatomy -d Dataset020_NSCLC --splits Tr,Ts
```

Folder mode for independent test cohorts:

```bash
APRM_prepare_anatomy -i /data/external/images -o /data/external/anatomy
```

The command caches per-organ masks. Use `--skip_totalseg --cache_dir /path/to/cache` to reuse a complete extraction. A cached case uses `<case>_0000/<roi>.nii.gz` in folder mode, or `<split>/<case>_0000/<roi>.nii.gz` in dataset mode. All required mask files must exist, even if a particular structure has no positive voxels within the scan.

Existing valid final priors are reused. To rebuild priors with the complete CENTRAL definition, use `--overwrite`; add `--skip_totalseg` when reusing already complete cached masks. Image and prior shape, orientation, origin, and spacing must agree.

## Preprocessing and routing sidecars

```bash
APRM_plan_and_preprocess -d 20 --verify_dataset_integrity -np 4
APRM_prepare_routing -d Dataset020_NSCLC -c 3d_fullres -p APRMPlans
```

Preprocessing follows the inherited nnU-Net image workflow. Prior conversion reuses its transpose, stored crop, segmentation resampler, and final image shape. Anatomical channels never pass through image-intensity augmentation or enter the image backbone.

The planner estimates memory for the backbone. Expert adapters at the highest-resolution stage add memory, so the selected patch/batch size is not a measured GPU-memory guarantee for APRM. If necessary, use a smaller planning target and a separate plans name:

```bash
APRM_plan_and_preprocess -d 20 -gpu_memory_target 6 \
  -overwrite_plans_name APRMPlansSmall
```

Then pass `-p APRMPlansSmall` to routing preparation, training, and prediction. This changes the experimental configuration and should be recorded. The paper's spacing and patch size are not hard-coded: the original plans are needed for exact reproduction. A spacing override is available via `-overwrite_target_spacing`; its axis order follows the image reader (typically z, y, x for SimpleITK).

After changing anatomical maps or plans, explicitly rebuild sidecars with `APRM_prepare_routing ... --overwrite`. An existing sidecar is not automatically checked for freshness against its source files.

## Training and evaluation splits

```bash
APRM_train Dataset020_NSCLC 3d_fullres 0
APRM_train Dataset020_NSCLC 3d_fullres 0 --c
APRM_train Dataset020_NSCLC 3d_fullres 0 --val
APRM_train Dataset020_NSCLC 3d_fullres 0 --val_best
```

`--val` and `--val_best` perform validation only. Resume requires an existing latest/final checkpoint. Starting a fresh run refuses to overwrite existing checkpoints.

For fixed splits, provide `APRM_preprocessed/<dataset>/splits_final.json` before training. It is a JSON list indexed by fold, each entry with `train` and `val` case-ID lists. The same patient must not appear in both lists. Without it, the inherited five-fold split is created from the training dataset. `fold=all` uses the same available data for training and validation and must not be reported as independent test performance. Final validation also writes an in-sample train report; use the `validation` folder and the specified held-out test cohort for performance reporting.

Outputs are saved under `APRM_results/<dataset>/APRMTrainer__<plans>__3d_fullres/fold_<fold>`. Checkpoints, training progress, experiment configuration, and routing usage statistics are saved there. `CUDA_VISIBLE_DEVICES=0` selects a GPU. `-device cpu` is available for small checks. DDP and `-num_gpus > 1` are intentionally unsupported until their conditional routing behavior has been validated.

## Inference

```bash
APRM_predict -i /data/external/images -o /data/external/predictions \
  --anatomy_dir /data/external/anatomy \
  -d Dataset020_NSCLC -c 3d_fullres -f 0
```

Here `-d` identifies the **training model**, even when predicting another cohort. The predictor uses the training plans; it does not refit preprocessing to the test set. The default uses overlapping sliding windows (step 0.5), Gaussian-weighted logit aggregation, and mirroring. `-chk checkpoint_best.pth` selects the best checkpoint. Use `-npp 0 -nps 0` for sequential preprocessing and export.

Without `APRM_results`, provide the model directory directly:

```bash
APRM_predict_from_modelfolder -i /data/external/images \
  -o /data/external/predictions --anatomy_dir /data/external/anatomy \
  -m /path/to/APRMTrainer__APRMPlans__3d_fullres -f 0
```

Use only checkpoints produced by this release. Renaming older checkpoint keys cannot create the added stage-0 expert/router weights. Legacy development checkpoints require their matching archived implementation.

Evaluate binary tumor predictions with:

```bash
APRM_evaluate_simple /path/to/ground_truth /path/to/predictions -l 1
```

The summary reports per-case and aggregate Dice/IoU. Keep the test split and checkpoint-selection protocol fixed when comparing runs.

## Development checks

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -q
ruff check aprm tests
```

Tests use small synthetic volumes and real PyTorch networks. They do not establish clinical segmentation accuracy or reproduce the paper's reported scores.
