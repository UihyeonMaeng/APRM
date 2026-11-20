# Implementation notes

## Scope of the release

This repository exposes one default model: a U-Net with APRM at **every encoder stage**, including the first stage and the encoder bottleneck. The framework code derives from the supplied nnU-Net v2 source snapshot. The supported training path is single-device, categorical, non-cascaded 3D segmentation. The intended application is lung tumor segmentation in CT.

Removed components include alternative-backbone trainers, baseline/Vanilla-MoE trainers, anatomy-input comparison trainers, expert-count/top-k/threshold/stage ablations, competition/dataset-specific scripts, rebuttal scripts, old visualization experiments, build metadata, and bytecode caches. The plain `BaseTrainer` remains internal infrastructure for optimization, augmentation, checkpointing, and validation; it is not an exposed comparison method.

Repository-local packages, commands, plans, environment variables, and paths use APRM names. True external APIs and upstream attribution retain their original spelling, including `batchgeneratorsv2.transforms.nnunet` imports and the nnU-Net project URL.

## Fixed model

| Component | Release behavior |
| --- | --- |
| Anatomical labels | OUT, RUL, RML, RLL, LUL, LLL, CENTRAL |
| Patch gate | OUT if its occupancy is at least 0.9; otherwise largest non-OUT occupancy |
| Ties | First maximal anatomical class in the fixed label order |
| Encoder placement | All stages, determined from the instantiated backbone |
| Expert pool | Eight experts at each stage, shared across anatomical gates at that stage |
| Expert routing | Token-wise top-2 weights, normalized over the selected experts |
| Expert adapter | Linear C→4C, ReLU, dropout 0.1, Linear 4C→C |
| Router noise | Gaussian with learned Softplus scale during training; absent at evaluation |
| Residual update | Original stage features plus the weighted expert output |
| Balancing | Dense softmax probabilities before expert top-k; M times squared mean expert probabilities |
| Stage reduction | Equal mean across routed stages |
| Objective | Deep-supervised Dice + CE, plus 0.01 times balancing loss |
| Optimizer | Inherited SGD: learning rate 0.01, momentum 0.99, weight decay 3e-5, Nesterov |
| Training schedule | Inherited 1000 epochs, 250 training and 50 patch-validation iterations per epoch |
| Inference | Step 0.5, Gaussian-weighted logit aggregation, mirroring |

The expert pool is shared **between anatomical gates within a stage**, not between encoder stages with different channel sizes. Dense balancing gradients supervise every expert's routing logit; only dispatched expert networks receive the segmentation gradient for a given token.

## Important changes from the supplied development snapshot

### Encoder coverage

The earlier named default selected stages `[1, 2, 3, 4, 5]`, which excludes stage 0 in a six-stage encoder. This release follows the requested all-encoder definition. The paper text says all encoder scales, while its figure depicts five non-middle placements; the supplied source and figure do not uniquely establish one historical placement.

For an assumed six-stage channel schedule `[32, 64, 128, 256, 320, 320]`, the earlier stages 1–5 yield 122,416 router parameters, consistent with the paper's rounded 0.122M. All six stages yield 126,112 router parameters and add 70,512 total parameters relative to that earlier configuration. This is an inference from the source and a conventional channel schedule, not verification of the absent original plans.

Old checkpoints are not directly compatible. Architecture/member names changed and the first-stage expert/router weights did not exist in the earlier configuration. Evaluate newly trained release checkpoints before quoting the earlier accuracy, parameter, or runtime tables.

### Anatomical prior definition

The original default preparation included only heart, aorta, trachea, and esophagus in CENTRAL. The release also includes pulmonary vein and both venae cavae, matching the paper's listed definition. An external configuration may have supplied these for the historical experiments; that configuration was not present in the ZIP.

Prior generation now requires all selected organ-mask files and validates their geometry. An empty structure mask within a cropped field of view is valid; a missing extraction file is an error.

### Raw-image inference

The original generic raw predictor did not load anatomical priors, although trainer-based validation and the external-validation script did. The network could then silently route all image-only patches to OUT. Both public prediction commands now require `--anatomy_dir`. File and NumPy inference use the same prior alignment logic as routing-sidecar preparation.

Input maps must share the raw image's physical grid. Shape, spacing, orientation, and origin are checked before applying the stored image crop, transpose, and label resampling. Missing or invalid priors raise errors instead of changing the model's behavior silently.

### Padding and training alignment

Training applies shared crops and spatial augmentation to the image, tumor target, and anatomical labels. It removes anatomy from the segmentation target and computes occupancy after augmentation. Inference computes occupancy once per input patch and reuses it at every scale.

Categorical padding is OUT. For channel priors, an all-zero padded voxel also counts as OUT. This removes the earlier discrepancy where zero-padded one-hot voxels disappeared from the occupancy denominator.

### Runtime and reporting

The public trainer rejects DDP until conditional unused parameters and synchronized failure handling have been validated on multiple devices. Nonfinite predictions/losses raise instead of becoming finite zeros. The usual CUDA AMP overflow behavior is retained with a log message.

Epoch routing statistics record actual top-2 activation counts separately from top-1 diagnostic assignments. Experiment configuration is also saved locally. Optional HD95 reporting uses Euclidean distance transforms instead of a quadratic surface-distance matrix; finite-only means are named explicitly and nonfinite-case counts are retained. Dice/IoU remain the paper's reported metrics.

Raw multiprocessing preprocessing uses spawn-context pipe-backed queues/events with NumPy payloads. This avoids an extra Manager server and PyTorch storage-sharing sockets while preserving the same float32 inference tensors.

## Reproducibility limits

The uploaded source did not contain the historical experiment plans, patient splits, anatomical-extraction configuration, trained weights, logs, or per-case outputs. Consequently, this release is a runnable cleaned implementation, **not a numerical reproduction certificate**. The paper's reported spacing/patch dimensions and historical stage placement cannot be inferred reliably from dataset-independent defaults alone.

A release evaluation should retain its plans, splits, anatomical-prior version/weights, checkpoints, per-case predictions, hardware, and environment versions. Existing `_routing.npy` sidecars are reused by name; rebuild with `--overwrite` after changing priors or plans.

The inherited planner estimates backbone memory, not measured all-stage APRM memory. Full-resolution expert routing can be substantially more expensive than backbone-only segmentation. CPU tests do not establish GPU memory requirements, CUDA AMP stability, or dataset-scale latency.

## Validation performed

The included synthetic regression suite tests routing equations, exact sparse dispatch, gradients, every encoder stage, anatomical data/target separation, image/prior geometry, sequential/NumPy/multiprocessing inference parity, NIfTI restoration, and an actual tiny trainer update/checkpoint/resume path. Syntax, module resolution, package metadata, and command entry points are checked separately for release packaging.

The verification environment uses Python 3.12 and PyTorch 2.15.0+cpu. No real cohort training, TotalSegmentator weight inference, GPU/CUDA execution, DDP, or paper-score reproduction was performed. The dependency constraints are not a claim that every allowed dependency combination has been tested.
