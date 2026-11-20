# Release validation

These checks were run on synthetic data, not on the study cohorts.

## Results

- Regression tests: **31 passed**; no skipped tests.
- Static undefined-name/syntax checks: passed.
- Python source compilation: passed.
- Internal module resolution: no missing APRM modules.
- Wheel build and separate installation: passed.
- All 13 installed console commands: `--help` completed successfully.
- No local path names contain the old framework/project namespace.
- Original Apache license file: byte-identical to the supplied source.

The tests cover real small U-Net forward/backward, sparse routing equations, anatomical alignment and augmentation, sequential/NumPy/parallel prediction parity, NIfTI export geometry, and a trainer checkpoint/resume path. Separate synthetic checks also exercised final-validation reports and epoch plotting.

No dataset-scale training, paper-score reproduction, TotalSegmentator model execution, GPU/CUDA run, or DDP validation was performed. Dependency bounds express package requirements, not a test matrix across all versions.

## Environment used

| Component | Version |
| --- | --- |
| Python | 3.12.14 |
| torch | 2.15.0+cpu |
| numpy | 2.3.5 |
| scipy | 1.17.0 |
| dynamic-network-architectures | 0.4.4 |
| acvl-utils | 0.2.7 |
| batchgenerators | 0.25.6 |
| batchgeneratorsv2 | 0.3.8 |
| blosc2 | 4.15.0 |
| SimpleITK | 2.5.8 |
| nibabel | 5.5.0 |
| pytest | 9.2.0 |

## Commands

```bash
python -m pytest tests -q
ruff check aprm tests
python -m compileall -q aprm tests
python -m pip wheel --no-deps --no-build-isolation .
```

A wheel was installed into an isolated package directory for the CLI checks; imports were verified to resolve to that installed copy.
