LLM Agent Guide for pyable-dataloader

Purpose
-------
This document is written for language models, assistant agents, and engineers who
will modify or extend the `pyable-dataloader` package. It describes the
project's intent, invariants, common failure modes, and step-by-step recipes
for common tasks (adding transforms, creating canonical references, and
running the dataset).

Core principles
---------------
- Imaginable-first: When possible, transforms should accept and return
  `pyable` Imaginable objects (e.g., `SITKImaginable`, `LabelMapable`,
  `Roiable`). Use the Imaginable API for spatial operations to preserve
  label integrity.

- Dual-path compatibility: For convenience, many transforms still accept
  numpy arrays. If you add a new transform, implement both paths or provide
  a small bridge that converts Imaginable -> numpy -> Imaginable.

- Label safety: Intensity-only transforms must not modify labelmaps/ROIs.
  Use `LabelMapable` and `Roiable` type checks to skip intensity transforms
  or to apply nearest-neighbour logic where required.

Important files
---------------
- `src/pyable_dataloader/dataset.py` — Dataset lifecycle: loading, padding,
  transform application, resampling to reference, and conversion to numpy.

- `src/pyable_dataloader/transforms.py` — Collection of existing transforms.
  Follow the patterns in this file.

- `src/pyable_dataloader/utils.py` — Helpers such as
  `update_manifest_with_reference` to generate canonical references.

Key APIs and idioms
-------------------
- Imaginable methods you will commonly use (from `pyable.imaginable`):
  - `getImageAsNumpy()` / `setImageFromNumpy()` — convert to/from numpy.
  - `resampleOnTargetImage(target, interpolator=...)` — resample and
    return an Imaginable aligned to `target`.
  - `applyTransform(transform, target_image=..., interpolator=...)` — apply
    SimpleITK transforms.
  - `rotateImage`, `translateImage`, `transformImageAffine` — helpers for
    spatial ops.

- When writing code that calls SimpleITK methods that expect continuous
  indices (e.g., `TransformContinuousIndexToPhysicalPoint`), ensure you pass
  Python floats or lists/tuples of floats (not numpy ints) to avoid type
  errors.

Recipes
-------
1) Add a new spatial transform (recommended approach)
   - Implement transform as a `MedicalImageTransform` subclass in
     `transforms.py`.
   - In `__call__`, detect whether `images` is a list of Imaginable objects
     (check `hasattr(images[0], 'getImage')` or `getImageAsNumpy`). If so:
     - Use the Imaginable API (`getDuplicate()` or `SITKImaginable(image=...)`)
       and call `applyTransform`/`rotateImage`/`translateImage` with
       appropriate interpolators.
     - Return Imaginable objects so the dataset can resample and debug-save.
   - Also implement a numpy-path (or use a bridging decorator) to keep
     backwards compatibility.

2) Add an intensity transform that should skip labelmaps/ROIs
   - In `__call__`, detect `LabelMapable`/`Roiable` and skip normalization.
   - For Imaginable inputs, call `getImageAsNumpy()`, perform numpy ops,
     then `setImageFromNumpy()` to write back.

3) Create a canonical reference and resample everything to it
   - Use `update_manifest_with_reference(manifest, resolution, target_size)`
     which constructs a centered SITK reference and writes a resampled
     reference image to `/tmp` and updates the manifest.
   - In `PyableDataset.__getitem__`, resample all images/labelmaps/rois to
     that `reference` using `resampleOnTargetImage(reference)`.

Common pitfalls and how to detect/fix them
-----------------------------------------
- TypeError: SimpleITK complaining about dtype=object
  - Cause: A transform returned Python objects (Imaginable instances) inside a
    numpy array or vice-versa. Fix by ensuring numpy arrays passed to
    SimpleITK are numeric dtype (use `np.asarray(arr)`), and Imaginable
    returns are actual Imaginable objects when the dataset expects them.

- NameError inside transform math
  - Cause: precomputing matrices in `__init__` with variables undefined.
  - Fix: sample random parameters inside `__call__` and compute matrices
    there.

- AxisError when flipping arrays
  - Cause: calling `np.flip` on an Imaginable or using an axis that doesn't
    exist for the array shape. Fix by handling Imaginable inputs separately
    (convert to numpy first), and ensure axis indices map correctly for
    channel-first stacks (C, D, H, W) vs 3D arrays (D, H, W).

Testing guidance
-----------------
- Write small, deterministic tests for each transform that exercise both
  Imaginable and numpy input paths.
- Use tiny synthetic images (e.g., ones/zeros arrays) and simple ROIs so the
  expected output is easy to assert.

Maintenance notes for LLM agents
--------------------------------
- Always check `transforms.py` and `dataset.py` together when changing
  transform behavior — the dataset expects Imaginable objects in many
  places.
- Use the `able` conda environment when running scripts in this repo:

```bash
conda activate able
python -m examples.some_example  # if an example exists
```

- If you change the Imaginable API usage, search for `getImageAsNumpy`,
  `setImageFromNumpy`, `resampleOnTargetImage`, and `applyTransform` to
  update other code paths.

If you'd like, I can also add a small test harness in `tests/` with a few
synthetic cases and a helper function that creates tiny Imaginable objects
for faster local testing.
