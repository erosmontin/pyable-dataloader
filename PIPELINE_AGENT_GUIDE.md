pyable-dataloader - Pipeline Agent Guide

Purpose
-------
This guide is written for LLM agents and engineers who want to build data-processing
and training pipelines using the `pyable-dataloader` package. It focuses on
practical recipe steps, runnable snippets, manifest structure, and operational
best practices so you can quickly integrate `pyable-dataloader` into a
training or inference pipeline.

Quick prerequisites
-------------------
- Use the `able` conda environment for running scripts in this repo.

```bash
conda activate able
```

- The package expects `pyable` and its Imaginable types to be installed and
  available in Python path. The dataset uses SimpleITK under-the-hood.

Pipeline Quickstart (minimal)
-----------------------------
The example below shows the minimal pieces to build a training pipeline:
- prepare/update a manifest
- create augmentation transforms (Imaginable-first)
- instantiate the dataset
- create a PyTorch DataLoader
- plug into your model/training loop

Small runnable sketch
---------------------
```python
import json
from pyable_dataloader.utils import update_manifest_with_reference
from pyable_dataloader.transforms import create_augmentation_transforms
from pyable_dataloader.dataset import AlphaAngleDataset
from torch.utils.data import DataLoader
import torch

# 1) Load manifest (list/dict like examples in repo)
with open('patientd.json') as f:
    manifest = json.load(f)

# 2) Optionally create/ensure a canonical reference (target_size, spacing)
#    This will write a reference NIfTI to /tmp and update manifest['S']['reference']
manifest = update_manifest_with_reference(manifest, target_size=[64,64,64], resolution=[2,2,2])

# 3) Create augmentation transforms (Imaginable-first recommended)
transforms = create_augmentation_transforms({
    'rotation_range': [[-3,3],[-3,3],[-3,3]],
    'translation_range': [[-3,3],[-3,3],[-3,3]],
    'use_noise_augmentation': False,
}, include_flip=True)

# 4) Build dataset
config = {
    'target_size': [64,64,64],
    'target_resolution': [2.0,2.0,2.0],
    'debug_save_dir': None,
}
# If your manifest is a dict keyed by subject, pass manifest.values() or convert
patient_list = list(manifest.values())
dataset = AlphaAngleDataset(patient_list, config=config, use_flipped=False, transforms=transforms)

# 5) DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=None)

# 6) Model + training (sketch)
from treno import EMResNet, EarlyStopping
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = EMResNet(in_channels=2, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for batch in loader:
        imgs = batch['images'].to(device)   # (B, C, D, H, W)
        bins = batch['bin'].to(device)      # classification bin
        out = model(imgs)
        loss = criterion(out, bins)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Manifest format
---------------
The `pyable-dataloader` expects a manifest-like structure with keys for each
subject. Each subject entry should include:
- `images`: list of image file paths
- `labelmaps`: list of segmentation file paths (optional)
- `rois`: list of ROI file paths (optional)
- `angle` (metadata) or similar fields used by consuming dataset
- `reference` (optional): path to a canonical reference image to resample onto

Example subject entry (JSON object):
```json
{
  "FT1013": {
    "images": ["/path/to/image.nii"],
    "labelmaps": ["/path/to/seg.nii"],
    "rois": ["/path/to/roi.nii"],
    "angle": 67.9,
    "reference": "/tmp/FT1013_reference.nii.gz"
  }
}
```

Best practices
--------------
- Imaginable-first: Prefer transforms that accept and return `pyable`
  Imaginable types (SITKImaginable/LabelMapable/Roiable). Spatial operations
  (rotation, affine, translation) performed by Imaginable methods keep
  physical metadata consistent and avoid label corruption.

- Intensity-only transforms must skip labelmaps/ROIs. Use `LabelMapable`/
  `Roiable` checks to avoid normalizing segmentation arrays.

- Use a `reference` image (via `update_manifest_with_reference`) to ensure all
  samples end up with consistent voxel spacing and crop/size. This simplifies
  batching and reduces on-the-fly resampling costs during training.

- Keep batch sizes small on low-GPU-memory hardware. Use AMP (`torch.cuda.amp`) and
  small networks or reduce `target_size` when experimenting.

Transforms and augmentation
---------------------------
- Use rigid transforms (small rotations/translations) for angle-preserving
  augmentations. Avoid large nonrigid warps if the target is an angle metric.

- Always validate transforms on small synthetic inputs first: create a
  synthetic image and labelmap so you can assert that label positions are
  preserved after augmentation.

- Where performance matters, precompute deterministic augmentations and cache
  them on disk; keep random augmentations on-the-fly.

Labelmap encoding and consistent channels
----------------------------------------

Two new transforms help make label encoding explicit and consistent across
datasets:

- `LabelMapOneHot(values=..., keep_original=False)`: converts labelmaps into
  one-hot binary masks. Use `values` to declare the full set of labels you
  expect across the dataset; this guarantees the same channel ordering even if
  some labels are missing in an individual volume.
- `LabelMapContiguous(values=..., keep_original=False)`: remaps arbitrary
  label integers to contiguous integers starting at 1 (0 remains background).

Usage patterns:

- Supply `values` explicitly when creating the transform:

```python
from pyable_dataloader.transforms import LabelMapOneHot
lm = LabelMapOneHot(values=[1,2,3], keep_original=False)
```

- Or set `manifest[subject]['meta']['labelmap_values'] = [1,2,3]` so the
  transform reads the list at runtime. The transform prefers constructor
  `values`, then `meta`, then falls back to per-image detection.

Both transforms write a mapping entry back to `meta` (under the configured
`meta_key`) so pipelines can decode predictions to original label IDs.

Debugging tips
--------------
- If SimpleITK raises dtype or shape errors, inspect types passed into
  `setImageFromNumpy` and `resampleOnTargetImage`. Ensure numeric dtypes
  (float32 / uint8) and `numpy.ndarray` (not lists or object arrays).

- If transforms return Python objects where arrays are expected, add
  assertions in your transform's `__call__` to always return Imaginable
  or numpy arrays depending on the chosen convention.

- Use `config['debug_save_dir']` in dataset config to emit processed
  images/labelmaps as NIfTI for visual checks.

Testing and CI
--------------
- Add small unit tests that run transforms on tiny (e.g., 8x8x8) synthetic
  images and assert the labelmap centroid is unchanged after rigid
  augmentation.

- Run these tests in CI in the `able` environment or container that includes
  `pyable` and SimpleITK.

Operational checklist before model training
------------------------------------------
- [ ] Manifest entries validated (paths exist, image sizes readable).
- [ ] Reference image created and present in manifest (use `update_manifest_with_reference`).
- [ ] Transforms configured and unit-tested on synthetic data.
- [ ] Dataset returns expected shapes for `images` and `labelmaps`.
- [ ] Training loop includes checkpointing and early stopping.

Suggested small utilities to add to pipelines
--------------------------------------------
- `manifest_validator(manifest)` — checks presence of files, reasonable spacing,
  and that each subject has an image and either a labelmap or ROI.

- `make_mini_dataset_for_test(manifest, n=2)` — builds and saves a tiny subset
  for fast unit tests.

- `visual_debug_export(sample, outdir)` — saves `images`, `labelmaps`, and
  overlay PNGs for quick visual verification.

Where to extend
---------------
- Convert more transforms to Imaginable-first implementations in
  `src/pyable_dataloader/transforms.py` to avoid scattered conversions.

- Add a `bridging` helper (Imaginable -> numpy -> transform -> Imaginable) if
  you want to keep many legacy numpy-based transforms without rewriting them.

- Add focused examples in `examples/` (kept minimal) showing: (A) dataset
  instantiation, (B) a single-epoch training run, (C) a deterministic
  transform unit test.

If you want, I can also:
- Add a short `pipeline_starter.py` file that ties these pieces together and
  can be executed in `able` to run a single-epoch training smoke test.
- Add a `manifest_validator` utility and a couple of unit tests for transforms.

