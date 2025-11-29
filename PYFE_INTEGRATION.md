# PyFE Integration Guide

This document explains how to use the pyable-dataloader from within pyfe.

## Architecture

The package dependency chain is:
```
pyfe → pyable-dataloader → pyable
```

This avoids circular dependencies while allowing pyfe to use the dataloader.

## Installation Options

### Option 1: Development Mode (Recommended for local development)

When working locally with symlinks already in place:

```bash
# Install pyable first
cd /home/erosm/pyable
pip install -e .

# Install able-dataloader
cd /home/erosm/able-dataloader
pip install -e .

# Install pyfe
cd /home/erosm/pyfe
pip install -e .
```

### Option 2: From Git (For production/deployment)

```bash
# Install pyfe with dataloader support
pip install "pyfe[dataloader] @ git+https://github.com/erosmontin/pyfe.git"
```

This will automatically install:
- pyable
- pyable_dataloader
- All other dependencies

### Option 3: Install pyfe without dataloader

```bash
pip install "pyfe @ git+https://github.com/erosmontin/pyfe.git"
```

Then add dataloader later:
```bash
pip install "pyable_dataloader @ git+https://github.com/erosmontin/able-dataloader.git"
```

## Usage from pyfe

### Import directly from pyfe

```python
# Import dataloader from pyfe
from pyfe.dataloader import (
    PyableDataset,
    Compose,
    RandomTranslation,
    RandomRotation,
    RandomAffine,
    IntensityNormalization
)

# Create transforms
transforms = Compose([
    RandomTranslation(translation_range=[[-5, 5], [-5, 5], [-2, 2]], p=0.8),
    RandomRotation(rotation_range=[[-10, 10], [-10, 10], [-5, 5]], p=0.5),
    RandomAffine(
        translation_range=[[-3, 3], [-3, 3], [-1, 1]],
        rotation_range=[[-15, 15], [-15, 15], [-10, 10]],
        scale_range=[[0.9, 1.1], [0.9, 1.1], [0.95, 1.05]],
        p=0.3
    ),
    IntensityNormalization(method='zscore')
])

# Create dataset
dataset = PyableDataset(
    manifest='train.json',
    target_size=[128, 128, 64],
    target_spacing=2.0,
    transforms=transforms,
    cache_dir='./cache'
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# Training loop
for batch in dataloader:
    images = batch['images']  # Shape: [B, C, D, H, W]
    labels = batch['label']   # Shape: [B]
    # Your training code here
```

### Check if dataloader is available

```python
from pyfe.dataloader import DATALOADER_AVAILABLE

if DATALOADER_AVAILABLE:
    print("✅ Dataloader is available")
    from pyfe.dataloader import PyableDataset
else:
    print("❌ Dataloader not installed")
    print("Install with: pip install 'pyfe[dataloader]'")
```

### Import directly from pyable_dataloader (if installed)

You can also import directly:

```python
from pyable_dataloader import PyableDataset, Compose
```

Both approaches work identically.

## Integration with pyfe Feature Extraction

Combine dataloader with pyfe's radiomics extraction:

```python
import pyfe.pyfe as pyfe
from pyfe.dataloader import PyableDataset

# Create dataset
dataset = PyableDataset(
    manifest='data.json',
    target_size=[128, 128, 64],
    target_spacing=2.0
)

# Extract features for each sample
features_list = []
for idx in range(len(dataset)):
    sample = dataset[idx]
    
    # Get original space overlayer
    overlay_fn = dataset.get_original_space_overlayer(idx)
    
    # Extract features using pyfe
    A = pyfe.PYRAD()
    A.setOptions({"bin": 32, "radius": 2})
    A.setImage(sample['original_images'][0])  # Original space
    A.setROI(sample['original_rois'][0])
    A.setROIvalue(1)
    
    features = A.getFeatures()
    features_list.append(features)
```

## Augmentation Configuration

Use the same augmentation style as your existing code:

```python
from pyfe.dataloader import (
    Compose,
    RandomTranslation,
    RandomRotation,
    RandomAffine,
    RandomBSpline
)

# Configure augmentation similar to your augmentem() function
aug_config = {
    "translation": [[-5, 5], [-5, 5], [-2, 2]],  # mm
    "rotation": [[-10, 10], [-10, 10], [-5, 5]],  # degrees
    "scale": [[0.9, 1.1], [0.9, 1.1], [0.95, 1.05]],  # multiplicative
    "center": "femur",  # or "image" or [x, y, z]
}

transforms = Compose([
    RandomTranslation(
        translation_range=aug_config["translation"],
        p=0.8,
        center=aug_config["center"]
    ),
    RandomRotation(
        rotation_range_deg=aug_config["rotation"],
        p=0.5,
        center=aug_config["center"]
    ),
    RandomAffine(
        translation_range=aug_config["translation"],
        rotation_range=aug_config["rotation"],
        scale_range=aug_config["scale"],
        p=0.3,
        center=aug_config["center"]
    )
])

# If using femur center, pass it in metadata
# The dataset will provide center_physical to transforms
```

## Symlink Setup (Development)

The current symlink setup is:

```
/home/erosm/able-dataloader/
├── pyable -> /home/erosm/pyable/
├── pyfe -> /home/erosm/pyfe/
└── src/pyable_dataloader/

/home/erosm/pyfe/
├── able-dataloader -> /home/erosm/able-dataloader/
├── pyable -> /home/erosm/pyable/
└── pyfe/
    └── dataloader.py  (re-exports from pyable_dataloader)
```

This setup:
- ✅ Avoids circular dependencies
- ✅ Allows development in each repo independently
- ✅ Provides convenient imports from pyfe
- ✅ Works with `pip install -e .` for all packages

## Testing the Integration

```bash
# Test that pyfe can import dataloader
cd /home/erosm/pyfe
python -c "from pyfe.dataloader import PyableDataset; print('✅ Import successful')"

# Test full workflow
python -c "
from pyfe.dataloader import PyableDataset, DATALOADER_AVAILABLE
print(f'Dataloader available: {DATALOADER_AVAILABLE}')
if DATALOADER_AVAILABLE:
    print('✅ Full integration working')
"
```

## Troubleshooting

### ImportError: No module named 'pyable_dataloader'

**Solution**: Install able-dataloader:
```bash
cd /home/erosm/able-dataloader
pip install -e .
```

### Circular import / symlink issues

The symlinks should be:
- `able-dataloader/pyable` → `/home/erosm/pyable/` ✅
- `able-dataloader/pyfe` → `/home/erosm/pyfe/` ✅ (for reference only)
- `pyfe/able-dataloader` → `/home/erosm/able-dataloader/` ✅
- `pyfe/pyable` → `/home/erosm/pyable/` ✅

Make sure you don't have:
- `pyable/pyfe` ❌ (would create cycle)
- `pyable/able-dataloader` ❌ (would create cycle)

### Labels not preserved during augmentation

The transforms automatically use nearest-neighbor interpolation for `Roiable` and `LabelMapable` objects. Verify your ROIs are loaded as the correct type:

```python
from pyable.imaginable import Roiable, LabelMapable

# For binary masks
roi = Roiable('mask.nii.gz')

# For multi-label segmentations
labelmap = LabelMapable('segmentation.nii.gz')
```

## Publishing to GitHub

When ready to publish, update these repositories:

1. **able-dataloader**: Push to `github.com/erosmontin/able-dataloader`
2. **pyfe**: Push updated `pyproject.toml` and `dataloader.py` to `github.com/erosmontin/pyfe`
3. **pyable**: Already at `github.com/erosmontin/pyable`

Then users can install with:
```bash
pip install "pyfe[dataloader] @ git+https://github.com/erosmontin/pyfe.git"
```

## Summary

- ✅ pyfe can now use the dataloader via `from pyfe.dataloader import PyableDataset`
- ✅ No circular dependencies (pyfe → able-dataloader → pyable)
- ✅ Optional dependency (pyfe works without dataloader if not needed)
- ✅ Development mode works with symlinks
- ✅ Production mode works via GitHub URLs
- ✅ All transforms available: Translation, Rotation, Affine, BSpline
- ✅ Labels automatically preserved during augmentation
