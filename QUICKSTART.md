# Quick Start Guide

## Installation (2 minutes)

```bash
# 1. Install pyable (if not already installed)
cd /path/to/pyable
pip install -e .

# 2. Install pyable-dataloader
cd /path/to/able-dataloader
pip install -e .

# 3. (Optional) Install development dependencies
pip install pytest pytest-cov
```

## Minimal Working Example (5 minutes)

### 1. Create a simple manifest

```python
# create_manifest.py
import json

manifest = {
    "subject_001": {
        "images": ["/path/to/image1.nii.gz"],
        "rois": [],
        "labelmaps": [],
        "label": 1.0
    },
    "subject_002": {
        "images": ["/path/to/image2.nii.gz"],
        "rois": [],
        "labelmaps": [],
        "label": 0.0
    }
}

with open('manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print("✅ Created manifest.json")
```

### 2. Load and use the dataset

```python
# train.py
from pyable_dataloader import PyableDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = PyableDataset(
    manifest='manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0
)

# Create dataloader
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Use it!
for batch in loader:
    images = batch['images']  # Shape: B × C × D × H × W
    labels = batch['label']
    print(f"Batch: images {images.shape}, labels {labels}")
    break

print("✅ Dataset working!")
```

## Test Your Installation (1 minute)

```bash
# Run tests
cd able-dataloader
pytest tests/test_basic.py -v
```

Expected output:
```
test_basic.py::test_dataset_creation PASSED
test_basic.py::test_multi_channel_stacking PASSED
test_basic.py::test_roi_masking PASSED
test_basic.py::test_label_preservation PASSED
test_basic.py::test_intensity_normalization PASSED
test_basic.py::test_random_flip PASSED
test_basic.py::test_caching PASSED
test_basic.py::test_dataloader_integration PASSED
test_basic.py::test_overlay_function PASSED

========== 9 passed in 5.23s ==========
```

## Convert Your Existing CSV Files

If you have existing CSV files (like your `train2.csv`):

```python
from examples.example_unified_nifti_style import convert_csv_to_manifest

# Convert CSV to JSON manifest
convert_csv_to_manifest(
    csv_file='train2.csv',
    csv_roi_file='trainROI2.csv',  # Optional
    output_json='train_manifest.json'
)

# Now use the manifest
dataset = PyableDataset(manifest='train_manifest.json', ...)
```

## Common Recipes

### Recipe 1: Simple Classification

```python
from pyable_dataloader import PyableDataset
from torch.utils.data import DataLoader

dataset = PyableDataset(
    manifest='data.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    cache_dir='cache'
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
```

### Recipe 2: With Data Augmentation

```python
from pyable_dataloader import (
    PyableDataset,
    Compose,
    IntensityNormalization,
    RandomFlip,
    RandomRotation90
)

transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomFlip(axes=[1, 2], prob=0.5),
    RandomRotation90(prob=0.3)
])

dataset = PyableDataset(
    manifest='train.json',
    target_size=[64, 64, 64],
    transforms=transforms
)
```

### Recipe 3: Multi-Modal (Multiple Images)

```python
manifest = {
    "subject_001": {
        "images": [
            "/data/T1.nii.gz",
            "/data/T2.nii.gz",
            "/data/FLAIR.nii.gz"
        ],
        "label": 1.0
    }
}

dataset = PyableDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    stack_channels=True  # Stack as C × D × H × W
)

# Will produce 3-channel images
sample = dataset[0]
print(sample['images'].shape)  # [3, 64, 64, 64]
```

### Recipe 4: With ROI Masking

```python
dataset = PyableDataset(
    manifest='data.json',
    target_size=[64, 64, 64],
    mask_with_roi=True,              # Multiply image by mask
    roi_labels=[1, 2, 3],            # Keep only these labels
    roi_center_target=[0, 0, 50]     # Center ROI at z=50mm
)
```

### Recipe 5: Inference with Overlay

```python
# Test dataset (no augmentation, return metadata)
dataset = PyableDataset(
    manifest='test.json',
    target_size=[64, 64, 64],
    transforms=None,
    return_meta=True
)

# Get sample
sample = dataset[0]
subject_id = sample['id']

# Run model
with torch.no_grad():
    prediction = model(sample['images'].unsqueeze(0))

# Overlay to original space
overlayer = dataset.get_original_space_overlayer(subject_id)
original_space = overlayer(
    prediction[0, 0].cpu().numpy(),
    interpolator='linear'
)

# Save
import SimpleITK as sitk
sitk.WriteImage(original_space, 'prediction.nii.gz')
```

## Troubleshooting

### Issue: "pyable not found"

```bash
# Install pyable first
cd /path/to/pyable
pip install -e .
```

### Issue: "Out of memory"

```python
# Reduce target_size or batch_size
dataset = PyableDataset(
    ...,
    target_size=[32, 32, 32],  # Smaller size
)

loader = DataLoader(dataset, batch_size=2)  # Smaller batch
```

### Issue: "Slow loading"

```python
# Enable caching
dataset = PyableDataset(
    ...,
    cache_dir='cache'  # Add cache directory
)

# Use more workers
loader = DataLoader(dataset, num_workers=4)
```

### Issue: "Labels are interpolated (non-integer values)"

This should NOT happen with the new dataset! If it does:

```python
# Make sure you're using Roiable/LabelMapable (automatic in PyableDataset)
# Check that original image has integer pixel type
import SimpleITK as sitk
img = sitk.ReadImage('roi.nii.gz')
print(img.GetPixelIDTypeAsString())  # Should be UInt8, Int16, etc.
```

## Performance Tips

1. **Use caching** - First epoch slow, rest fast
2. **Use multiple workers** - `num_workers=4` or more
3. **Pin memory** for GPU - `pin_memory=True`
4. **Persistent workers** - `persistent_workers=True`

```python
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

## Next Steps

1. ✅ Read `README.md` for complete documentation
2. ✅ Check `examples/` for full working examples
3. ✅ Read `IMPLEMENTATION_SUMMARY.md` for migration guide
4. ✅ Run `pytest tests/` to verify everything works

## Getting Help

- Check `README.md` first
- Look at `examples/` for similar use cases
- Read `pyable/docs/LLM_DESCRIPTOR.yaml` for pyable API
- Check test files for usage patterns

---

**You're ready to go!** 🚀

Start with a simple example, then add complexity as needed. The package is designed to be intuitive and handle the hard parts automatically.
