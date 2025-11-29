# Installation and Verification Checklist

## ✅ Pre-Installation Checklist

- [ ] Python 3.8+ installed
- [ ] pyable package installed (`cd pyable && pip install -e .`)
- [ ] PyTorch installed
- [ ] SimpleITK installed (should come with pyable)

## ✅ Installation Steps

```bash
# 1. Navigate to project
cd /home/erosm/able-dataloader

# 2. Install in editable mode
pip install -e .

# 3. Install dev dependencies (optional)
pip install pytest pytest-cov
```

## ✅ Verification Steps

### Step 1: Test Import
```bash
python -c "from pyable_dataloader import PyableDataset; print('✅ Import successful!')"
```
Expected: `✅ Import successful!`

### Step 2: Run Tests
```bash
pytest tests/test_basic.py -v
```
Expected: All 9 tests pass ✅

### Step 3: Check Dependencies
```bash
python -c "
import torch
import numpy
import pandas
import scipy
import SimpleITK as sitk
from pyable.imaginable import SITKImaginable
print('✅ All dependencies available!')
"
```
Expected: `✅ All dependencies available!`

## ✅ Quick Functionality Test

Create this test file: `test_installation.py`

```python
#!/usr/bin/env python3
"""Quick installation test."""

import tempfile
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from pyable_dataloader import PyableDataset, IntensityNormalization

# Create temporary directory
tmpdir = Path(tempfile.mkdtemp())

# Create synthetic image
arr = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
img = sitk.GetImageFromArray(arr)
img.SetSpacing([1.0, 1.0, 1.0])
img_path = tmpdir / 'test_image.nii.gz'
sitk.WriteImage(img, str(img_path))

# Create manifest
manifest = {
    "test_001": {
        "images": [str(img_path)],
        "rois": [],
        "labelmaps": [],
        "label": 1.0
    }
}

manifest_path = tmpdir / 'manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f)

# Create dataset
dataset = PyableDataset(
    manifest=str(manifest_path),
    target_size=[16, 16, 16],
    target_spacing=2.0,
    transforms=IntensityNormalization(method='zscore')
)

# Get sample
sample = dataset[0]

# Verify
assert 'images' in sample
assert 'label' in sample
assert sample['images'].shape[-3:] == (16, 16, 16)
assert sample['label'].item() == 1.0

print("✅ All checks passed!")
print(f"   - Dataset created: {len(dataset)} samples")
print(f"   - Sample loaded: images shape = {sample['images'].shape}")
print(f"   - Transform applied: normalized")
print("\n🎉 Installation verified successfully!")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
```

Run it:
```bash
python test_installation.py
```

Expected output:
```
✅ All checks passed!
   - Dataset created: 1 samples
   - Sample loaded: images shape = torch.Size([1, 16, 16, 16])
   - Transform applied: normalized

🎉 Installation verified successfully!
```

## ✅ Performance Test (Optional)

Create `test_performance.py`:

```python
#!/usr/bin/env python3
"""Test caching performance."""

import time
import tempfile
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from pyable_dataloader import PyableDataset

# Setup
tmpdir = Path(tempfile.mkdtemp())
cache_dir = tmpdir / 'cache'

# Create synthetic images
print("Creating test data...")
for i in range(5):
    arr = np.random.randn(32, 32, 32).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(img, str(tmpdir / f'img_{i}.nii.gz'))

manifest = {
    f"sub_{i:03d}": {
        "images": [str(tmpdir / f'img_{i}.nii.gz')],
        "label": float(i % 2)
    }
    for i in range(5)
}

manifest_path = tmpdir / 'manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f)

# First pass (no cache)
print("\n1st pass (no cache)...")
dataset = PyableDataset(
    manifest=str(manifest_path),
    target_size=[16, 16, 16],
    cache_dir=str(cache_dir)
)

start = time.time()
for i in range(len(dataset)):
    _ = dataset[i]
time_no_cache = time.time() - start

# Second pass (cached)
print("2nd pass (cached)...")
dataset2 = PyableDataset(
    manifest=str(manifest_path),
    target_size=[16, 16, 16],
    cache_dir=str(cache_dir)
)

start = time.time()
for i in range(len(dataset2)):
    _ = dataset2[i]
time_cached = time.time() - start

# Results
print("\n" + "="*60)
print("Performance Results:")
print("="*60)
print(f"No cache:  {time_no_cache:.3f}s ({time_no_cache/5:.3f}s per sample)")
print(f"Cached:    {time_cached:.3f}s ({time_cached/5:.3f}s per sample)")
print(f"Speedup:   {time_no_cache/time_cached:.1f}x")
print("="*60)

if time_no_cache / time_cached > 5:
    print("✅ Caching working efficiently!")
else:
    print("⚠️  Caching speedup less than expected (but still working)")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
```

Run it:
```bash
python test_performance.py
```

Expected: Significant speedup on 2nd pass (>5x)

## ✅ Integration Test (Optional)

Test with PyTorch DataLoader:

```python
#!/usr/bin/env python3
"""Test PyTorch DataLoader integration."""

import tempfile
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from pyable_dataloader import PyableDataset

# Setup
tmpdir = Path(tempfile.mkdtemp())

# Create data
for i in range(10):
    arr = np.random.randn(32, 32, 32).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(img, str(tmpdir / f'img_{i}.nii.gz'))

manifest = {
    f"sub_{i:03d}": {
        "images": [str(tmpdir / f'img_{i}.nii.gz')],
        "label": float(i % 2)
    }
    for i in range(10)
}

manifest_path = tmpdir / 'manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f)

# Create dataset
dataset = PyableDataset(
    manifest=str(manifest_path),
    target_size=[16, 16, 16]
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# Test iteration
print("Testing DataLoader...")
for batch_idx, batch in enumerate(loader):
    images = batch['images']
    labels = batch['label']
    
    print(f"Batch {batch_idx}: images {images.shape}, labels {labels.shape}")
    
    assert images.ndim == 5  # B × C × D × H × W
    assert labels.ndim == 1  # B
    assert images.shape[0] == labels.shape[0]  # Same batch size

print("\n✅ DataLoader integration working!")
print(f"   - Processed {len(loader)} batches")
print(f"   - Batch size: 4")
print(f"   - Workers: 2")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
```

Run it:
```bash
python test_integration.py
```

## ✅ Final Checklist

- [ ] All imports work
- [ ] Unit tests pass (9/9)
- [ ] Installation test passes
- [ ] Performance test shows caching speedup
- [ ] DataLoader integration test passes
- [ ] Can load sample data
- [ ] Can apply transforms
- [ ] Can create batches

## 🎉 If All Checks Pass

You're ready to use the dataloader in production! 

Next steps:
1. Convert your existing CSV files to manifests
2. Update your training scripts
3. Run training with new dataloader
4. Verify results match or improve upon original

## ❌ If Any Check Fails

### Import Errors
```bash
# Reinstall pyable
cd /path/to/pyable
pip install -e .

# Reinstall dataloader
cd /home/erosm/able-dataloader
pip install -e .
```

### Test Failures
```bash
# Check dependencies
pip list | grep -E "torch|numpy|pandas|scipy|SimpleITK"

# Reinstall from scratch
pip uninstall pyable-dataloader
pip install -e .
```

### Performance Issues
- Check if cache directory is on SSD (not network drive)
- Verify sufficient disk space
- Try reducing num_workers if memory-constrained

## 📞 Need Help?

Check these resources in order:
1. `QUICKSTART.md` - Quick start guide
2. `README.md` - Complete documentation
3. `IMPLEMENTATION_SUMMARY.md` - Migration guide
4. `ARCHITECTURE.md` - How it works
5. `tests/test_basic.py` - Usage examples

---

**Good luck! The package is ready to use.** 🚀
