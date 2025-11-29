# Agent Handoff: PyTorch DataLoader Integration Task

## Context
You are taking over the development of `pyable-dataloader`, a PyTorch Dataset for medical imaging that integrates with two existing packages: `pyable` (image processing) and `pyfe` (feature extraction/radiomics).

## Current State

### Project Structure
You have access to three sibling directories:
- **`/home/erosm/pyable/`** - Core medical image processing library (SimpleITK wrapper)
- **`/home/erosm/able-dataloader/`** - PyTorch dataloader (this project - YOUR PRIMARY FOCUS)
- **`/home/erosm/pyfe/`** - Feature extraction and radiomics library

### Dependency Chain
```
pyfe → able-dataloader → pyable
```
This avoids circular dependencies.

### What's Already Done
✅ Core `PyableDataset` class created in `src/pyable_dataloader/dataset.py`
✅ Basic transforms implemented in `src/pyable_dataloader/transforms.py`
✅ Initial documentation written
✅ `pyfe/pyfe/dataloader.py` created (re-exports from pyable_dataloader)
✅ `pyfe/pyproject.toml` updated with dataloader as optional dependency

### What's NOT Done (Your Tasks)
❌ Advanced spatial transforms not implemented (RandomTranslation, RandomRotation, RandomAffine, RandomBSpline)
❌ Integration testing between all three packages
❌ Verify no circular import issues
❌ Test that pyfe can properly use the dataloader
❌ Ensure label preservation during augmentation

## Your Primary Tasks

### Task 1: Implement Advanced Spatial Transforms
The user wants augmentation transforms that accept per-axis ranges for translation, rotation, scaling, and shear. Similar to their existing `augmentem()` function in the bio_toolboxes code they provided.

**Requirements:**
- Accept ranges like `[[-5, 5], [-5, 5], [-2, 2]]` for per-axis control (X, Y, Z in mm/degrees)
- Support `center='image'` or `center='femur'` or `center=(x,y,z)` for rotation/scaling center
- Work in physical space (mm) using SimpleITK transforms
- Automatically preserve labels (use nearest-neighbor for Roiable/LabelMapable)
- Support these transforms:
  - `RandomTranslation(translation_range, p, center, random_state)`
  - `RandomRotation(rotation_range_deg, p, center, random_state)`
  - `RandomAffine(translation_range, rotation_range, scale_range, shear_range, p, center, random_state)`
  - `RandomBSpline(grid_physical_spacing, max_displacement, sigma, p, random_state)`

**Implementation approach:**
- Use `sitk.TranslationTransform`, `sitk.Euler3DTransform`, `sitk.AffineTransform`, `sitk.BSplineTransform`
- Apply transforms via `Imaginable.applyTransform(transform)` which handles interpolator selection
- Transforms should work on Imaginable objects, not numpy arrays (spatial transforms in physical space)
- After spatial transforms, convert to numpy and apply intensity transforms

**Files to modify:**
- `able-dataloader/src/pyable_dataloader/transforms.py`
- `able-dataloader/src/pyable_dataloader/__init__.py` (add new transforms to exports)

### Task 2: Verify Integration with pyfe
Ensure pyfe can import and use the dataloader without issues.

**Test this:**
```python
# From pyfe directory
from pyfe.dataloader import PyableDataset, Compose, RandomTranslation
# Should work without errors
```

**Check:**
- No circular imports
- `pyfe/pyfe/dataloader.py` properly re-exports all transforms
- `DATALOADER_AVAILABLE` flag works correctly
- Both import paths work: `from pyfe.dataloader import X` and `from pyable_dataloader import X`

### Task 3: Label Preservation Testing
Critical: Verify that after augmentation, labelmap voxels contain ONLY the original label values (no interpolated values).

**Create test:**
```python
# Test that RandomRotation + RandomAffine preserve labels
labelmap = LabelMapable('seg.nii')
original_labels = set(labelmap.getImageUniqueValues())

transform = Compose([
    RandomRotation([[-15, 15], [-15, 15], [-10, 10]], p=1.0),
    RandomAffine(rotation_range=[[-20, 20], [-20, 20], [-10, 10]], p=1.0)
])

transformed_labelmap, _, _ = transform(labelmap, rois=None, labelmaps=None)
new_labels = set(transformed_labelmap.getImageUniqueValues())

assert new_labels.issubset(original_labels), "Labels were interpolated!"
```

### Task 4: Update Documentation
Update these files to reflect the new transforms:
- `able-dataloader/README.md` - Add transform examples
- `able-dataloader/QUICKSTART.md` - Show how to use spatial transforms
- `able-dataloader/PYFE_INTEGRATION.md` - Update with complete examples
- `able-dataloader/__init__.py` - Export new transforms

## Key Implementation Details

### pyable Package API (Your Dependency)
Located in: `/home/erosm/pyable/pyable/imaginable.py`

**Key Classes:**
- `SITKImaginable` - Base class for continuous/grayscale images
- `Roiable` - Binary masks (uses nearest-neighbor interpolation automatically)
- `LabelMapable` - Multi-label segmentations (uses nearest-neighbor automatically)
- `Fieldable` - Vector fields/deformations

**Key Methods (v3 API):**
```python
# Coordinate conversions
img.getImageAsNumpyZYX()  # Returns (Z,Y,X) array - STANDARD in v3
img.setImageFromNumpyZYX(arr)  # Sets from (Z,Y,X) array
img.getPhysicalPointFromArrayIndex(k, j, i)  # Array index → physical mm
img.getArrayIndexFromPhysicalPoint(x, y, z)  # Physical mm → array index

# Spatial transforms
img.applyTransform(sitk_transform, target_image=None, interpolator=None)
# - For Roiable/LabelMapable: automatically uses sitk.sitkNearestNeighbor
# - For SITKImaginable: uses sitk.sitkLinear by default
# - Returns new Imaginable object with transformed image

# Resampling
img.resampleOnTargetImage(target_img)  # Resample to match target geometry
img.resampleOnCanonicalSpace()  # Resample to axis-aligned LPS orientation
img.changeImageSpacing(new_spacing)  # Resample to new voxel size

# Metadata
img.getImageSize()  # (Nx, Ny, Nz) in ITK convention (X,Y,Z)
img.getImageSpacing()  # (sx, sy, sz) in mm
img.getImageOrigin()  # (ox, oy, oz) in mm
img.getImageDirection()  # 9-element direction cosine matrix
img.getImage()  # Returns sitk.Image object
```

**Important Convention:**
- Physical space: (X, Y, Z) in mm - right-handed coordinate system
- Array space: (Z, Y, X) = (K, J, I) - numpy indexing
- ITK/SimpleITK space: (i, j, k) = (X, Y, Z) - SimpleITK indexing

### pyfe Package Structure
Located in: `/home/erosm/pyfe/pyfe/`

**Files:**
- `pyfe.py` - Main radiomics extraction (PYRAD class)
- `pyml.py` - Machine learning utilities
- `dataloader.py` - YOUR INTEGRATION POINT (re-exports from pyable_dataloader)

**Current dataloader.py:**
- Imports from `pyable_dataloader` if available
- Sets `DATALOADER_AVAILABLE = True/False` flag
- Provides stubs if not installed

### able-dataloader Package Structure
Located in: `/home/erosm/able-dataloader/src/pyable_dataloader/`

**Your files:**
- `dataset.py` - PyableDataset class (✅ implemented)
- `transforms.py` - Transform classes (⚠️ needs spatial transforms)
- `__init__.py` - Package exports

## User's Requirements (From Their Code)

The user provided an `augmentem()` function showing their desired API:

```python
aug = {
    "version": 0,
    "n": 1,  # number of augmentations
    "options": {
        "d": 3,  # dimension
        "r": [[-1, 1]],  # rotations per axis [min, max] in degrees
        "t": [[-1, 1]],  # translations per axis [min, max] in mm
        "s": [[1, 1]],   # scaling per axis [min, max] multiplicative
        "CIR": [10, 10, 10],  # center of rotation
        "isCIRindex": True,  # whether CIR is in index or physical coords
        "resample": [4, 4, 4]  # optional resampling after transform
    }
}
```

**Your transforms should support:**
- Per-axis ranges: `[[-5, 5], [-5, 5], [-2, 2]]` (3 separate ranges for X, Y, Z)
- Scalar shortcuts: `[-5, 5]` → same range for all axes
- Center specification: `center='image'`, `center='femur'`, `center=[x, y, z]`
- Probability: `p=0.8` (80% chance to apply)
- Random seed: `random_state=42` for reproducibility

## Testing Strategy

### Quick Integration Test
Create this file: `able-dataloader/tests/test_pyfe_integration.py`

```python
def test_pyfe_can_import_dataloader():
    """Test that pyfe can import the dataloader"""
    from pyfe.dataloader import DATALOADER_AVAILABLE, PyableDataset
    assert DATALOADER_AVAILABLE == True
    assert PyableDataset is not None

def test_transforms_available_from_pyfe():
    """Test all transforms can be imported from pyfe"""
    from pyfe.dataloader import (
        Compose,
        RandomTranslation,
        RandomRotation,
        RandomAffine,
        RandomBSpline,
        IntensityNormalization
    )
    # Should not raise ImportError

def test_no_circular_imports():
    """Ensure no circular import issues"""
    import pyable
    import pyable_dataloader
    import pyfe
    # If we get here, no circular imports
```

### Label Preservation Test
Create: `able-dataloader/tests/test_label_preservation.py`

```python
def test_spatial_transforms_preserve_labels():
    """Verify spatial transforms don't interpolate label values"""
    # Create synthetic labelmap with labels [0, 1, 2, 3]
    # Apply RandomRotation, RandomAffine, RandomBSpline
    # Check unique values are still subset of [0, 1, 2, 3]
```

## Common Pitfalls to Avoid

1. **Don't transpose manually** - `getImageAsNumpyZYX()` already returns (Z,Y,X), don't transpose again
2. **Don't apply SITK transforms to numpy arrays** - Use `Imaginable.applyTransform()` on Imaginable objects
3. **Don't use linear interpolation for labels** - Roiable/LabelMapable handle this automatically
4. **Don't create symlinks** - You have full directory access, use relative imports or installed packages
5. **Don't modify pyable** - It's a stable dependency, work with its existing API
6. **Don't make pyable depend on pyfe or able-dataloader** - Keep dependency chain unidirectional

## Success Criteria

You'll know you're done when:

✅ All spatial transforms implemented (RandomTranslation, RandomRotation, RandomAffine, RandomBSpline)
✅ `from pyfe.dataloader import RandomAffine` works
✅ Label preservation test passes
✅ Can create a dataset with spatial augmentation and train a model
✅ No circular import errors when importing all three packages
✅ Documentation updated with spatial transform examples
✅ All exports in `__init__.py` include new transforms

## Example Final Usage

This should work after your changes:

```python
# From anywhere, import via pyfe
from pyfe.dataloader import (
    PyableDataset,
    Compose,
    RandomTranslation,
    RandomRotation,
    RandomAffine,
    IntensityNormalization
)

# Configure augmentation
transforms = Compose([
    RandomTranslation(
        translation_range=[[-5, 5], [-5, 5], [-2, 2]],
        p=0.8,
        center='femur'
    ),
    RandomRotation(
        rotation_range_deg=[[-15, 15], [-15, 15], [-10, 10]],
        p=0.5,
        center='femur'
    ),
    RandomAffine(
        translation_range=[[-3, 3], [-3, 3], [-1, 1]],
        rotation_range=[[-10, 10], [-10, 10], [-5, 5]],
        scale_range=[[0.9, 1.1], [0.9, 1.1], [0.95, 1.05]],
        shear_range=[[-0.1, 0.1], [-0.1, 0.1], [-0.05, 0.05]],
        p=0.3,
        center='image'
    ),
    IntensityNormalization(method='zscore')
])

# Create dataset
dataset = PyableDataset(
    manifest='data.json',
    target_size=[128, 128, 64],
    target_spacing=2.0,
    transforms=transforms,
    cache_dir='./cache'
)

# Use in training
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    images = batch['images']  # [B, C, D, H, W]
    labels = batch['label']
    # Train your model
```

## Files You'll Need to Read/Modify

**Must Read:**
- `/home/erosm/pyable/pyable/imaginable.py` - Understand the API
- `/home/erosm/able-dataloader/src/pyable_dataloader/transforms.py` - Add spatial transforms here
- `/home/erosm/able-dataloader/src/pyable_dataloader/dataset.py` - Understand how transforms are called

**Must Modify:**
- `/home/erosm/able-dataloader/src/pyable_dataloader/transforms.py` - Implement transforms
- `/home/erosm/able-dataloader/src/pyable_dataloader/__init__.py` - Export new transforms
- `/home/erosm/pyfe/pyfe/dataloader.py` - Add new transforms to re-exports

**Must Test:**
- Create `able-dataloader/tests/test_spatial_transforms.py`
- Create `able-dataloader/tests/test_pyfe_integration.py`
- Update `able-dataloader/tests/test_basic.py` if needed

**Must Update:**
- `/home/erosm/able-dataloader/README.md`
- `/home/erosm/able-dataloader/QUICKSTART.md`
- `/home/erosm/able-dataloader/PYFE_INTEGRATION.md`

## Questions to Ask User (If Needed)

If you're stuck or need clarification:
1. "Should B-spline transforms be applied to labelmaps, or only continuous images?"
2. "For femur center, should Dataset automatically detect femur ROI and compute center of mass?"
3. "Should transforms operate before or after resampling to target grid?"
4. "Do you want a config-based transform factory (JSON → transforms)?"

## Reference: User's Existing Code Style

The user likes this pattern from their `augmentem()` function:
- Config dict with nested options
- Per-axis control with fallback to same value for all axes
- Center of rotation/scaling configurable
- Apply same transform to all images and ROIs in a sample
- Random sampling from ranges

Match this style in your transform API design.

## Start Here

1. Read `/home/erosm/pyable/pyable/imaginable.py` to understand `applyTransform()`
2. Implement `RandomTranslation` in `transforms.py` (simplest case)
3. Test it works and preserves labels
4. Implement `RandomRotation`, then `RandomAffine`, then `RandomBSpline`
5. Update exports and test integration with pyfe
6. Update documentation

Good luck! The core dataset and basic transforms are already solid. You're adding the spatial augmentation layer that makes this production-ready for deep learning.
