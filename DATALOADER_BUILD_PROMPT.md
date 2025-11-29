# PyTorch DataLoader for Medical Images - Build Prompt

## Goal
Create a production-ready PyTorch Dataset package (`pyable-dataloader`) for loading medical images using the existing `pyable` package (v3). The dataloader must seamlessly integrate with pyable's API and follow medical imaging best practices.

---

## 🔑 Critical Context: The pyable Package

**You have access to the complete pyable package in this workspace at `./pyable/`**

### Essential Documentation (READ THESE FIRST):
1. **`./pyable/docs/LLM_DESCRIPTOR.yaml`** - Complete API reference optimized for LLMs
2. **`./pyable/PYABLE_V3_BREAKING_CHANGES.md`** - V3 conventions (ZYX ordering!)
3. **`./pyable/ORIENTATION_GUIDE.md`** - Coordinate systems and resampling
4. **`./pyable/docs/DEFORMATION_WORKFLOW.md`** - Transform/registration workflow
5. **`./pyable/QUICKREF.md`** - Quick reference guide

### Key pyable v3 Conventions (CRITICAL):
- **NumPy arrays are Z,Y,X ordered** (not X,Y,Z!) via `getImageAsNumpy()` or `getImageAsNumpyZYX()`
- **PyTorch tensors should be C × D × H × W** (channels × depth × height × width)
- **Label-preserving transforms**: Integer images automatically use nearest-neighbor interpolation
- **Standard orientation**: LPS (Left-Posterior-Superior) via `resampleOnCanonicalSpace()` or `reorientToLPS()`

### Core pyable Classes to Use:
```python
from pyable.imaginable import (
    SITKImaginable,  # For continuous/grayscale images
    Roiable,         # For ROIs (binary masks)
    LabelMapable,    # For multi-label segmentations
    Fieldable        # For vector fields/deformations
)
```

### Key Methods You'll Need:
```python
# Loading
img = SITKImaginable(filename='path.nii.gz')
roi = Roiable(filename='roi.nii.gz')

# Resampling to reference space
img.resampleOnTargetImage(reference_img)           # Resample to match reference geometry
img.resampleOnCanonicalSpace()                     # Resample to LPS axis-aligned
img.resampleToAxisAligned()                        # For oblique acquisitions

# Getting numpy arrays (Z,Y,X)
arr = img.getImageAsNumpy()        # Returns (Z,Y,X) for 3D, (Y,X) for 2D
arr = img.getImageAsNumpyZYX()     # Explicit alias

# Applying transforms (e.g., from registration)
img.applyTransform(
    transform=sitk_transform,
    target_image=reference_img,    # Optional: resample to this space
    interpolator=None,             # Uses class default
    default_value=0
)

# Orientation queries
code = img.getOrientationCode()    # Returns 'LPS', 'RAS', etc.
aligned = img.isAxisAligned()      # True if not oblique

# Getting ITK/VTK objects
sitk_img = img.getITKImage()       # Get SimpleITK.Image
vtk_img = img.getVTKImage()        # Get VTK image
```

---

## 📦 Project Structure

Create package under: `./src/pyable_dataloader/`

```
pyable-dataloader/
├── pyproject.toml                 # Package metadata & dependencies
├── README.md                      # Installation & usage guide
├── src/
│   └── pyable_dataloader/
│       ├── __init__.py
│       ├── dataset.py             # Main PyableDataset class
│       ├── transforms.py          # Transform adapters & utilities
│       ├── collate.py             # Batch collation for variable shapes
│       └── utils.py               # Manifest parsing, caching
├── tests/
│   ├── test_dataset_basic.py
│   ├── test_dataset_labels.py
│   ├── test_dataset_resample.py
│   ├── test_transforms.py
│   └── conftest.py                # pytest fixtures
├── examples/
│   ├── example_basic_usage.py
│   ├── example_with_transforms.py
│   ├── example_registration_workflow.py
│   └── data/
│       └── sample_manifest.json
└── docs/
    ├── API.md
    └── TRANSFORMS_GUIDE.md
```

---

## 🎯 Dataset API Specification

### Main Class: `PyableDataset`

```python
class PyableDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for loading medical images via pyable.
    
    Supports:
    - Multiple images per subject (stacked as channels)
    - ROIs and labelmaps with label-preserving resampling
    - Automatic resampling to reference space
    - Transform/augmentation pipeline
    - Caching for performance
    - Registration output integration
    """
    
    def __init__(
        self,
        manifest: Union[str, dict, List[str]],
        reference_selector: Union[str, int, Callable] = 'first',
        reference_space: Optional[Union[str, dict]] = None,
        transforms: Optional[Callable] = None,
        stack_channels: bool = True,
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        return_meta: bool = True
    ):
        """
        Args:
            manifest: Path to JSON/CSV file(s) or dict with structure:
                {
                    "subject_id": {
                        "images": [path1, path2, ...],
                        "rois": [path1, ...],
                        "labelmaps": [path1, ...],
                        "reference": int or path (optional),
                        "transforms": [transform_paths] (optional)
                    }
                }
            
            reference_selector: How to choose reference image per subject
                - 'first': Use first image
                - int: Use image at this index
                - callable: Function(images_list) -> int
            
            reference_space: Global reference space (optional)
                - str: Path to reference image file
                - dict: {'spacing': [x,y,z], 'size': [x,y,z], 'origin': [...], 'direction': [...]}
                If provided, ALL subjects resampled to this space
            
            transforms: Callable(images, rois, labelmaps, meta) -> (images, rois, labelmaps)
                Applied after resampling to reference space
            
            stack_channels: If True, stack multiple images as channels (C × D × H × W)
            
            cache_dir: Optional directory to cache resampled results
            
            dtype: PyTorch dtype for output tensors
            
            return_meta: If True, include metadata dict in output
        """
```

### Output Format

```python
sample = dataset[idx]
# Returns dict:
{
    'id': str,                          # Subject identifier
    'images': torch.Tensor,             # Shape: C × D × H × W (or D × H × W if C=1)
    'rois': List[torch.Tensor],         # Each shape: D × H × W
    'labelmaps': List[torch.Tensor],    # Each shape: D × H × W
    'meta': {                           # If return_meta=True
        'spacing': Tuple[float],        # Physical spacing (mm)
        'origin': Tuple[float],         # Physical origin
        'direction': Tuple[float],      # Direction cosines
        'orientation': str,             # 'LPS', 'RAS', etc.
        'source_paths': dict,           # Original file paths
        'reference_used': str,          # Which reference was used
        'transforms_applied': List[str] # Transform descriptions
    }
}
```

---

## 🔧 Functional Requirements

### 1. Manifest Parsing (`utils.py`)

Support these input formats:

**JSON format:**
```json
{
    "subject_001": {
        "images": [
            "/data/sub001/T1w.nii.gz",
            "/data/sub001/T2w.nii.gz"
        ],
        "rois": ["/data/sub001/brain_mask.nii.gz"],
        "labelmaps": ["/data/sub001/aparc+aseg.nii.gz"],
        "reference": 0,
        "reference_space": "/data/templates/MNI152_1mm.nii.gz"
    },
    "subject_002": { ... }
}
```

**CSV format:**
```csv
id,image_paths,roi_paths,labelmap_paths,reference,reference_space
sub001,"['/data/sub001/T1.nii.gz','/data/sub001/T2.nii.gz']",/data/sub001/mask.nii.gz,,0,
sub002,/data/sub002/T1.nii.gz,,,first,/templates/MNI.nii.gz
```

**Multi-CSV format:**
```csv
# images.csv
id,sequence,path,is_reference
sub001,T1,/data/sub001/T1.nii.gz,true
sub001,T2,/data/sub001/T2.nii.gz,false

# rois.csv
id,roi_type,path
sub001,brain_mask,/data/sub001/mask.nii.gz
```

### 2. Reference Space Selection Logic

**Per-subject reference** (most common):
```python
# Load all images for subject
images = [SITKImaginable(p) for p in paths]

# Select reference
if manifest has 'reference':
    if isinstance(reference, int):
        ref = images[reference]
    elif isinstance(reference, str):  # path
        ref = SITKImaginable(reference)
elif reference_selector == 'first':
    ref = images[0]
elif reference_selector == 'largest':
    ref = max(images, key=lambda x: np.prod(x.getImageSize()))
elif callable(reference_selector):
    idx = reference_selector(images)
    ref = images[idx]

# Resample all to reference
for img in images:
    if img is not ref:
        img.resampleOnTargetImage(ref)
```

**Global reference space** (for standardized analysis):
```python
if reference_space is not None:
    if isinstance(reference_space, str):
        global_ref = SITKImaginable(reference_space)
    else:  # dict with spacing/size/etc
        global_ref = create_reference_from_params(reference_space)
    
    # Resample ALL subjects to this space
    for img in images:
        img.resampleOnTargetImage(global_ref)
```

### 3. Resampling Pipeline

**Critical: Label preservation!**

```python
def resample_to_reference(imaginable, reference):
    """
    Resample with appropriate interpolation based on image type.
    
    pyable handles this automatically:
    - Integer pixel types → nearest neighbor (labels preserved)
    - Float pixel types → linear interpolation
    """
    # Create copy to avoid modifying original
    resampled = type(imaginable)(image=sitk.Image(imaginable.getImage()))
    
    # This automatically uses correct interpolator based on pixel type
    resampled.resampleOnTargetImage(reference)
    
    return resampled
```

### 4. Transform Integration (Registration Outputs)

```python
def apply_transform_pipeline(imaginable, transforms, target_space=None):
    """
    Apply SimpleITK transforms from registration.
    
    Args:
        imaginable: SITKImaginable/Roiable/LabelMapable
        transforms: List of SimpleITK Transform objects or paths
        target_space: Optional target image to resample into
    """
    result = imaginable
    
    for transform in transforms:
        if isinstance(transform, str):
            transform = sitk.ReadTransform(transform)
        
        # applyTransform handles label vs continuous automatically
        result.applyTransform(
            transform=transform,
            target_image=target_space,
            interpolator=None,  # Uses class default
            default_value=0
        )
    
    return result
```

### 5. Conversion to PyTorch

```python
def imaginable_to_tensor(imaginable, dtype=torch.float32):
    """
    Convert pyable Imaginable to PyTorch tensor.
    
    Returns:
        torch.Tensor: Shape (D, H, W) for 3D, (H, W) for 2D
    """
    # getImageAsNumpy() returns (Z,Y,X) in v3
    arr = imaginable.getImageAsNumpy()
    
    # Convert to torch (maintains Z,Y,X → D,H,W mapping)
    tensor = torch.from_numpy(arr).to(dtype)
    
    return tensor

def stack_as_channels(tensors):
    """Stack multiple images as channels."""
    return torch.stack(tensors, dim=0)  # C × D × H × W
```

### 6. Caching Strategy

```python
def get_cache_key(subject_id, file_paths, reference_geometry, transforms):
    """
    Generate deterministic cache key.
    
    Include: subject_id, file hashes, reference spacing/size/origin,
             transform fingerprints
    """
    import hashlib
    
    key_parts = [
        subject_id,
        *[hash_file(p) for p in file_paths],
        str(reference_geometry),
        *[str(t) for t in transforms]
    ]
    
    return hashlib.md5('_'.join(key_parts).encode()).hexdigest()

def load_or_compute_cached(cache_dir, cache_key, compute_fn):
    """Load from cache or compute and save."""
    cache_path = Path(cache_dir) / f"{cache_key}.npy"
    
    if cache_path.exists():
        return np.load(cache_path)
    else:
        result = compute_fn()
        np.save(cache_path, result)
        return result
```

---

## 🎨 Transform System

### Transform Interface

```python
class MedicalImageTransform:
    """
    Base class for transforms.
    
    Transforms receive:
    - images: torch.Tensor (C × D × H × W)
    - rois: List[torch.Tensor] (each D × H × W)
    - labelmaps: List[torch.Tensor]
    - meta: dict with spacing, orientation, etc.
    
    Returns: Same structure with transforms applied
    """
    
    def __call__(self, images, rois, labelmaps, meta):
        raise NotImplementedError
```

### Example Transforms (include in `transforms.py`)

```python
class IntensityNormalization(MedicalImageTransform):
    """Z-score normalization per channel."""
    def __call__(self, images, rois, labelmaps, meta):
        # Normalize each channel independently
        normalized = []
        for c in range(images.shape[0]):
            channel = images[c]
            mean = channel.mean()
            std = channel.std()
            normalized.append((channel - mean) / (std + 1e-8))
        
        return torch.stack(normalized), rois, labelmaps

class RandomFlip(MedicalImageTransform):
    """Random axis flip (maintains Z,Y,X ordering)."""
    def __call__(self, images, rois, labelmaps, meta):
        if random.random() > 0.5:
            axis = random.choice([0, 1, 2])  # D, H, or W
            images = torch.flip(images, dims=[axis + 1])  # +1 for channel dim
            rois = [torch.flip(r, dims=[axis]) for r in rois]
            labelmaps = [torch.flip(lm, dims=[axis]) for lm in labelmaps]
        
        return images, rois, labelmaps

class RandomRotation90(MedicalImageTransform):
    """Random 90-degree rotation in axial plane."""
    def __call__(self, images, rois, labelmaps, meta):
        if random.random() > 0.5:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            # Rotate in H-W plane (last two dims)
            images = torch.rot90(images, k=k, dims=[-2, -1])
            rois = [torch.rot90(r, k=k, dims=[-2, -1]) for r in rois]
            labelmaps = [torch.rot90(lm, k=k, dims=[-2, -1]) for lm in labelmaps]
        
        return images, rois, labelmaps

class Compose(MedicalImageTransform):
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, images, rois, labelmaps, meta):
        for t in self.transforms:
            images, rois, labelmaps = t(images, rois, labelmaps, meta)
        return images, rois, labelmaps
```

---

## 🧪 Testing Requirements

### Test Files to Create

**`test_dataset_basic.py`:**
```python
def test_single_image_loading():
    """Load single 3D image, verify shape and dtype."""
    
def test_multi_channel_stacking():
    """Load multiple images, verify stacked as C × D × H × W."""
    
def test_roi_loading():
    """Load image + ROI, verify both have same spatial dimensions."""
```

**`test_dataset_labels.py`:**
```python
def test_label_preservation():
    """
    Create label image with discrete values [0, 1, 2, 3].
    Resample to different resolution.
    Assert: output only contains original label values (no interpolated values).
    """
    
def test_roi_binary_preservation():
    """Binary ROI should remain binary after resampling."""
```

**`test_dataset_resample.py`:**
```python
def test_reference_space_matching():
    """
    Load two images with different spacings.
    After resampling to reference, verify:
    - Same size
    - Same spacing
    - Same origin
    """

def test_global_reference_space():
    """
    Use global reference space for all subjects.
    Verify all outputs have identical geometry.
    """
```

**`test_transforms.py`:**
```python
def test_intensity_normalization():
    """Verify mean≈0, std≈1 after normalization."""
    
def test_random_flip_shape():
    """Shape unchanged after flip."""
    
def test_transform_label_preservation():
    """Labels not interpolated during augmentation."""
```

### Test Data Creation

Use synthetic SimpleITK images (avoid external files):

```python
import SimpleITK as sitk
import numpy as np

def create_test_image_3d(size=(32, 32, 32), spacing=(1.0, 1.0, 1.0)):
    """Create synthetic 3D image."""
    arr = np.random.randn(*size).astype(np.float32)
    img = sitk.GetImageFromArray(arr.transpose(2, 1, 0))  # XYZ for SITK
    img.SetSpacing(spacing)
    return img

def create_test_labelmap(size=(32, 32, 32), num_labels=4):
    """Create synthetic label map."""
    arr = np.random.randint(0, num_labels, size, dtype=np.uint8)
    img = sitk.GetImageFromArray(arr.transpose(2, 1, 0))
    return img
```

---

## 📖 Documentation Requirements

### README.md Structure

1. **Installation**
   ```bash
   # Install pyable in editable mode
   cd pyable
   pip install -e .
   
   # Install dataloader
   cd ../pyable-dataloader
   pip install -e .
   ```

2. **Quick Start**
   - Minimal working example (5-10 lines)
   - Expected output shapes

3. **Manifest Format Guide**
   - JSON example
   - CSV example
   - Field descriptions

4. **Common Use Cases**
   - Single-channel image loading
   - Multi-channel stacking
   - With ROIs/labelmaps
   - Using registration transforms
   - Custom augmentations

5. **Advanced Topics**
   - Caching strategy
   - Custom reference spaces
   - Handling oblique acquisitions
   - Integration with PyTorch Lightning

### API.md

- Full API reference for all public classes/functions
- Parameter descriptions
- Return value specifications
- Examples for each method

---

## ⚡ Performance Considerations

1. **Use DataLoader workers**
   ```python
   loader = DataLoader(
       dataset,
       batch_size=4,
       num_workers=4,      # Process-based parallelism
       persistent_workers=True,
       pin_memory=True
   )
   ```

2. **Caching strategy**
   - Cache resampled arrays (not SimpleITK images)
   - Use memory-mapped arrays for large datasets
   - Implement cache eviction for memory limits

3. **Lazy loading**
   - Don't load all images at `__init__`
   - Load files in `__getitem__` only

4. **Reference image reuse**
   - If using same reference for all subjects, create it once

---

## 🚀 Example Usage Scripts

**`examples/example_basic_usage.py`:**
```python
#!/usr/bin/env python3
"""Basic usage example."""

from pyable_dataloader import PyableDataset
from torch.utils.data import DataLoader

# Create manifest
manifest = {
    "subject_001": {
        "images": ["/data/sub001/T1.nii.gz"],
        "rois": ["/data/sub001/brain_mask.nii.gz"],
    }
}

# Create dataset
dataset = PyableDataset(
    manifest=manifest,
    reference_selector='first',
    transforms=None
)

# Create dataloader
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)

# Iterate
for batch in loader:
    images = batch['images']    # B × C × D × H × W
    rois = batch['rois']        # List of B tensors
    meta = batch['meta']
    
    print(f"Images shape: {images.shape}")
    print(f"Spacing: {meta['spacing'][0]}")
    break
```

**`examples/example_with_transforms.py`:**
Show augmentation pipeline

**`examples/example_registration_workflow.py`:**
Show using registration transforms

---

## ✅ Acceptance Criteria

Your implementation will be accepted if:

1. ✅ **All unit tests pass** (`pytest`)
2. ✅ **Label preservation verified** (no interpolated label values)
3. ✅ **Shapes are correct** (C × D × H × W with Z,Y,X→D,H,W mapping)
4. ✅ **Examples run without errors**
5. ✅ **Documentation is complete** (README + API docs)
6. ✅ **Code is well-commented** (docstrings for all public APIs)
7. ✅ **No external file dependencies in tests** (use synthetic data)
8. ✅ **Integrates with PyTorch DataLoader** (batch collation works)
9. ✅ **Handles edge cases**:
   - Empty ROI lists
   - Single-channel vs multi-channel
   - 2D vs 3D images
   - Different spacing/orientations
   - Missing files (clear error messages)

---

## 🎁 Bonus Features (Optional)

If time permits, consider adding:

1. **Patch sampling** - Extract random 3D patches for memory efficiency
2. **On-the-fly augmentation** - Apply SimpleITK transforms before conversion
3. **Multi-modal alignment** - Auto-register multiple sequences
4. **Preprocessing pipeline** - Built-in skull stripping, bias correction
5. **TorchIO integration** - Wrapper for TorchIO transforms
6. **MONAI compatibility** - Make dataset compatible with MONAI transforms
7. **DDP support** - Proper sharding for distributed training
8. **Progress bars** - Show loading progress with tqdm
9. **Validation mode** - Deterministic transforms for validation set

---

## 📚 Resources

- pyable documentation: `./pyable/docs/`
- pyable examples: `./pyable/example_*.py`
- pyable tests: `./pyable/test_*.py`
- PyTorch Dataset docs: https://pytorch.org/docs/stable/data.html

---

## 🚨 Common Pitfalls to Avoid

1. ❌ **Don't transpose numpy arrays unnecessarily** - pyable v3 already returns (Z,Y,X)
2. ❌ **Don't use linear interpolation for labels** - Use nearest neighbor (pyable does this automatically)
3. ❌ **Don't modify Imaginable objects in-place** - Create copies before resampling
4. ❌ **Don't ignore orientation** - Check `getOrientationCode()` and standardize if needed
5. ❌ **Don't assume same geometry** - Always resample to reference space
6. ❌ **Don't forget metadata** - Preserve spacing/origin/direction in output
7. ❌ **Don't create large intermediates** - Stream data, use generators where possible
8. ❌ **Don't use pickle for caching** - Use numpy (.npy) or HDF5 for arrays

---

## 📝 Deliverable Checklist

Before submitting, ensure:

- [ ] `src/pyable_dataloader/dataset.py` implemented
- [ ] `src/pyable_dataloader/transforms.py` with at least 3 transforms
- [ ] `src/pyable_dataloader/collate.py` for batch handling
- [ ] `src/pyable_dataloader/utils.py` with manifest parsing
- [ ] `tests/test_dataset_basic.py` (3+ tests)
- [ ] `tests/test_dataset_labels.py` (label preservation test)
- [ ] `tests/test_dataset_resample.py` (resampling test)
- [ ] `tests/test_transforms.py` (transform tests)
- [ ] `examples/example_basic_usage.py`
- [ ] `examples/example_with_transforms.py`
- [ ] `examples/data/sample_manifest.json`
- [ ] `README.md` with installation & examples
- [ ] `docs/API.md` with full API reference
- [ ] `pyproject.toml` with dependencies
- [ ] All tests pass: `pytest -v`
- [ ] Example runs: `python examples/example_basic_usage.py`
- [ ] No warnings about deprecated pyable APIs

---

## 🤝 Getting Help

If you need clarification on pyable APIs:

1. Read `./pyable/docs/LLM_DESCRIPTOR.yaml` first
2. Check `./pyable/QUICKREF.md` for common patterns
3. Look at `./pyable/test_*.py` for usage examples
4. Refer to `./pyable/ORIENTATION_GUIDE.md` for coordinate systems
5. Check `./pyable/docs/DEFORMATION_WORKFLOW.md` for transform workflows

**Questions about specific methods?**
- Search for method name in `./pyable/pyable/imaginable.py`
- All methods have detailed docstrings

---

## 🎯 Start Here

1. **Read** `./pyable/docs/LLM_DESCRIPTOR.yaml` (5 min)
2. **Skim** `./pyable/PYABLE_V3_BREAKING_CHANGES.md` (focus on numpy convention)
3. **Run** one of the test files to see pyable in action:
   ```bash
   cd pyable
   python test_v3_numpy_convention.py
   ```
4. **Create** the project structure under `./src/pyable_dataloader/`
5. **Implement** `dataset.py` with the API specified above
6. **Test** as you go (write tests early!)
7. **Document** each component

---

Good luck! Create a robust, production-ready dataloader that the medical imaging community will love. 🚀
