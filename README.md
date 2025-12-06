# PyAble DataLoader

**PyTorch DataLoader for Medical Images using pyable**

A production-ready PyTorch Dataset for loading and preprocessing medical images (NIfTI, DICOM, etc.) with proper spatial transformations, ROI handling, and support for deep learning workflows.

## Key Features

- ✅ **Proper Spatial Transformations**: Uses `pyable`'s resampling for correct spatial reference
- ✅ **ROI-Aware Processing**: Centers images around ROI for maximum information preservation
- ✅ **Label Preservation**: Automatic nearest-neighbor interpolation for segmentations
- ✅ **Multi-Modal Support**: Stack multiple image sequences as channels
- ✅ **Registration Integration**: Apply transforms from registration directly
- ✅ **Caching**: Fast loading with automatic caching of preprocessed data
- ✅ **Augmentation**: Medical imaging-aware data augmentation
- ✅ **Debug Save**: Save processed images for manual inspection and pipeline verification
- ✅ **Overlay Support**: Easy mapping from model outputs back to original space

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- SimpleITK (installed with pyable)

### 1. Install pyable (required dependency)

```bash
# Clone and install pyable
cd /path/to/pyable
pip install -e .
```

### 2. Install pyable-dataloader

```bash
cd /path/to/able-dataloader
pip install -e .
```

### 3. (Optional) Install development dependencies

```bash
pip install -e .[dev]
```

### Verification

```bash
# Test import
python -c "from pyable_dataloader import PyableDataset; print('✅ Import successful!')"

# Run tests
pytest tests/ -v
```

## Quick Start

### Basic Usage

```python
from pyable_dataloader import PyableDataset
from torch.utils.data import DataLoader

# Create dataset from JSON manifest
dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    stack_channels=True
)

# Create PyTorch DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Training loop
for epoch in range(num_epochs):
    for batch in loader:
        images = batch['images']      # torch.Tensor: B × C × D × H × W
        labels = batch['label']        # torch.Tensor: B (if present)
        meta = batch['meta']          # dict with spacing, origin, etc.
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch shape: {images.shape}, Loss: {loss.item():.4f}")
```

### Debug Save Functionality

For debugging and verifying your preprocessing pipeline, you can save the processed images that are sent to training:

```python
from pyable_dataloader import PyableDataset

# Create dataset with debug save enabled
dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    transforms=train_transforms,
    debug_save_dir='./debug_images',  # Directory to save processed images
    debug_save_format='nifti'          # 'nifti' or 'numpy'
)

# Process first sample - images will be saved to ./debug_images/
sample = dataset[0]

# Files saved will include:
# - subject_id_image_0.nii.gz (first image)
# - subject_id_image_1.nii.gz (second image, if multi-modal)
# - subject_id_roi_0.nii.gz (first ROI)
# - subject_id_labelmap_0.nii.gz (first labelmap)
```

This saves images after all preprocessing (resampling, transforms, ROI masking) but before tensor conversion, allowing you to manually inspect what your model actually receives.

### Complete Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pyable_dataloader import PyableDataset, Compose, IntensityNormalization, RandomFlip

# Setup transforms
train_transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomFlip(axes=[1, 2], prob=0.5)
])

# Create datasets
train_dataset = PyableDataset(
    manifest='data/train.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    transforms=train_transforms,
    cache_dir='./cache/train'
)

val_dataset = PyableDataset(
    manifest='data/val.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    cache_dir='./cache/val'
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=8, 
    shuffle=False, 
    num_workers=4, 
    pin_memory=True
)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YourModel().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    # Training
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        images = batch['images'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
          f"Val Loss = {val_loss/len(val_loader):.4f}")
```

### Manifest Format

#### JSON Format

```json
{
    "subject_001": {
        "images": [
            "/data/sub001/T1w.nii.gz",
            "/data/sub001/T2w.nii.gz"
        ],
        "rois": ["/data/sub001/brain_mask.nii.gz"],
        "labelmaps": ["/data/sub001/segmentation.nii.gz"],
        "reference": 0,
        "label": 1.0
    },
    "subject_002": {
        "images": ["/data/sub002/T1w.nii.gz"],
        "rois": [],
        "labelmaps": [],
        "label": 0.0
    }
}
```

#### CSV Format

```csv
id,image_paths,roi_paths,labelmap_paths,reference,label
sub001,"['/data/sub001/T1.nii.gz','/data/sub001/T2.nii.gz']",/data/sub001/mask.nii.gz,,0,1.0
sub002,/data/sub002/T1.nii.gz,,,first,0.0
```

## Architecture Overview

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          INPUT DATA                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Manifest (JSON/CSV)              Medical Images (NIfTI/DICOM)         │
│  ┌──────────────────┐             ┌──────────────┐                     │
│  │ subject_001:     │             │ T1.nii.gz    │                     │
│  │   images: [...]  │────────────>│ T2.nii.gz    │                     │
│  │   rois: [...]    │             │ FLAIR.nii.gz │                     │
│  │   label: 1.0     │             └──────────────┘                     │
│  └──────────────────┘             ┌──────────────┐                     │
│                                    │ roi.nii.gz   │                     │
│                                    └──────────────┘                     │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PyableDataset.__getitem__(idx)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: Load with pyable                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ img = SITKImaginable(filename)                                   │  │
│  │ roi = Roiable(filename)                                          │  │
│  │ img.resampleOnCanonicalSpace()  # Ensures LPS orientation        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 2: Select Reference Space                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ reference = select_reference(images)                             │  │
│  │   - First image                                                  │  │
│  │   - Largest image                                                │  │
│  │   - Custom function                                              │  │
│  │   - Global template                                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 3: Compute ROI Center (if needed)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ roi_center = compute_roi_center(roi)                             │  │
│  │ # Physical coordinates (X, Y, Z) in mm                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 4: Create Centered Reference                                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ centered_ref = create_centered_reference(                        │  │
│  │     source_image,                                                │  │
│  │     roi_center,                                                  │  │
│  │     target_size,                                                 │  │
│  │     target_spacing                                               │  │
│  │ )                                                                │  │
│  │ # Centers volume around ROI to maximize information              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 5: Resample All Images                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ for img in images:                                               │  │
│  │     img.resampleOnTargetImage(reference)                      │  │
│  │     # Linear interpolation for images                            │  │
│  │                                                                  │  │
│  │ for roi in rois:                                                 │  │
│  │     roi.resampleOnTargetImage(reference)                      │  │
│  │     # Nearest-neighbor for labels (automatic!)                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 6: Convert to NumPy                                              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ arrays = [img.getImageAsNumpy() for img in images]              │  │
│  │ # Returns (Z, Y, X) format - pyable v3 convention!              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 7: Apply ROI Masking (if enabled)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ if mask_with_roi:                                                │  │
│  │     combined_mask = np.any([roi > 0 for roi in rois], axis=0)   │  │
│  │     images = [arr * combined_mask for arr in images]            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 8: Stack Channels (if enabled)                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ if stack_channels and len(images) > 1:                           │  │
│  │     images = np.stack(images, axis=0)  # C × Z × Y × X           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 9: Apply Transforms (if provided)                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ if transforms is not None:                                       │  │
│  │     images, rois, labelmaps = transforms(                        │  │
│  │         images, rois, labelmaps, meta                            │  │
│  │     )                                                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  Step 10: Cache Results                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ save_to_cache(cache_path, images, rois, labelmaps, meta)        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  {                                                                    │
│    'id': 'subject_001',                                              │
│    'images': torch.Tensor(C × D × H × W),                            │
│    'rois': [torch.Tensor(D × H × W), ...],                           │
│    'labelmaps': [torch.Tensor(D × H × W), ...],                      │
│    'meta': {                                                         │
│      'spacing': [2.0, 2.0, 2.0],                                     │
│      'origin': [0.0, 0.0, 0.0],                                      │
│      'direction': [...],                                             │
│      ...                                                             │
│    }                                                                 │
│  }                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Components

- **PyableDataset**: Main dataset class with spatial awareness
- **Transforms**: Composable augmentation pipeline
- **Caching**: Content-based caching for performance
- **Overlay**: Functions to map results back to original space

## Advanced Usage

### Non-PyTorch Usage (Feature Extraction)

For feature extraction or other non-PyTorch applications (like pyfe), use `get_numpy_item()` to get numpy arrays, NIfTI images, or pyable objects instead of PyTorch tensors.

#### Basic Numpy Arrays

```python
from pyable_dataloader import PyableDataset

dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0
)

# Get numpy arrays for feature extraction
sample = dataset.get_numpy_item(0)
images = sample['images']  # List of numpy arrays (Z, Y, X)
rois = sample['rois']      # List of numpy arrays (Z, Y, X)
meta = sample['meta']      # Metadata dict

# Use with pyfe or other libraries
# features = pyfe.extract_features(images, rois, ...)
```

#### NIfTI Images

```python
# Get SimpleITK images
sample = dataset.get_numpy_item(0, as_nifti=True)
images = sample['images']  # List of SimpleITK images
rois = sample['rois']      # List of SimpleITK images

# Save or process as NIfTI
import SimpleITK as sitk
sitk.WriteImage(images[0], 'processed_image.nii.gz')
```

### Pyfe Integration: Requesting Files and Augmented Data

This section explains recommended patterns for pyfe (or any feature-extraction agent) to request data from `pyable-dataloader`.

- Use original file paths when pyfe can operate directly on NIfTI files.
- Use `get_numpy_item(..., as_nifti=True)` and save to temporary NIfTI files when augmentation or preprocessing is required.

1) Get original file paths (no augmentation)

```python
def get_original_files_for_pyfe(dataset, subject_idx):
    """Return original file paths from the dataset manifest."""
    subject_id = dataset.ids[subject_idx]
    item = dataset.data[subject_id]
    return {
        'images': item.get('images', []),
        'rois': item.get('rois', []),
        'labelmaps': item.get('labelmaps', [])
    }

# Example usage
dataset = PyableDataset(manifest='data/manifest.json', target_size=[64,64,64])
paths = get_original_files_for_pyfe(dataset, 0)
# paths['images'] contains the original NIfTI paths you can pass to pyfe
```

2) Get augmented/processed files (safe for pyfe) — recommended when you need the exact preprocessed inputs

```python
import tempfile
from pathlib import Path
import SimpleITK as sitk

def get_augmented_files_for_pyfe(dataset, subject_idx, augmentation_transforms=None):
    """Return paths to temporary NIfTI files containing processed (and optionally augmented) data.

    The files live in a temporary directory; process them immediately or copy them out.
    """
    sample = dataset.get_numpy_item(subject_idx, as_nifti=True, transforms=augmentation_transforms)

    tmpdir = Path(tempfile.mkdtemp(prefix=f"pyfe_{dataset.ids[subject_idx]}_"))
    tmpdir.mkdir(parents=True, exist_ok=True)

    out = {'images': [], 'rois': [], 'labelmaps': [], 'meta': sample.get('meta', {})}

    for i, sitk_img in enumerate(sample['images']):
        p = tmpdir / f"image_{i}.nii.gz"
        sitk.WriteImage(sitk_img, str(p))
        out['images'].append(str(p))

    for i, sitk_roi in enumerate(sample.get('rois', [])):
        p = tmpdir / f"roi_{i}.nii.gz"
        sitk.WriteImage(sitk_roi, str(p))
        out['rois'].append(str(p))

    for i, sitk_lm in enumerate(sample.get('labelmaps', [])):
        p = tmpdir / f"labelmap_{i}.nii.gz"
        sitk.WriteImage(sitk_lm, str(p))
        out['labelmaps'].append(str(p))

    # Return the temp directory paths. Caller is responsible for cleanup.
    return out

# Example usage
# aug_transforms = Compose([...])
# files = get_augmented_files_for_pyfe(dataset, 0, augmentation_transforms=aug_transforms)
# features = pyfe.extract_features_from_files(files['images'], files['rois'])
```

Notes:
- The dataset already caches preprocessed resampled volumes; repeated calls are fast.
- If you need automatic cleanup, adapt `get_augmented_files_for_pyfe` to use `TemporaryDirectory()` as a context manager and perform pyfe extraction inside that context.

#### Pyable Objects

```python
# Get pyable objects for advanced processing
sample = dataset.get_numpy_item(0, as_pyable=True)
images = sample['images']  # List of SITKImaginable objects
rois = sample['rois']      # List of Roiable objects

# Apply pyable operations
for img in images:
    img.resampleOnTargetImage(target_img)
```

#### File Paths for Feature Extraction

If your feature extraction tool requires file paths instead of numpy arrays, use `save_to_files=True` to save the processed (and optionally augmented) data to temporary NIfTI files:

```python
# Get file paths to processed/augmented NIfTI files
sample = dataset.get_numpy_item(0, save_to_files=True, transforms=aug_transforms)
image_paths = sample['images']  # List of file paths to NIfTI files
roi_paths = sample['rois']      # List of file paths to ROI NIfTI files

# Use with pyfe or other tools that need file inputs
# features = pyfe.extract_features_from_files(image_paths, roi_paths, ...)
```

The temporary files are automatically cleaned up when your Python session ends, or you can manually clean them using the `temp_dir` in the metadata.

### Data Augmentation

Data augmentation is applied using composable transform classes. All transforms automatically preserve labels in ROIs and labelmaps using nearest-neighbor interpolation.

#### Basic Data Augmentation Pipeline

```python
from pyable_dataloader import (
    Compose, 
    IntensityNormalization, 
    RandomTranslation, 
    RandomFlip,
    RandomRotation90
)

# Create augmentation pipeline
transforms = Compose([
    # Normalize intensity values
    IntensityNormalization(method='zscore'),
    
    # Apply random spatial transformations
    RandomTranslation(translation_range=[[-3,3], [-3,3], [-3,3]], prob=0.8),
    RandomFlip(axes=[0, 1, 2], prob=0.5),
    RandomRotation90(prob=0.3)
])

# Apply to dataset
dataset = PyableDataset(
    manifest='manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    transforms=transforms,  # Augmentation pipeline
    cache_dir='./cache/train'  # Cache augmented data
)
```

#### Advanced Spatial Augmentation

For more sophisticated augmentation with per-axis control:

```python
from pyable_dataloader import (
    Compose,
    IntensityNormalization,
    RandomTranslation,
    RandomRotation,
    RandomBSpline,
    RandomFlip
)

# Advanced pipeline with per-axis control
advanced_transforms = Compose([
    IntensityNormalization(method='zscore'),
    
    # Per-axis translation in mm (more control in Z direction)
    RandomTranslation(
        translation_range=[
            [-5, 5],   # X-axis: ±5mm
            [-5, 5],   # Y-axis: ±5mm  
            [-3, 3]    # Z-axis: ±3mm (less variation)
        ],
        prob=0.8
    ),
    
    # Per-axis rotation in degrees
    RandomRotation(
        rotation_range=[
            [-10, 10],  # X-axis rotation
            [-10, 10],  # Y-axis rotation
            [-15, 15]   # Z-axis rotation (more variability)
        ],
        prob=0.7
    ),
    
    # Realistic deformation using BSpline
    RandomBSpline(
        mesh_size=(4, 4, 4),    # BSpline grid size
        magnitude=3.0,           # Maximum displacement in mm
        prob=0.5
    ),
    
    RandomFlip(axes=[1, 2], prob=0.5)  # Flip Y and Z axes
])

dataset = PyableDataset(
    manifest='data/train.json',
    target_size=[64, 64, 64],
    transforms=advanced_transforms
)
```

#### Training vs Validation Augmentation

```python
# Training transforms (with augmentation)
train_transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomTranslation(translation_range=[[-3,3], [-3,3], [-3,3]], prob=0.8),
    RandomFlip(axes=[0, 1, 2], prob=0.5)
])

# Validation transforms (no augmentation, just normalization)
val_transforms = Compose([
    IntensityNormalization(method='zscore')
])

train_dataset = PyableDataset(
    manifest='train.json',
    target_size=[64, 64, 64],
    transforms=train_transforms
)

val_dataset = PyableDataset(
    manifest='val.json', 
    target_size=[64, 64, 64],
    transforms=val_transforms
)
```

### ROI-Centered Processing

```python
dataset = PyableDataset(
    manifest='manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    roi_center_target=[0, 0, 32],  # Center ROI at this physical coordinate
    mask_with_roi=True,            # Mask images with ROI
    roi_labels=[1, 2, 3]           # Only use these label values
)
```

### Multi-Modal Stacking

```python
# Stack T1, T2, FLAIR as channels
dataset = PyableDataset(
    manifest='manifest.json',
    target_size=[64, 64, 64],
    stack_channels=True  # Output: C × D × H × W where C = num_images
)
```

### Caching for Performance

```python
dataset = PyableDataset(
    manifest='manifest.json',
    target_size=[64, 64, 64],
    cache_dir='./cache',     # Cache directory
    force_reload=False       # Use cached data when available
)
```

### Overlay Results Back to Original Space

```python
# Get overlayer function
overlayer = dataset.get_original_space_overlayer(subject_id)

# Overlay prediction back to original space
original_space_sitk = overlayer(prediction_array, interpolator='linear')

# Save result
import SimpleITK as sitk
sitk.WriteImage(original_space_sitk, f'results/{subject_id}_prediction.nii.gz')
```

## API Reference

### PyableDataset

```python
PyableDataset(
    manifest: Union[str, dict, List[str]],
    target_size: List[int],
    target_spacing: Union[float, List[float]] = 1.0,
    reference_selector: Union[str, int, Callable] = 'first',
    reference_space: Optional[Union[str, dict]] = None,
    roi_center_target: Optional[List[float]] = None,
    mask_with_roi: bool = False,
    roi_labels: Optional[List[int]] = None,
    transforms: Optional[Callable] = None,
    stack_channels: bool = True,
    cache_dir: Optional[str] = None,
    force_reload: bool = False,
    dtype: torch.dtype = torch.float32,
    return_meta: bool = True,
    orientation: str = 'LPS'
)
```

**Parameters:**
- `manifest`: Path to JSON/CSV file(s) or dict with subject data
- `target_size`: Target voxel dimensions [D, H, W]
- `target_spacing`: Target spacing in mm (float or [x,y,z])
- `reference_selector`: How to choose reference image ('first', 'largest', int, callable)
- `reference_space`: Global reference space (path to image or dict)
- `roi_center_target`: Target physical coordinates for ROI center
- `mask_with_roi`: Whether to multiply images by ROI mask
- `roi_labels`: List of label values to keep in ROI
- `transforms`: Optional transform pipeline
- `stack_channels`: Stack multiple images as channels
- `cache_dir`: Directory for caching preprocessed data
- `force_reload`: Force reload from disk instead of cache
- `dtype`: PyTorch dtype for output tensors
- `return_meta`: Include metadata in output
- `orientation`: Standard orientation code

**Returns:**
Dictionary with:
- `'id'`: Subject identifier
- `'images'`: torch.Tensor (C × D × H × W or D × H × W)
- `'rois'`: List[torch.Tensor] (each D × H × W)
- `'labelmaps'`: List[torch.Tensor] (each D × H × W)
- `'meta'`: dict with spacing, origin, direction, etc. (if return_meta=True)

### Transforms

All transforms follow the interface:

```python
def __call__(images, rois, labelmaps, meta) -> (images, rois, labelmaps)
```

#### Intensity Transforms

**IntensityNormalization**: Standardize image intensities
```python
IntensityNormalization(
    method: str = 'zscore',  # 'zscore', 'minmax', 'percentile'
    per_channel: bool = True,
    clip_percentile: Tuple[float, float] = (1, 99)
)
```

#### Spatial Transforms (with automatic label preservation)

**RandomTranslation**: Translate images in physical space
```python
RandomTranslation(
    translation_range: List[Tuple[float, float]] = [[-5,5], [-5,5], [-5,5]],  # [X, Y, Z] ranges in mm
    prob: float = 0.5
)
```

**RandomRotation**: Rotate images around axes
```python
RandomRotation(
    rotation_range: List[Tuple[float, float]] = [[-10,10], [-10,10], [-10,10]],  # [X, Y, Z] ranges in degrees
    prob: float = 0.5
)
```

**RandomBSpline**: Apply realistic elastic deformation
```python
RandomBSpline(
    mesh_size: Tuple[int, int, int] = (4, 4, 4),  # BSpline grid size
    magnitude: float = 3.0,                        # Maximum displacement in mm
    prob: float = 0.5
)
```

**RandomFlip**: Randomly flip along axes
```python
RandomFlip(
    axes: List[int] = [0, 1, 2],  # 0=Z/D, 1=Y/H, 2=X/W
    prob: float = 0.5
)
```

**RandomRotation90**: Random 90° rotations
```python
RandomRotation90(prob: float = 0.5)
```

#### Composition

**Compose**: Chain multiple transforms together
```python
Compose(transforms: List[MedicalImageTransform])
```

#### Example: Complete Augmentation Pipeline

```python
from pyable_dataloader import *

# Complete augmentation pipeline
transforms = Compose([
    # Intensity normalization
    IntensityNormalization(method='zscore', per_channel=True),
    
    # Spatial augmentations
    RandomTranslation(translation_range=[[-3, 3], [-3, 3], [-2, 2]], prob=0.8),
    RandomRotation(rotation_range=[[-5, 5], [-5, 5], [-10, 10]], prob=0.6),
    RandomFlip(axes=[1, 2], prob=0.5),  # Don't flip Z-axis for brain data
    
    # Elastic deformation
    RandomBSpline(mesh_size=(3, 3, 3), magnitude=2.0, prob=0.3)
])

dataset = PyableDataset(
    manifest='train.json',
    target_size=[64, 64, 64],
    transforms=transforms
)
```

## Migration from Original Code

If you're migrating from the original `UnifiedNiftiDataset`:

### Key Changes

1. **Manifest Format**: Use JSON instead of separate CSV files
2. **ROI Labels**: Specify `roi_labels` explicitly instead of morphology operations
3. **Transforms**: Use composable transform classes instead of hardcoded functions
4. **Caching**: Automatic content-based caching (no manual cache keys)
5. **Label Preservation**: Automatic with `Roiable`/`LabelMapable`

### Before (Original)

```python
dataset = UnifiedNiftiDataset(
    csv_file='train2.csv',
    target_size=[50, 50, 50],
    csv_roi_file='trainROI2.csv',
    target_spacing=2.0,
    mask_with_roi=True,
    roi_morphology='dilate',
    roi_morphology_radius=3,
    femur_z_target=None,
    augmentation=False,
    force_reload=False,
    cache_dir='tmp',
    return_transform=False
)
```

### After (New)

```python
from pyable_dataloader import PyableDataset, Compose, RandomFlip, IntensityNormalization

transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomFlip(axes=[1, 2], prob=0.5)
])

dataset = PyableDataset(
    manifest='manifest.json',  # Converted from your CSV
    target_size=[50, 50, 50],
    target_spacing=2.0,
    mask_with_roi=True,
    roi_labels=[1, 34, 35],  # Explicit labels
    roi_center_target=[0, 0, 50],  # Replace femur_z_target
    transforms=transforms,  # Composable transforms
    cache_dir='cache'
)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

## Agent & Human Guide

This short guide unifies the most common interactions for both human developers and automated agents (e.g. pyfe callers, orchestration agents) so you can consistently request data for training or feature extraction.

- **PyTorch training**: Use `__getitem__` (via `DataLoader`) — the dataset runs the full preprocessing pipeline (resampling, ROI centering/masking, stacking, and the `transforms` pipeline provided at dataset construction). Augmentations defined in `PyableDataset(..., transforms=...)` are applied automatically during `__getitem__` calls used by training loops.

- **Feature extraction / pyfe (non-PyTorch)**: Use `get_numpy_item()` to obtain processed outputs in several formats: numpy arrays (`as_nifti=False, as_pyable=False`), SimpleITK images (`as_nifti=True`), or pyable Imaginable objects (`as_pyable=True`). If you need augmentation for feature extraction, pass an on-demand transforms pipeline via the `transforms=` argument to `get_numpy_item()`.

- **Request patterns for agents**:
    - Original files (no preprocessing): read file paths from the manifest via `dataset.data[subject_id]['images']` and pass them directly to file-oriented tools.
    - Preprocessed but not augmented: `dataset.get_numpy_item(idx)` returns the exact inputs the model would receive (after preprocessing and dataset-level transforms). Use `as_nifti=True` to get SimpleITK objects you can write to disk and hand to file-based tools.
    - Preprocessed + additional on-demand augmentation: `dataset.get_numpy_item(idx, as_nifti=True, transforms=agent_transforms)` — saves or returns the augmented images. Use `save_to_files=True` when you want the dataset to create temporary NIfTI files and return file paths.

- **Debugging & inspection**: Enable `debug_save_dir` on `PyableDataset` to persist pre-tensor images (NIfTI or numpy) after all preprocessing but before conversion to torch tensors. This is the most direct way to inspect exactly what the model receives.

- **Reproducibility**:
    - Seed all involved RNGs: `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, and, if using CUDA, `torch.cuda.manual_seed_all(seed)`.
    - For transformation determinism, ensure any stochastic transform supports receiving a seed or deterministic flag; when composing transforms for agents, create them with fixed random states or run the augmentation + extraction inside a seeded context.

- **Caching & performance**: The dataset caches resampled/preprocessed volumes. Repeated calls to `get_numpy_item()` for the same subject are fast. If you request different on-demand transforms, the dataset will re-run the transform stage but still benefit from cached resampled bases.

- **Temporary files & cleanup**: Helper examples in this README show how to write augmented `SimpleITK` images to a temporary directory. Prefer using `TemporaryDirectory()` as a context manager in agent code so temp files are removed automatically after extraction.

- **Copy-ready agent snippet** (request augmented NIfTI files and run pyfe inside a temp dir):

```python
import tempfile
from pathlib import Path
import SimpleITK as sitk

def agent_extract_with_aug(dataset, subject_idx, aug_transforms, extract_fn):
        sample = dataset.get_numpy_item(subject_idx, as_nifti=True, transforms=aug_transforms)

        with tempfile.TemporaryDirectory(prefix=f"pyfe_{dataset.ids[subject_idx]}_") as td:
                out = {'images': [], 'rois': [], 'labelmaps': []}
                td = Path(td)
                for i, sitk_img in enumerate(sample['images']):
                        p = td / f"image_{i}.nii.gz"
                        sitk.WriteImage(sitk_img, str(p))
                        out['images'].append(str(p))

                for i, sitk_roi in enumerate(sample.get('rois', [])):
                        p = td / f"roi_{i}.nii.gz"
                        sitk.WriteImage(sitk_roi, str(p))
                        out['rois'].append(str(p))

                # Call feature extraction while files exist
                features = extract_fn(out['images'], out['rois'])
        return features
```

- **When to use `as_pyable=True`**: If your downstream code expects pyable Imaginable objects (for advanced spatial operations or to reuse pyable's resampling utilities), request pyable objects and operate in-memory without writing files.

- **Agent best practices**:
    - Prefer `get_numpy_item(..., as_nifti=True)` + temporary files for black-box agents that only accept file paths.
    - Prefer `get_numpy_item(..., as_pyable=True)` for integrated, in-process workflows that can reuse pyable's spatial operations and keep everything in memory.
    - Use `debug_save_dir` during development to make visual verification easy for both humans and automated checks.

If you'd like, I can also add a small helper function to the repository that wraps `get_numpy_item(..., as_nifti=True)` and returns a context-managed temporary directory of files (i.e. the code above as a shipped utility).

## Examples

See the `examples/` directory for complete examples:

- `example_simple.py`: Basic usage with synthetic data demonstrating transforms and DataLoader integration
- `example_unified_nifti_style.py`: Replicates original UnifiedNiftiDataset workflow (requires CSV files)

## Contributing

1. Install development dependencies: `pip install -e .[dev]`
2. Run tests: `pytest`
3. Format code: `black src/ tests/`
4. Check types: `mypy src/`

## License

MIT License

## Agent Handoff (Full)

The following content is the full agent handoff prompt provided by the project owner. It includes context, tasks, tests, and implementation guidance intended for developers or automation agents taking over continued development of this repository.

---

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

---

End of agent handoff content.
