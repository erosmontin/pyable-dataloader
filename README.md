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

## Examples

See the `examples/` directory for complete examples:

- `example_unified_nifti_style.py`: Replicates original UnifiedNiftiDataset workflow
- `example_pytorch_and_pyfe.py`: Integration with pyfe for feature extraction

## Contributing

1. Install development dependencies: `pip install -e .[dev]`
2. Run tests: `pytest`
3. Format code: `black src/ tests/`
4. Check types: `mypy src/`

## License

MIT License
