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

# Iterate
for batch in loader:
    images = batch['images']      # B × C × D × H × W
    labels = batch['label']        # B
    meta = batch['meta']
    
    # Your training code here
    print(f"Batch shape: {images.shape}")
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

## Advanced Usage

### With ROI Masking and Centering

```python
dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    mask_with_roi=True,              # Multiply image by ROI mask
    roi_labels=[1, 34, 35],          # Filter specific label values
    roi_center_target=[0, 0, 50],    # Center ROI at specific coordinates
    stack_channels=True
)
```

### With Data Augmentation

```python
from pyable_dataloader import (
    Compose,
    IntensityNormalization,
    RandomFlip,
    RandomRotation90,
    RandomAffine,
    RandomNoise
)

# Create augmentation pipeline
train_transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomFlip(axes=[1, 2], prob=0.5),
    RandomRotation90(prob=0.3),
    RandomAffine(
        rotation_range=5.0,
        zoom_range=(0.95, 1.05),
        shift_range=2.0,
        prob=0.5
    ),
    RandomNoise(std=0.01, prob=0.3)
])

dataset = PyableDataset(
    manifest='data/train.json',
    target_size=[64, 64, 64],
    transforms=train_transforms,
    cache_dir='cache/train'
)
```

### With Advanced Spatial Transforms

New spatial transforms with **per-axis control** and **automatic label preservation**:

```python
from pyable_dataloader import (
    Compose,
    IntensityNormalization,
    RandomTranslation,
    RandomRotation,
    RandomBSpline,
    RandomFlip
)

# Create advanced augmentation pipeline
train_transforms = Compose([
    IntensityNormalization(method='zscore'),
    
    # Per-axis translation in mm
    RandomTranslation(
        translation_range=[
            [-5, 5],   # X-axis: ±5mm
            [-5, 5],   # Y-axis: ±5mm
            [-3, 3]    # Z-axis: ±3mm
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
    
    # BSpline deformation for realistic warping
    RandomBSpline(
        mesh_size=(4, 4, 4),    # BSpline grid size
        magnitude=3.0,           # Maximum displacement in mm
        prob=0.5
    ),
    
    RandomFlip(axes=[1, 2], prob=0.5)
])

dataset = PyableDataset(
    manifest='data/train.json',
    target_size=[64, 64, 64],
    transforms=train_transforms,
    cache_dir='cache/train'
)

# Labels are automatically preserved using nearest-neighbor interpolation!
# No need to manually configure interpolation for labelmaps/ROIs
```

**Key Features of Spatial Transforms:**
- ✅ **Per-axis control**: Different ranges for each axis
- ✅ **Automatic label preservation**: Uses nearest-neighbor for labelmaps/ROIs
- ✅ **Physical space**: Translation in mm, rotation in degrees
- ✅ **Realistic deformations**: BSpline for smooth, anatomically-plausible warping

### With Global Reference Space

```python
# All subjects resampled to same reference space
dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    reference_space='/data/templates/MNI152_2mm.nii.gz'
)
```

Or create reference from parameters:

```python
reference_params = {
    'size': [64, 64, 64],
    'spacing': [2.0, 2.0, 2.0],
    'origin': [0.0, 0.0, 0.0],
    'direction': (1,0,0, 0,1,0, 0,0,1)
}

dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    reference_space=reference_params
)
```

### Overlay Results Back to Original Space

```python
# During inference
dataset = PyableDataset(
    manifest='test.json',
    target_size=[64, 64, 64],
    return_meta=True
)

# Get sample
sample = dataset[0]
images = sample['images']
subject_id = sample['id']

# Run model
with torch.no_grad():
    prediction = model(images.unsqueeze(0))

# Get overlayer function
overlayer = dataset.get_original_space_overlayer(subject_id)

# Overlay prediction back to original space
prediction_np = prediction[0, 0].cpu().numpy()  # Remove batch and channel dims
original_space_sitk = overlayer(prediction_np, interpolator='linear')

# Save result
import SimpleITK as sitk
sitk.WriteImage(original_space_sitk, f'results/{subject_id}_prediction.nii.gz')
```

## PyFE Integration

**pyable-dataloader** can also be used with **pyfe** (feature extraction / radiomics):

### Using from pyfe

```python
# Import directly from pyfe
from pyfe.dataloader import (
    PyableDataset,
    Compose,
    RandomTranslation,
    RandomRotation,
    RandomBSpline
)

# Use the same API!
dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    transforms=Compose([
        RandomTranslation(translation_range=[[-3, 3], [-3, 3], [-2, 2]], prob=0.5),
        RandomRotation(rotation_range=[[-5, 5], [-5, 5], [-10, 10]], prob=0.5)
    ])
)
```

### Converting Manifest for pyfe

```python
from pyable_dataloader.pyfe_adapter import convert_manifest_to_pyfe

# Convert pyable manifest to pyfe format
convert_manifest_to_pyfe(
    input_manifest='data/manifest.json',
    output_path='data/pyfe_manifest.json'
)

# Now use with pyfe
from pyfe import pyfe

result, ids = pyfe.exrtactMyFeatures(
    'data/pyfe_manifest.json',
    dimension=3
)

# Convert to pandas DataFrame
df = pyfe.exrtactMyFeaturesToPandas(result, ids)
```

**Dual Usage:**
- Same codebase for **PyTorch training** (deep learning)
- And **pyfe feature extraction** (radiomics/handcrafted features)
- Share data preprocessing pipeline across both workflows!

## Examples

See `examples/` directory for complete examples:

- `example_basic_usage.py` - Simple dataset loading
- `example_with_transforms.py` - Data augmentation pipeline
- `example_classification.py` - Full classification workflow
- `example_segmentation.py` - Segmentation with overlay
- `example_pytorch_and_pyfe.py` - Complete dual usage example (PyTorch + pyfe)

## API Reference

### PyableDataset

Main dataset class for loading medical images.

**Parameters:**

- `manifest` (str, dict, or list): Path to manifest file or manifest dict
- `target_size` (list): Target voxel dimensions [D, H, W]
- `target_spacing` (float or list): Target spacing in mm
- `reference_selector` (str, int, or callable): How to select reference image
  - `'first'`: Use first image (default)
  - `'largest'`: Use largest image by volume
  - `int`: Use image at index
  - `callable`: Custom function
- `reference_space` (str or dict, optional): Global reference space
- `roi_center_target` (list, optional): Target coordinates for ROI center
- `mask_with_roi` (bool): Whether to mask image with ROI
- `roi_labels` (list, optional): Label values to keep in ROI
- `transforms` (callable, optional): Augmentation pipeline
- `stack_channels` (bool): Stack multiple images as channels
- `cache_dir` (str, optional): Directory for caching
- `force_reload` (bool): Force reload instead of using cache
- `dtype` (torch.dtype): Output tensor dtype
- `return_meta` (bool): Include metadata in output
- `orientation` (str): Standard orientation code (default 'LPS')

**Returns:**

Dictionary with keys:
- `'id'`: Subject identifier
- `'images'`: torch.Tensor (C × D × H × W)
- `'rois'`: List[torch.Tensor]
- `'labelmaps'`: List[torch.Tensor]
- `'label'`: torch.Tensor (if present in manifest)
- `'meta'`: dict (if return_meta=True)

### Transforms

All transforms follow the interface:

```python
def __call__(images, rois, labelmaps, meta) -> (images, rois, labelmaps)
```

Available transforms:

#### Intensity Transforms
- **`IntensityNormalization`**: Z-score, min-max, or percentile normalization
  ```python
  IntensityNormalization(method='zscore')  # or 'minmax', 'percentile'
  ```
- **`RandomNoise`**: Add Gaussian noise
  ```python
  RandomNoise(std=0.01, prob=0.3)
  ```

#### Spatial Transforms (with automatic label preservation)
- **`RandomTranslation`**: Per-axis translation in mm
  ```python
  RandomTranslation(
      translation_range=[[-5, 5], [-5, 5], [-3, 3]],  # [X, Y, Z] ranges
      prob=0.8
  )
  ```
- **`RandomRotation`**: Per-axis rotation in degrees
  ```python
  RandomRotation(
      rotation_range=[[-10, 10], [-10, 10], [-15, 15]],  # [X, Y, Z] ranges
      prob=0.7
  )
  ```
- **`RandomBSpline`**: BSpline deformation
  ```python
  RandomBSpline(
      mesh_size=(4, 4, 4),      # BSpline grid size
      magnitude=3.0,             # Maximum displacement in mm
      prob=0.5
  )
  ```
- **`RandomAffine`**: Combined rotation, zoom, shift (supports per-axis or scalar ranges)
  ```python
  # Scalar ranges (applied to all axes)
  RandomAffine(rotation_range=10.0, zoom_range=(0.9, 1.1), shift_range=3.0, prob=0.5)
  
  # Per-axis ranges
  RandomAffine(
      rotation_range=[[-5, 5], [-5, 5], [-10, 10]],
      zoom_range=[(0.9, 1.1), (0.9, 1.1), (0.85, 1.15)],
      shift_range=[[-3, 3], [-3, 3], [-5, 5]],
      prob=0.5
  )
  ```
- **`RandomFlip`**: Random axis flipping
  ```python
  RandomFlip(axes=[1, 2], prob=0.5)  # Flip Y and Z axes
  ```
- **`RandomRotation90`**: Random 90° rotations
  ```python
  RandomRotation90(prob=0.3)
  ```

#### Geometric Transforms
- **`CropOrPad`**: Crop or pad to target size
  ```python
  CropOrPad(target_size=[64, 64, 64])
  ```

#### Composition
- **`Compose`**: Chain multiple transforms
  ```python
  Compose([transform1, transform2, transform3])
  ```

**Note:** All spatial transforms automatically preserve labels in ROIs and labelmaps using nearest-neighbor interpolation. No manual configuration needed!

## Important Notes

### NumPy Array Ordering

This package uses **pyable v3** which returns arrays in **(Z, Y, X)** order (standard numpy convention for medical images). This matches PyTorch's **(D, H, W)** format.

### Label Preservation

ROIs and labelmaps are automatically resampled with **nearest-neighbor interpolation** to preserve discrete label values. No manual configuration needed!

### Caching

Preprocessed data is cached to disk for faster loading. Cache keys include:
- Subject ID
- File modification times
- Target size and spacing
- ROI settings

Clear cache by deleting the `cache_dir` or using `force_reload=True`.

## Coordinate Systems

The package handles coordinate systems correctly:

1. **Physical Space**: Real-world coordinates in mm (X, Y, Z)
2. **Array Space**: Numpy array indices (Z, Y, X) = (D, H, W)
3. **ITK Space**: SimpleITK indices (i, j, k) = (X, Y, Z)

pyable handles conversions automatically via:
- `getPhysicalPointFromArrayIndex(k, j, i)`
- `getArrayIndexFromPhysicalPoint(x, y, z)`

## Performance Tips

1. **Use caching** for training to avoid recomputing preprocessing
2. **Use multiple workers** in DataLoader (`num_workers=4`)
3. **Enable pin_memory** for GPU training
4. **Persistent workers** for faster epoch transitions
5. **Pre-compute reference space** if using the same for all subjects

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

## Troubleshooting

### Out of Memory

- Reduce `target_size`
- Reduce `batch_size`
- Use fewer `num_workers`
- Enable caching to avoid duplicate loading

### Slow Loading

- Enable caching with `cache_dir`
- Increase `num_workers`
- Use SSD storage for cache
- Consider pre-computing all preprocessing

### Label Interpolation

If you see non-integer values in ROIs/labelmaps, check:
- Are you using the correct class? (`Roiable` or `LabelMapable`)
- Is the pixel type integer? (uint8, int16, etc.)
- pyable v3 automatically uses nearest-neighbor for integer images

## Citation

If you use this package, please cite:

```bibtex
@software{pyable_dataloader,
  title = {PyAble DataLoader: Medical Image Loading for PyTorch},
  author = {Montin, Eros},
  year = {2025},
  url = {https://github.com/erosmontin/able-dataloader}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.

## Related Projects

- **pyable**: Core medical imaging library
- **PyTorch**: Deep learning framework
- **SimpleITK**: Medical image processing
- **MONAI**: Medical imaging AI framework
- **TorchIO**: PyTorch transforms for medical images
