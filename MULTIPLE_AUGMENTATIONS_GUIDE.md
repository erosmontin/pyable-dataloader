# Multiple Augmentations Support in PyableDataset

## Overview

The `PyableDataset` now supports generating multiple augmented versions of a single sample using different transformation configurations. This feature is essential for robust radiomics analysis, allowing you to assess feature stability across various augmentation strategies.

## New Method: `get_multiple_augmentations()`

```python
def get_multiple_augmentations(
    self,
    subject_idx: int,
    augmentation_configs: List[Dict[str, Any]],
    as_nifti: bool = False,
    save_to_files: bool = False,
    base_seed: int = 42
) -> List[Dict[str, Any]]:
```

### Parameters

- `subject_idx`: Index of the subject in the dataset
- `augmentation_configs`: List of dictionaries, each containing:
  - `'name'`: String identifier for the augmentation type
  - `'transforms'`: Compose object with the augmentation pipeline
  - `'params'`: Optional dictionary of parameters for logging
- `as_nifti`: Return SimpleITK images instead of numpy arrays
- `save_to_files`: Save to temporary NIfTI files and return file paths
- `base_seed`: Base random seed for reproducibility

### Returns

List of dictionaries, one per augmentation config, each containing:
- `'name'`: Augmentation identifier
- `'images'`: List of arrays/images/paths
- `'rois'`: List of arrays/images/paths
- `'labelmaps'`: List of arrays/images/paths
- `'meta'`: Metadata dictionary
- `'config'`: Original configuration dictionary

## Usage Examples

### Basic Usage

```python
from pyable_dataloader import PyableDataset, Compose, RandomRotation, RandomTranslation, IntensityNormalization

# Create dataset
dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    target_spacing=2.0
)

# Define augmentation configurations
augmentation_configs = [
    {
        'name': 'light_rotation',
        'transforms': Compose([
            IntensityNormalization(method='zscore'),
            RandomRotation(rotation_range=[[-5, 5], [-5, 5], [-5, 5]], prob=1.0)
        ]),
        'params': {'rotation_range': '±5°'}
    },
    {
        'name': 'heavy_translation',
        'transforms': Compose([
            IntensityNormalization(method='zscore'),
            RandomTranslation(translation_range=[[-10, 10], [-10, 10], [-5, 5]], prob=1.0)
        ]),
        'params': {'translation_range': '±10mm X/Y, ±5mm Z'}
    }
]

# Generate multiple augmentations for subject 0
augmented_samples = dataset.get_multiple_augmentations(
    subject_idx=0,
    augmentation_configs=augmentation_configs,
    base_seed=42
)

# Process each augmented sample
for sample in augmented_samples:
    print(f"Augmentation: {sample['name']}")
    print(f"Image shape: {sample['images'][0].shape}")
    print(f"ROI count: {len(sample['rois'])}")
```

### Integration with Pyfe Feature Extraction

```python
import pyfe

# Get augmentations with file paths for pyfe
augmented_samples = dataset.get_multiple_augmentations(
    subject_idx=0,
    augmentation_configs=augmentation_configs,
    as_nifti=True,
    save_to_files=True,
    base_seed=42
)

# Extract features for each augmentation
all_features = []
for sample in augmented_samples:
    features = pyfe.extract_features_from_files(
        sample['images'],
        sample['rois']
    )
    features['augmentation_type'] = sample['name']
    all_features.append(features)

# Analyze feature stability
import pandas as pd
df = pd.DataFrame(all_features)
print(df.groupby('augmentation_type').std())  # Feature variability across augmentations
```

### Advanced Example with Multiple Augmentation Types

```python
# Comprehensive augmentation strategies
comprehensive_configs = [
    {
        'name': 'baseline',
        'transforms': Compose([IntensityNormalization(method='zscore')]),
        'params': {'type': 'no_augmentation'}
    },
    {
        'name': 'rotation_only',
        'transforms': Compose([
            IntensityNormalization(method='zscore'),
            RandomRotation(rotation_range=[[-15, 15], [-15, 15], [-10, 10]], prob=1.0)
        ]),
        'params': {'rotation': '±15° X/Y, ±10° Z'}
    },
    {
        'name': 'translation_only',
        'transforms': Compose([
            IntensityNormalization(method='zscore'),
            RandomTranslation(translation_range=[[-8, 8], [-8, 8], [-4, 4]], prob=1.0)
        ]),
        'params': {'translation': '±8mm X/Y, ±4mm Z'}
    },
    {
        'name': 'rotation_and_translation',
        'transforms': Compose([
            IntensityNormalization(method='zscore'),
            RandomRotation(rotation_range=[[-10, 10], [-10, 10], [-5, 5]], prob=1.0),
            RandomTranslation(translation_range=[[-5, 5], [-5, 5], [-2, 2]], prob=1.0)
        ]),
        'params': {'rotation': '±10°', 'translation': '±5mm X/Y, ±2mm Z'}
    },
    {
        'name': 'full_augmentation',
        'transforms': Compose([
            IntensityNormalization(method='zscore'),
            RandomRotation(rotation_range=[[-10, 10], [-10, 10], [-5, 5]], prob=1.0),
            RandomTranslation(translation_range=[[-5, 5], [-5, 5], [-2, 2]], prob=1.0),
            RandomFlip(axes=[1, 2], prob=0.5)
        ]),
        'params': {'rotation': '±10°', 'translation': '±5mm', 'flip': 'Y/Z axes'}
    }
]

# Generate all augmentations
all_samples = dataset.get_multiple_augmentations(
    subject_idx=5,
    augmentation_configs=comprehensive_configs,
    save_to_files=True,
    base_seed=123
)

print(f"Generated {len(all_samples)} augmented versions")
for sample in all_samples:
    print(f"- {sample['name']}: {len(sample['images'])} images, {len(sample['rois'])} ROIs")
```

## Reproducibility and Seeding

The method ensures reproducible results by using `base_seed + config_index` for each augmentation configuration:

```python
# Same seed + same config = same results
results1 = dataset.get_multiple_augmentations(0, configs, base_seed=42)
results2 = dataset.get_multiple_augmentations(0, configs, base_seed=42)
# results1 == results2 (for same configs)
```

## Output Formats

### Numpy Arrays (Default)

```python
samples = dataset.get_multiple_augmentations(0, configs)
# sample['images'] -> list of numpy arrays
# sample['rois'] -> list of numpy arrays
```

### SimpleITK Images

```python
samples = dataset.get_multiple_augmentations(0, configs, as_nifti=True)
# sample['images'] -> list of SimpleITK images
# sample['rois'] -> list of SimpleITK images
```

### File Paths

```python
samples = dataset.get_multiple_augmentations(0, configs, save_to_files=True)
# sample['images'] -> list of file paths
# sample['rois'] -> list of file paths
# Temporary files are created and paths returned
```

## Testing

The feature includes comprehensive tests. Run them with:

```bash
# Test the new method specifically
python -m pytest tests/test_basic.py::test_get_multiple_augmentations -v

# Run all tests
python -m pytest tests/ -v
```

## Performance Considerations

- Base preprocessing is cached to avoid redundant resampling
- Each augmentation applies transforms independently
- File cleanup is handled automatically when `save_to_files=True`
- Memory usage scales with number of augmentations requested

## Integration with Existing Workflows

The method is fully compatible with existing `PyableDataset` functionality:

```python
# Works with all existing dataset configurations
dataset = PyableDataset(
    manifest='data.json',
    target_size=[64, 64, 64],
    roi_center_target=[0, 0, 32],
    mask_with_roi=True,
    cache_dir='./cache'
)

# Can be used alongside regular __getitem__ calls
regular_sample = dataset[0]
augmented_samples = dataset.get_multiple_augmentations(0, configs)
```

## Error Handling

The method validates inputs and provides clear error messages:

- Invalid subject indices
- Missing required config keys ('name', 'transforms')
- Transform application failures
- File I/O errors when saving

## Best Practices

1. **Use descriptive names** for augmentation configs
2. **Set appropriate seeds** for reproducible experiments
3. **Combine with caching** for better performance
4. **Use `save_to_files=True`** when integrating with external tools like pyfe
## PyTorch DataLoader Integration: `MultipleAugmentationDataset`

For training PyTorch models with multiple augmentations per sample, use the `MultipleAugmentationDataset` wrapper class. This class expands each subject into multiple augmented samples, allowing standard PyTorch `DataLoader` batching.

### Class Overview

```python
class MultipleAugmentationDataset(Dataset):
    def __init__(
        self,
        base_dataset: PyableDataset,
        augmentation_configs: List[Dict[str, Any]],
        base_seed: int = 42,
        cache_augmentations: bool = True
    ):
```

### Parameters

- `base_dataset`: PyableDataset instance to wrap
- `augmentation_configs`: List of augmentation configurations (same format as `get_multiple_augmentations`)
- `base_seed`: Base random seed for reproducibility
- `cache_augmentations`: Pre-compute all augmentations (faster but uses more memory)

### Key Features

- **Automatic Expansion**: Converts N subjects × M augmentations into N×M total samples
- **PyTorch Compatible**: Returns tensor-formatted samples compatible with DataLoader batching
- **Memory Efficient**: Optional caching balances speed vs memory usage
- **Reproducible**: Deterministic seeding ensures consistent augmentations

### PyTorch DataLoader Usage

```python
from torch.utils.data import DataLoader
from pyable_dataloader import PyableDataset, MultipleAugmentationDataset, Compose, RandomRotation, RandomTranslation

# Create base dataset
base_dataset = PyableDataset(
    manifest='data/manifest.json',
    target_size=[64, 64, 64],
    target_spacing=1.0,
    return_meta=True
)

# Define augmentation configurations
augmentation_configs = [
    {
        'name': 'original',
        'transforms': None
    },
    {
        'name': 'rotated',
        'transforms': Compose([
            RandomRotation(rotation_range=[[-10, 10], [-10, 10], [-10, 10]], prob=1.0)
        ])
    },
    {
        'name': 'translated',
        'transforms': Compose([
            RandomTranslation(translation_range=[[-5, 5], [-5, 5], [-5, 5]], prob=1.0)
        ])
    }
]

# Create augmented dataset
# 100 subjects × 3 augmentations = 300 total samples
aug_dataset = MultipleAugmentationDataset(
    base_dataset=base_dataset,
    augmentation_configs=augmentation_configs,
    base_seed=42,
    cache_augmentations=True  # Pre-compute for faster training
)

# Use with PyTorch DataLoader
train_loader = DataLoader(
    aug_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Training loop
for batch in train_loader:
    images = batch['images']        # Shape: [8, C, D, H, W]
    rois = batch['rois']           # List of ROI tensors
    labels = batch['label']        # Classification labels (if available)
    aug_names = batch['augmentation_name']  # Which augmentation was applied
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Sample Structure

Each sample returned by `MultipleAugmentationDataset` contains:

```python
{
    'id': 'subject_001_rotated',           # Unique identifier
    'images': tensor([C, D, H, W]),       # Stacked image channels
    'rois': [tensor([D, H, W]), ...],     # ROI masks
    'labelmaps': [tensor([D, H, W]), ...], # Label maps
    'meta': {...},                        # Metadata dictionary
    'augmentation_name': 'rotated'        # Augmentation type
}
```

### Memory vs Speed Trade-offs

- **cache_augmentations=True**: Pre-computes all augmented samples
  - **Pros**: Fast iteration during training, reproducible across epochs
  - **Cons**: Higher memory usage, longer initialization time
  
- **cache_augmentations=False**: Computes augmentations on-demand
  - **Pros**: Lower memory usage, faster initialization
  - **Cons**: Slower iteration, potential I/O bottlenecks with many workers

### Best Practices

1. **Use caching for small-to-medium datasets** (< 10,000 samples)
2. **Set appropriate batch sizes** considering the expansion factor
3. **Balance augmentation complexity** with training time
4. **Monitor memory usage** when caching large datasets
5. **Use deterministic seeding** for reproducible experiments

### Advanced Usage: Custom Collate Function

For more complex batching requirements, you can provide a custom collate function:

```python
from torch.utils.data import DataLoader
import torch

def custom_collate(batch):
    """Custom collate function for advanced batching."""
    # Separate different data types
    images = torch.stack([item['images'] for item in batch])
    rois = [item['rois'] for item in batch]  # Keep as list for variable-length
    labels = torch.tensor([item['label'] for item in batch])
    aug_names = [item['augmentation_name'] for item in batch]
    
    return {
        'images': images,
        'rois': rois,
        'labels': labels,
        'augmentation_names': aug_names
    }

loader = DataLoader(
    aug_dataset,
    batch_size=8,
    collate_fn=custom_collate
)
```

## Comparison: `get_multiple_augmentations` vs `MultipleAugmentationDataset`

| Feature | `get_multiple_augmentations` | `MultipleAugmentationDataset` |
|---------|-----------------------------|-------------------------------|
| Use Case | Feature extraction, analysis | Model training, batching |
| Output Format | List of dicts with numpy arrays | PyTorch Dataset with tensors |
| PyTorch Integration | Manual conversion required | Direct DataLoader compatibility |
| Memory Usage | Low (on-demand) | Variable (caching option) |
| Performance | Good for small batches | Optimized for large-scale training |
| Reproducibility | Manual seed management | Automatic deterministic seeding |

Choose `get_multiple_augmentations` for radiomics analysis and `MultipleAugmentationDataset` for deep learning training workflows.

```python
import pyfe
from pyable_dataloader import PyableDataset, Compose, RandomRotation, RandomTranslation

# Setup dataset
dataset = PyableDataset('manifest.json', target_size=[64, 64, 64])

# Define augmentations for robustness testing
robustness_configs = [
    {'name': 'original', 'transforms': None},
    {'name': 'rot_5', 'transforms': Compose([RandomRotation([[-5,5]]*3, prob=1.0)])},
    {'name': 'rot_10', 'transforms': Compose([RandomRotation([[-10,10]]*3, prob=1.0)])},
    {'name': 'trans_2', 'transforms': Compose([RandomTranslation([[-2,2]]*3, prob=1.0)])},
    {'name': 'trans_5', 'transforms': Compose([RandomTranslation([[-5,5]]*3, prob=1.0)])}
]

# Process all subjects
results = []
for subj_idx in range(len(dataset)):
    samples = dataset.get_multiple_augmentations(subj_idx, robustness_configs, save_to_files=True)
    
    for sample in samples:
        features = pyfe.extract_features_from_files(sample['images'], sample['rois'])
        features.update({
            'subject_id': sample['meta']['subject_id'],
            'augmentation': sample['name']
        })
        results.append(features)

# Save results
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('radiomics_robustness_analysis.csv', index=False)

# Analyze stability
stability_stats = df.groupby(['subject_id', 'augmentation']).std()
print("Feature stability across augmentations:")
print(stability_stats.mean().sort_values())
```

This feature enables comprehensive robustness analysis for radiomics studies, ensuring that extracted features are stable across different augmentation strategies.</content>
<parameter name="filePath">/home/erosm/packages/pyable-dataloader/MULTIPLE_AUGMENTATIONS_GUIDE.md