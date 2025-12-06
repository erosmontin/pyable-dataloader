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
5. **Validate feature stability** by comparing extracted features across augmentations

## Example Radiomics Workflow

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