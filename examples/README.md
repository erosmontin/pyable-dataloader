# PyTorch Training Examples

This directory contains examples showing how to use the `pyable-dataloader` package with PyTorch models for various medical imaging tasks.

## Examples Overview

### 1. `example_segmentation_training.py`
**Basic Segmentation Training**
- Simple 3D CNN for medical image segmentation
- Uses PyableDataset to load images and ROIs
- Trains model to predict ROI masks from input images
- Demonstrates basic dataset setup and training loop

### 2. `example_segmentation_with_augmentation.py`
**Segmentation with Data Augmentation**
- Same model as Example 1
- Adds data augmentation during training
- Shows how to apply transforms (rotation, translation) to improve model generalization
- Demonstrates the `transforms` parameter in PyableDataset

### 3. `example_segmentation_multiple_augmentations.py`
**Segmentation with Multiple Augmentations**
- Uses the new `get_multiple_augmentations()` method
- Generates multiple augmented versions of each sample
- Creates larger training batches by stacking augmentations
- Shows how to create custom Dataset class for augmented training

### 4. `example_lenet_classification.py`
**Classification with LeNet**
- 3D LeNet classifier for medical image classification
- Manifest includes classification labels
- Demonstrates train/validation split
- Shows evaluation and per-class accuracy metrics

## Running the Examples

Each example can be run independently:

```bash
cd examples

# Example 1: Basic segmentation
python example_segmentation_training.py

# Example 2: Segmentation with augmentation
python example_segmentation_with_augmentation.py

# Example 3: Multiple augmentations
python example_segmentation_multiple_augmentations.py

# Example 4: Classification
python example_lenet_classification.py
```

## Key Concepts Demonstrated

### Dataset Configuration
- Setting up manifests with images, ROIs, and labels
- Configuring target size and spacing
- Using different output formats (tensors, numpy arrays)

### Data Augmentation
- Single augmentation pipeline per dataset
- Multiple augmentation strategies per sample
- Reproducible augmentation with seeds

### Model Integration
- Segmentation models (CNN with encoder-decoder)
- Classification models (LeNet-style)
- Loss functions (Dice loss, CrossEntropy)
- Training loops with validation

### Performance Considerations
- Batch size selection
- Memory management
- Training metrics and model saving

## Output Files

Each example saves trained models:
- `segmentation_model.pth` - Basic segmentation model
- `segmentation_model_augmented.pth` - Model trained with augmentation
- `segmentation_model_multi_aug.pth` - Model trained with multiple augmentations
- `lenet_best.pth` / `lenet_final.pth` - Classification models

## Requirements

- PyTorch
- pyable-dataloader (installed in editable mode)
- NumPy, SimpleITK (included with pyable)

## Notes

- All examples use synthetic data for demonstration
- Models are intentionally simple for clarity
- Training is kept short (5-10 epochs) for quick testing
- Real applications should use proper validation and longer training