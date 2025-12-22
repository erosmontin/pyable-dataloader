# Test: PyableDataset with DataLoader using same manifest and transforms as test1
import os
import torch
from torch.utils.data import DataLoader
from pyable_dataloader.dataset import PyableDataset
from pyable_dataloader.utils import update_manifest_with_reference
from pyable_dataloader.transforms import IntensityPercentile, Compose, RandomRototranslation, LabelMapToRoi,FlipDimensions

def test_pyable_dataset_with_dataloader():
	manifest = {
		"FT1013": {
			"images": ["data/FT1013/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20131021083019_701.nii"],
			"labelmaps": ["data/FT1013/BL/ROI/segmentation.nii"],
		},
		"FT1033": {
			"images": ["data/FT1033/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20140203102514_801.nii"],
			"labelmaps": ["data/FT1033/BL/ROI/segmentation.nii"],
		}
	}
	manifest = update_manifest_with_reference(
		manifest,
		orientation='LPS',
		resolution=[4.0, 4.0, 4.0],
		reference_idx=0,
		target_size=[20, 40, 40],
		target_output_dir='debug_outputs/simple_dataset_test'
	)

	transforms = Compose([
		LabelMapToRoi(labelmapvalues=[1, 2, 3]),
		RandomRototranslation(angle_range=((-10, 10), (-10, 10), (-10, 10)), translation_range=[(-5, 5), (-5, 5), (-5, 5)]),
		FlipDimensions(dimensions=(1))
	])
	normalizations = Compose([
		IntensityPercentile(low=0, high=100)
	])
	dataset = PyableDataset(
		manifest=manifest,
		transforms=transforms,
		normalizations=normalizations,
		debug_save_dir='debug_outputs/',
	)
	
	DataLoader_dataset = DataLoader(dataset, batch_size=2, shuffle=False)
	for i, sample in enumerate(DataLoader_dataset):
		print(f"Batch {i}:")
		# Check images
		for img in sample['images']:
			print(f"  Image dtype: {img.dtype}, shape: {img.shape}, min: {img.min().item()}, max: {img.max().item()}")
		# Check labelmaps
		for img in sample['labelmaps']:
			print(f"  Labelmap dtype: {img.dtype}, shape: {img.shape}, min: {img.min().item()}, max: {img.max().item()}")
		# Check rois	
		for k, img in enumerate(sample['rois']):
			print(f"  ROI {k} dtype: {img.dtype}, shape: {img.shape}, min: {img.min().item()}, max: {img.max().item()}")

	
	

if __name__ == "__main__":
	test_pyable_dataset_with_dataloader()

