import os
import torch
from pyable_dataloader.dataset import PyableDataset
from pyable_dataloader.utils import update_manifest_with_reference
from pyable_dataloader.transforms import IntensityPercentile, Compose,RandomRototranslation,LabelMapToRoi
import pyable.imaginable as ima

def test_intensity_percentile_transform_on_realdata():
    # Use the same manifest as in the example
    manifest = {
        "FT1013": {
            "images": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20131021083019_701.nii"],
            "labelmaps": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/ROI/segmentation.nii"],

        },
        "FT1033": {
            "images": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20140203102514_801.nii"],
            "labelmaps": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/ROI/segmentation.nii"],

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

    # Use ToTorch as the only transform
    transforms = Compose([
        LabelMapToRoi(labelmapvalues=[1, 2, 3]),
        RandomRototranslation(angle_range=((-10, 10), (-10, 10), (-10, 10)),translation_range=[( -5, 5), ( -5, 5), ( -5, 5)])
    ])
    
    normalizations = Compose([
        IntensityPercentile(low=0, high=100)
    ])
    dataset = PyableDataset(
        manifest=manifest,
        transforms=transforms,
        normalizations=normalizations,
        debug_save_dir=None,
        debug_save_format=None,
        dtype=None
    )
    

    for i in range(len(dataset)):
        sample = dataset[i]
        # Check images
        for img in sample['images'] if isinstance(sample['images'], (list, tuple)) else [sample['images']]:
            print(f"Image dtype: {img.dtype}, shape: {img.shape}, min: {img.min().item()}, max: {img.max().item()}")
            ima.saveNumpy(img.numpy(), f"debug/aimage_{i}.nii")
        # Check labelmaps
        for img in sample['labelmaps']:
            ima.saveNumpy(img.numpy(), f"debug/alabelmap_{i}.nii")
        # Check rois
        for k,img in enumerate(sample['rois']):
            ima.saveNumpy(img.numpy(), f"debug/aroi_{i}_{k}.nii")


if __name__ == "__main__":
    test_intensity_percentile_transform_on_realdata()
