import tempfile


import os
from pyable_dataloader.dataset import PyableDataset
from pyable_dataloader.utils import update_manifest_with_reference
from pyable import imaginable as ima

def main():
    # Example manifest for two subjects
    manifest = {
        "FT1013": {
            "images": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20131021083019_701.nii"],
            "labelmaps": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/ROI/segmentation.nii"],
            "labelmap_values": [[1, 2, 3]]
            
        },
        "FT1033": {
            "images": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20140203102514_801.nii"],
            "labelmaps": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/ROI/segmentation.nii"],
            "labelmap_values": [[1, 2, 3]]
            
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

    # Example: Compose a set of transforms for augmentation
    from pyable_dataloader.transforms import Compose, RandomFlip, IntensityNormalization, RandomRotation,RandomAffine,RandomNoise,LabelMapOneHot
    transforms = Compose([
        # RandomRotation(rotation_range=[[-10, 10]]*3),
        #RandomFlip(axes=[1]),
        #RandomNoise(std=2),
        RandomAffine(scale_range=[0.98, 1.02], rotation_range=[[-5, 5]]*3, translation_range=[[-5, 5]]*3),
        IntensityNormalization(method='minmax',clip_percentile=(0, 99.5)),
        LabelMapOneHot(values=[1,2,3,4], meta_key='labelmap_values')
    ])
    # transforms=None  # Disable transforms for testing
    # Create dataset with augmentation and debug saving enabled
    dataset = PyableDataset(
        manifest=manifest,
        transforms=transforms,
        debug_save_dir="debug_outputs/simple_dataset_test",
        debug_save_format="nifti",
        dtype=None
    )

    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"Subject: {sample['id']}")
        print(f"  Image shape: {sample['images'].shape}")
        print(f"  Labelmap count: {len(sample['labelmaps'])}")
        print(f"  Meta: {sample['meta']}")

if __name__ == "__main__":
    main()
