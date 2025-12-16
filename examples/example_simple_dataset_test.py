import os
from pyable_dataloader.dataset import PyableDataset
from pyable import imaginable as ima

def main():
    # Example manifest for two subjects
    manifest = {
        "FT1013": {
            "images": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20131021083019_701.nii"],
            "labelmaps": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/ROI/segmentation.nii"],
            "reference": "/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20131021083019_701.nii"
        },
        "FT1033": {
            "images": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20140203102514_801.nii"],
            "labelmaps": ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/ROI/segmentation.nii"],
            "reference": "/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20140203102514_801.nii"
        }
    }
    
    for k, v in manifest.items():
        print(f"Subject ID: {k}")
        for img_path in v["images"]:
            R=ima.Imaginable(img_path)
            R.changeImageSpacing([2.0, 2.0, 2.0])
            R.reorientToLPS()
            R.cropImage([0,0,50], [0,0,0])  # Example crop
            r=os.path.join("debug_outputs", f"{img_path.replace('/', '_')}_image_resampled.nii")
            R.writeImageAs(r)
        manifest[k]["reference"] = [r]  # Update reference to resampled image
    manifest["FT1013"]["reference"] = None
    # Create dataset (no augmentation, debug saving enabled)
    dataset = PyableDataset(
        manifest=manifest,
        transforms=None,
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
