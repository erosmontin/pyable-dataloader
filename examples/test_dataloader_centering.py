#!/usr/bin/env python3
"""Test script for PyableDataset centering and cropping.

Creates a small manifest with two subjects and verifies that the ROI centroid
is centered correctly in the resampled reference volume. Saves debug NIfTI
outputs to `debug_outputs/` for visual inspection.

Usage:
    python examples/test_dataloader_centering.py

Make sure you're using the `able` conda environment when running.
"""
import os
import numpy as np
import SimpleITK as sitk
import warnings

# Ensure local src/ is first on sys.path so we exercise our edited code
import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(REPO_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from pyable_dataloader import PyableDataset

# Always show warnings in this diagnostic script
warnings.simplefilter('always')


def make_manifest():
    return {
        'FT1013': {
            'images': ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20131021083019_701.nii"],
            'rois': ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/ROI/segmentation.nii"],
            'labelmaps': ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1013/BL/ROI/segmentation.nii"]
        },
        'FT1033': {
            'images': ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/MRI/_3D_WATSf_(true_sag)_CLEAR_20140203102514_801.nii"],
            'rois': ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/ROI/segmentation.nii"],
            'labelmaps': ["/data/MYDATA/OXFORD/FAIT_2_NII/FT1033/BL/ROI/segmentation.nii"]
        }
    }


def compute_mask_centroid_phys(sitk_img: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk_img)  # z,y,x
    mask_idx = np.argwhere(arr > 0)
    if mask_idx.size == 0:
        return None
    center_voxel = mask_idx.mean(axis=0)  # (z, y, x)
    # Convert to SI index order (x,y,z)
    cont_index = (float(center_voxel[2]), float(center_voxel[1]), float(center_voxel[0]))
    phys = sitk_img.TransformContinuousIndexToPhysicalPoint(cont_index)
    return np.array(phys)


def main():
    # Ensure we import the local src copy rather than an installed package
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_path = os.path.join(repo_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import pyable_dataloader
    print('Using pyable_dataloader package from:', pyable_dataloader.__file__)

    manifest = make_manifest()

    debug_dir = os.path.abspath('debug_outputs')
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir, exist_ok=True)

    dataset = PyableDataset(
        manifest=manifest,
        target_size=[64, 64, 64],
        target_spacing=[2.0, 2.0, 2.0],
        stack_channels=True,
        transforms=None,
        debug_save_dir=debug_dir,
        debug_save_format='nifti',
        return_numpy=True,
        force_reload=True  # ensure we run the pipeline fresh
    )

    print(f"Loaded dataset with {len(dataset)} subjects. Debug outputs in: {debug_dir}")

    for i in range(len(dataset)):
        subj = dataset.ids[i]
        print('\n--- Subject:', subj, '---')
        # Quick check: load Roiable and compute centroid using dataset helper
        item = dataset.data[subj]
        for roi_path in item.get('rois', []):
            if os.path.exists(roi_path):
                from pyable.imaginable import Roiable
                roi_obj = Roiable(filename=roi_path)
                roi_obj.resampleOnCanonicalSpace()
                computed = dataset._compute_roi_center(roi_obj)
                print('  dataset._compute_roi_center returned:', computed)

                # Also compute centroid directly from the Roiable's SimpleITK image
                try:
                    sitk_roi_before = sitk.Image(roi_obj.getImage())
                    phys_before = compute_mask_centroid_phys(sitk_roi_before)
                    print('  centroid on the ROI object (before resample to reference):', phys_before)
                except Exception as e:
                    print('  Could not compute centroid on ROI object:', e)
        # Use get_numpy_item to save temp files and get meta
        sample = dataset.get_numpy_item(i, as_nifti=True, save_to_files=True)

        meta = sample.get('meta', {})
        print('meta keys:', list(meta.keys()))

        # If debug images were saved by dataset._save_debug_images, they are in debug_dir
        saved_roi_paths = sample.get('rois', [])
        saved_image_paths = sample.get('images', [])
        saved_labelmap_paths = sample.get('labelmaps', [])

        # Print reference origin/size/spacing
        spacing = np.array(meta.get('spacing', [np.nan, np.nan, np.nan]))
        origin = np.array(meta.get('origin', [np.nan, np.nan, np.nan]))
        size = np.array(meta.get('size', [np.nan, np.nan, np.nan]))
        print('spacing:', spacing)
        print('origin:', origin)
        print('size:', size)

        # Compute center of reference
        ref_center = origin + (size * spacing) / 2.0
        print('reference_center:', ref_center)

        # If meta includes roi_center (computed before centering), print it
        roi_center_meta = meta.get('roi_center', None)
        if roi_center_meta is not None:
            print('meta roi_center:', np.array(roi_center_meta))

        # Now inspect saved ROI files and compute centroid
        centroids = []
        for p in saved_roi_paths:
            if p and os.path.exists(p):
                sitk_roi = sitk.ReadImage(p)
                phys = compute_mask_centroid_phys(sitk_roi)
                centroids.append((p, phys))
                print('saved ROI:', p, 'centroid_phys:', phys)
            else:
                print('saved ROI missing:', p)

        # Check distance between computed centroid and reference center
        for p, phys in centroids:
            if phys is None:
                print('  Warning: ROI empty in', p)
                continue
            dist = np.linalg.norm(ref_center - phys)
            print(f'  Distance from reference center to ROI centroid: {dist:.3f} mm')
            if dist > 5.0:
                print('  WARNING: ROI centroid is more than 5mm from reference center. Investigate centering/cropping logic.')
            else:
                print('  OK: ROI centroid close to reference center.')

    print('\nDone. Check NIfTI outputs in', debug_dir, 'for visual inspection.')


if __name__ == '__main__':
    main()
