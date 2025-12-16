from pyable import imaginable as ima
import tempfile
import numpy as np

def update_manifest_with_reference(manifest, orientation='LPS', resolution=[2.0, 2.0, 2.0], reference_idx=0, target_size=None):
    """
    For each subject in the manifest, create a reference image with the specified orientation and resolution,
    save it to /tmp, and update the 'reference' field in the manifest.
    """
    from pyable import imaginable as ima
    import os
    updated_manifest = manifest.copy()
    for k, v in manifest.items():
        if not v['images']:
            continue
        print(f"Creating reference for subject {k} from image {v['images'][reference_idx]}")
        img_path = v['images'][reference_idx]
        img = ima.Imaginable(img_path)
        img.dicomOrient(orientation)
        img.changeImageSpacing(resolution)
        # If a target_size is provided, build a centered reference image of that size and resample
        if target_size is not None:
            # target_size expected as [D, H, W]
            try:
                ts = [int(x) for x in target_size]
                # SimpleITK expects size as (X, Y, Z) -> (W, H, D)
                sitk_size = (ts[2], ts[1], ts[0])
            except Exception:
                raise ValueError("target_size must be an iterable of three ints [D,H,W]")

            # Compute center of the source image in physical coordinates
            try:
                center_phys = np.array(img.getImageCenterCoordinates(), dtype=float)
            except Exception:
                # fallback: compute from size/spacing
                s = np.array(img.getImageSize())
                center_phys = img.getCoordinatesFromIndex((s[0]/2.0, s[1]/2.0, s[2]/2.0))

            # Direction cosines (as 3x3 matrix)
            dir_tuple = img.getImageDirection()
            dir_mat = np.array(dir_tuple, dtype=float).reshape((3, 3))

            spacing = (float(resolution[0]), float(resolution[1]), float(resolution[2]))

            # index center in SITK index order (X,Y,Z) -> use sitk_size
            idx_center = np.array([(sitk_size[0] - 1) / 2.0, (sitk_size[1] - 1) / 2.0, (sitk_size[2] - 1) / 2.0], dtype=float)
            phys_offset = dir_mat.dot(idx_center * np.array(spacing))
            origin = center_phys - phys_offset

            # Create SITK reference image with desired size/spacing/origin/direction
            imgref = __import__('SimpleITK').Image(sitk_size, __import__('SimpleITK').sitkFloat32)
            imgref.SetSpacing(spacing)
            imgref.SetOrigin(tuple(origin.tolist()))
            imgref.SetDirection(tuple(dir_tuple))

            # Wrap into Imaginable and resample source image onto it
            ref_im = ima.SITKImaginable(image=imgref)
            # Compute target reference path and skip work if already present
            ref_path = os.path.join(tempfile.gettempdir(), f"{k}_reference.nii.gz")
            if not os.path.exists(ref_path):
                try:
                    resampled = img.resampleOnTargetImage(ref_im)
                except Exception:
                    # fallback: try passing the underlying SimpleITK image
                    resampled = img.resampleOnTargetImage(ref_im.getImage())
                resampled.writeImageAs(ref_path)
            else:
                print(f"Reference exists for subject {k}: {ref_path} (skipping creation)")
        else:
            # Save reference image to /tmp (no size enforcement)
            ref_path = os.path.join(tempfile.gettempdir(), f"{k}_reference.nii.gz")
            if not os.path.exists(ref_path):
                img.writeImageAs(ref_path)
            else:
                print(f"Reference exists for subject {k}: {ref_path} (skipping write)")
        updated_manifest[k]['reference'] = ref_path
    return updated_manifest