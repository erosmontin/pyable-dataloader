import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
try:
    from pyable.imaginable import SITKImaginable, LabelMapable,Roiable
except ImportError:
    raise ImportError(
        "pyable not found. Please install it first:\n"
        "  cd pyable && pip install -e ."
    )

import uuid

class PyableDataset(Dataset):
    """
    Simple PyTorch Dataset for medical images using pyable.
    - Each subject must have a reference image (for resampling/cropping)
    - If transforms (augmentation) are set, images/labelmaps are padded before augmentation
    - Resampling to reference is always performed after augmentation (if any)
    - Debug saving as NIfTI is supported
    - Only torch output (numpy in derived class)
    """
    def __init__(
        self,
        manifest,
        transforms=None,
        debug_save_dir=None,
        debug_save_format='nifti',
        dtype=torch.float32,
        pad_width=10,
    ):
        if isinstance(manifest, str):
            if manifest.endswith('.json'):
                import json
                with open(manifest, 'r') as f:
                    self.data = json.load(f)
            else:
                raise ValueError(f"Unsupported manifest file type: {manifest}")
        elif isinstance(manifest, dict):
            self.data = manifest
        else:
            raise ValueError(f"Unsupported manifest type: {type(manifest)}")
        self.ids = list(self.data.keys())
        self.transforms = transforms
        self.debug_save_dir = debug_save_dir
        self.debug_save_format = debug_save_format
        self.dtype = dtype
        self.pad_width = pad_width

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subject_id = self.ids[idx]
        item = self.data[subject_id]
        _s=str(uuid.uuid4()).split('-')[0] #index for debug saving

        roivalue = item.get('roivalues', item.get('roi_values', item.get('roivalue', 1)))
        labelmapsvalue = item.get('labelmapvalues', item.get('labelmap_values', item.get('labelmapvalue', None)))
        images = [SITKImaginable(filename=p) for p in item.get('images', [])]
        labelmaps = [LabelMapable(filename=p) for p in item.get('labelmaps', [])]
        
        # labelmapsvalue is optional in the manifest. If provided it tells the
        # dataset which label values to keep/encode for each labelmap. Supported
        # manifest forms remain:
        # - flat list of ints: [1,2,3] -> apply same values to every labelmap
        # - list-of-lists: [[1,2],[3],[1]] -> per-labelmap value lists
        # - scalar/int: 1 -> treat as single value for each labelmap
        #
        # If `labelmap_values` is provided we will apply `filterValues(lv)` to
        # each `LabelMapable`. If not provided we will leave the labelmaps as
        # loaded and auto-detect per-labelmap unique values (these are written
        # into `meta['labelmap_values']` later so downstream transforms can read
        # them if needed).
        normalized_labelmap_values = labelmapsvalue
        # If it's a numpy array, convert to python list for introspection
        try:
            import numpy as _np
            if isinstance(normalized_labelmap_values, _np.ndarray):
                normalized_labelmap_values = normalized_labelmap_values.tolist()
        except Exception:
            pass

        new_labelmaps = []
        detected_labelmap_values = []
        for i, lm in enumerate(labelmaps):
            lv = None
            # Determine per-labelmap values (list-of-lists or flat list) if provided
            if normalized_labelmap_values is not None:
                if isinstance(normalized_labelmap_values, (list, tuple)) and len(normalized_labelmap_values) > 0:
                    first = normalized_labelmap_values[0]
                    if isinstance(first, (list, tuple)) or (hasattr(first, '__iter__') and not isinstance(first, (str, bytes)) and not isinstance(first, int)):
                        # list-of-lists: take i-th entry if present
                        if len(normalized_labelmap_values) > i:
                            lv = normalized_labelmap_values[i]
                        else:
                            raise ValueError(f"Manifest 'labelmap_values' does not contain an entry for labelmap index {i} (subject {subject_id})")
                    else:
                        # flat list: apply same values to every labelmap
                        lv = normalized_labelmap_values
                else:
                    # scalar or other single value
                    lv = normalized_labelmap_values

            # If lv is provided, apply filterValues and fail loudly if it errors
            if lv is not None:
                try:
                    filtered = lm.filterValues(lv)
                    new_labelmaps.append(filtered)
                except Exception as e:
                    raise RuntimeError(f"Failed to apply filterValues for subject {subject_id}, labelmap {i} with values={lv}: {e}")
                # Record the requested values for meta
                try:
                    import numpy as _np
                    if isinstance(lv, _np.ndarray):
                        lv_list = lv.tolist()
                    else:
                        lv_list = list(lv) if isinstance(lv, (list, tuple)) else [int(lv)]
                except Exception:
                    lv_list = lv
                detected_labelmap_values.append(lv_list)
            else:
                # No filtering requested; keep original LabelMapable and detect its values
                new_labelmaps.append(lm)
                try:
                    if hasattr(lm, 'getImageAsNumpy'):
                        arr = lm.getImageAsNumpy()
                    else:
                        arr = np.asarray(lm)
                    vals = np.unique(arr).tolist()
                except Exception:
                    vals = []
                detected_labelmap_values.append(vals)

        labelmaps = new_labelmaps
        rois = [Roiable(filename=p, roivalue=roivalue) for p in item.get('rois', [])]
        
        
        ref_path = item.get('reference', None)
        if isinstance(ref_path, list):
            ref_path = ref_path[0] if ref_path else None


        if ref_path is not None:
            reference = SITKImaginable(filename=ref_path)
        
            # Pad reference image
            pad = self.pad_width if self.transforms is not None else 0
            padded_reference = reference.forkDuplicate()
            if pad > 0:
                lower = [pad] * 3
                upper = [pad] * 3
                padded_reference.padImage(lower, upper)

        
            if self.debug_save_dir is not None and pad > 0:
                from pathlib import Path
                debug_dir = Path(self.debug_save_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                padded_reference.writeImageAs(str(debug_dir / f"{subject_id}_padded_reference.nii.gz"))

            cir = item.get('cir_mm', None)
            # Center for transforms: cir_mm if present, else center of padded reference
            if cir is None:
                if hasattr(padded_reference, 'getImageCenterCoordinates'):
                    cir = np.array(padded_reference.getImageCenterCoordinates())
                else:
                    size = np.array(padded_reference.getImageSize())
                    # Use float indices for TransformContinuousIndexToPhysicalPoint
                    idx = (float(size[0]) / 2.0, float(size[1]) / 2.0, float(size[2]) / 2.0)
                    cir = padded_reference.getCoordinatesFromIndex(idx)
            # Build meta and include any manifest-provided label/roi value hints so
            # transforms (e.g., LabelMapOneHot) can read consistent label lists.
            meta = {'center': cir}
            # Normalize and attach manifest label/roi hints to meta under the
            # canonical keys expected by transforms.
            # If manifest provided explicit labelmap values, preserve that
            # structure in meta; otherwise populate meta using detected values
            # we computed earlier per-labelmap.
            if normalized_labelmap_values is not None:
                if isinstance(normalized_labelmap_values, (list, tuple)) and normalized_labelmap_values and all(isinstance(x, (int, np.integer)) for x in normalized_labelmap_values):
                    meta['labelmap_values'] = [list(normalized_labelmap_values)]
                else:
                    meta['labelmap_values'] = normalized_labelmap_values
            else:
                meta['labelmap_values'] = detected_labelmap_values

            if roivalue is not None:
                meta['roi_values'] = roivalue
            else:
                meta['roi_values'] = None

            # Batch transform pipeline
            if self.transforms is not None:
                transforms = self.transforms
                if not isinstance(transforms, (list, tuple)):
                    transforms = [transforms]
                # Capture original Imaginable references so we can map converted
                # numpy outputs back to the correct spatial geometry per-object.
                orig_image_refs = [img.getImage() for img in images]
                orig_labelmap_refs = [lm.getImage() for lm in labelmaps]
                orig_roi_refs = [r.getImage() for r in rois]

                for t in transforms:
                    images, rois, labelmaps = t(images, rois, labelmaps, meta)
                # Transforms are expected to return Imaginable-like objects
                # (SITKImaginable / LabelMapable / Roiable). If they do not,
                # raise an explicit error so the problem is fixed at the source.
                def _all_imaginable(objs):
                    return isinstance(objs, list) and all(hasattr(o, 'resampleOnTargetImage') for o in objs)

                if not (_all_imaginable(images) and _all_imaginable(labelmaps) and _all_imaginable(rois)):
                    raise TypeError(
                        f"Transforms must return Imaginable/LabelMapable/Roiable objects."
                        f" Subject {subject_id} produced non-Imaginable outputs."
                    )
            # After all transforms, resample each to the original reference
            images = [img.resampleOnTargetImage(reference) for img in images]
            labelmaps = [lm.resampleOnTargetImage(reference) for lm in labelmaps]
            rois = [roi.resampleOnTargetImage(reference) for roi in rois]

            # Debug save transformed images, labelmaps, rois
            if self.debug_save_dir is not None:
                from pathlib import Path
                debug_dir = Path(self.debug_save_dir)
                debug_dir.mkdir(parents=True, exist_ok=True)
                for i, img in enumerate(images):
                    img.writeImageAs(str(debug_dir / f"{_s}_{subject_id}_transformed_image_{i}.nii.gz"))
                for i, lm in enumerate(labelmaps):
                    lm.writeImageAs(str(debug_dir / f"{_s}_{subject_id}_transformed_labelmap_{i}.nii.gz"))
                for i, roi in enumerate(rois):
                    roi.writeImageAs(str(debug_dir / f"{_s}_{subject_id}_transformed_roi_{i}.nii.gz"))
            meta = {
                'subject_id': subject_id,
                'spacing': reference.getImageSpacing(),
                'origin': reference.getImageOrigin(),
                'direction': reference.getImageDirection(),
                'size': reference.getImageSize(),
                'source_paths': {
                    'images': item.get('images', []),
                    'labelmaps': item.get('labelmaps', []),
                    'reference': ref_path
                }
            }
        # Build the returned sample (convert Imaginable objects to numpy arrays)
        def _to_numpy_list(objs):
            out = []
            for o in objs:
                if hasattr(o, 'getImageAsNumpy'):
                    arr = o.getImageAsNumpy()
                    # ensure numeric dtype
                    out.append(np.asarray(arr))
                else:
                    out.append(np.asarray(o))
            return out

        images_np_list = _to_numpy_list(images)
        if len(images_np_list) > 1:
            images_stacked = np.stack(images_np_list, axis=0)
        else:
            images_stacked = images_np_list[0][np.newaxis, ...]

        labelmaps_np = _to_numpy_list(labelmaps)
        rois_np = _to_numpy_list(rois)

        sample = {
            'id': subject_id,
            'images': images_stacked,
            'rois': rois_np,
            'labelmaps': labelmaps_np,
            'meta': meta
        }

        return sample
