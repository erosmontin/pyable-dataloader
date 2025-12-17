**Manifest Structure and Label/ROI Rules**

- **Images**: `images` — list of image file paths or Imaginable objects used as the input volumes.
- **Labelmaps**: `labelmaps` — list of segmentation file paths or arrays. Supply an explicit array/list of expected label values via the manifest (recommended key: `labelmap_values`) or via transform constructor (e.g., `LabelMapOneHot(values=[...])`) to guarantee consistent channel ordering. Any voxels in the labelmap whose value is not present in the provided `labelmap_values` will be filtered out by the encoding transforms (they will not produce channels or will be remapped to background).
- **ROIs**: `rois` — list of ROI file paths or arrays. By default the dataset expects ROIs to be binary masks (value `1`). If your ROI file contains values other than `1`, include the intended ROI value(s) in the manifest (recommended key: `roi_values`) so the dataset can extract the correct region. For example, set `manifest[subject]['roi_values'] = [1]` or provide a per-ROI list when multiple ROIs are present.

Example manifest snippet (per-subject entry):

```
{
  "subject_id": {
    "images": ["/path/to/image.nii.gz"],
    "labelmaps": ["/path/to/labelmap.nii.gz"],
    "rois": ["/path/to/roi.nii.gz"],
    "labelmap_values": [[0, 1, 2]],   # list-of-lists allowed for multiple labelmaps
    "roi_values": [1]                 # single value or list for ROI masks
  }
}
```

Prefer adding `labelmap_values` to the manifest (or passing `values` to labelmap transforms) so the dataset produces stable, reproducible channels across samples and training runs.
