# Data layout

Medical images and masks are not distributed with this repository.

A reproducible workflow for preprocessing the downloaded public datasets is in progress.
Until that workflow is available, users must prepare the source datasets themselves in the
layout referenced by the index.

`ml_dataset.csv` is the current research index used by the training notebook. Paths are
resolved from the project root; the current index expects the private image tree under
`../nii_data/`.

## Columns

| Column | Role |
| --- | --- |
| `case_id` | Required pseudonymous identifier used in evaluation output. |
| `t1_img_path` | Required relative path to the CE-T1w image. |
| `t1_seg_path` | Required relative path to the reference mask. |
| `fold` | Required fixed cross-validation fold. |
| `volume_mm3` | Optional tumor volume derived from the reference mask. |
| `quartile_label` | Optional zero-based volume-quartile label. |
Only `case_id`, the two paths, and `fold` are required by the current training workflow.
