# Data layout

Images and masks are not included. Prepare the source datasets in the layout referenced
by `ml_dataset.csv`; a reproducible preprocessing workflow is still in development.

Paths resolve from the `vestibular_schwannoma/` directory. The current index expects data
under `../nii_data/`.

## Columns

| Column | Role |
| --- | --- |
| `case_id` | Required pseudonymous identifier used in evaluation output. |
| `t1_img_path` | Required relative path to the CE-T1w image. |
| `t1_seg_path` | Required relative path to the reference mask. |
| `fold` | Required fixed cross-validation fold. |
| `volume_mm3` | Optional tumor volume derived from the reference mask. |
| `quartile_label` | Optional zero-based volume-quartile label. |

Only `case_id`, both paths, and `fold` are required.
