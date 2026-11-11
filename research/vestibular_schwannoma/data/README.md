# Reproduce the study dataset

`ml_dataset.csv` defines the **344 native-space CE-T1 scans and tumour masks**: their order,
volumes, quartile labels and fixed cross-validation folds. `reference_dataset.json`
fingerprints the data; preparation must match it exactly.

| Source | Public cases | Study cases | Selection |
| --- | ---: | ---: | --- |
| Queen Square / TCIA Vestibular-Schwannoma-SEG, version 2 | 242 | 239 | Excludes `vs_gk_73` (meningioma), and `vs_gk_8` and `vs_gk_131` (radiologist review, commit `d8bd2de`) |
| crossMoDA 2022 / Tilburg | 105 | 105 | `training_source`, `center == Tilburg`, CE-T1 only |

crossMoDA London cases overlap Queen Square and are left out, as are T2, target/validation
images, cochlea labels and HUS data.

## Setup

Run from `research/vestibular_schwannoma/`. Python 3.10+ and a CPU are enough; fastMONAI and
PyTorch are not needed. Allow 80-100 GB of free disk.

```bash
pip install -r data/requirements.txt
python scripts/prepare_data.py download   # or --dataset tilburg | queen-square | labels
python scripts/prepare_data.py prepare
python scripts/prepare_data.py verify
```

## Download

`download` fetches the 7.8 GB [crossMoDA 2022 training archive](https://zenodo.org/records/6504722)
and the 239 Queen Square T1 series (about 15.5 GB) from
[NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/collections/vestibular_schwannoma_seg/)
with `idc-index` 0.12.5 (IDC v24). No account or credentials are needed. Downloads are
checksum-verified and resume when rerun. Files go to `data/raw/`; existing sources are never
changed.

To use the [TCIA Data Retriever](https://github.com/TCIA/data-retriever) instead:

```bash
python scripts/prepare_data.py download --dataset queen-square \
  --tcia-backend retriever --retriever /path/to/TCIA_Data_Retriever
```

For a manual TCIA download, fetch the support files with
`download --dataset queen-square --assets-only` and pass the collection to
`prepare --queen-square /path/to/Vestibular-Schwannoma-SEG`.

## Prepare and verify

`prepare` writes to `../nii_data/` by default, matching the CSV. Another `--output` gets its
own CSV with the new paths and the same folds:

```bash
python scripts/prepare_data.py prepare --output data/prepared
python scripts/prepare_data.py verify --data-root data/prepared
python scripts/train_5fold.py --data-csv data/prepared/ml_dataset.csv --models unet
```

`prepare --from-existing /path/to/VS_Seg` reuses an existing VS_Seg checkout, and `--workers`
sets the CPU parallelism (default 2). Existing output folders are refused.

Preparation converts the Queen Square T1 DICOM, applies the 17 corrected masks, keeps
Tilburg tumour label 1, and removes only these verified stray components:

| Case | Kept voxels | Removed voxels |
| --- | ---: | ---: |
| `vs_gk_12` | 6685 | 79 |
| `vs_gk_13` | 2927 | 10 |
| `vs_gk_53` | 2756 | 52 |
| `vs_gk_86` | 3336 | 55 |
| `vs_gk_87` | 4144 | 17 |
| `vs_gk_250` | 24946 | 1 |
| `crossmoda2022_etz_38` | 3616 | 1 |
| `crossmoda2022_etz_67` | 3568 | 1 |

Every other mask keeps all its regions. The result is 344 images, 344 masks, a manifest and a
CSV, each pair checked against the reference voxel hashes, geometry and volume. Any mismatch
aborts and removes the partial output.

The Queen Square conversion reproduces the study's historical SimpleITK rasterization; it is
not a general converter, and changing it needs a new dataset version. Fingerprints hash
C-order uint16 images and uint8 masks, so compression and storage type don't matter.

## Corrected masks

`corrected_labels/vs_corrected_labels_v1.zip` holds 17 new masks for Queen Square cases 4, 5,
29, 35, 43, 44, 45, 47, 55, 56, 61, 63, 71, 74, 76, 96 and 119, whose original annotations
contained errors. Drawn by Njål Lura and Satheshkumar Kaliyugarasan, licensed CC BY 4.0
([README](corrected_labels/README.md)). `download` extracts the ZIP automatically.

The ZIP is rebuilt with `python scripts/prepare_data.py bundle-labels --data-root ../nii_data`;
an existing bundle is never overwritten. Queen Square data are
[CC BY 4.0](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053) and
crossMoDA data [CC BY-NC-SA 4.0](https://zenodo.org/records/6504722), separate from the
Apache-2.0 code license.

## Reference

Don't regenerate the reference to make a mismatch pass. Fingerprint a reviewed new version to a
new file:

```bash
python scripts/prepare_data.py freeze-reference \
  --data-root /path/to/reviewed/data --output data/reference_dataset_v2.json
```

All 344 pairs were verified from the original downloads. The IDC route was tested on cases 1,
4 and 12. Tests:

```bash
python -m pytest -q tests/workflow/test_data_preparation.py \
  tests/workflow/test_data_download.py tests/workflow/test_data_conversion.py \
  tests/workflow/test_data_idc.py
```
