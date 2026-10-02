# Reproduce the study dataset

The shared `ml_dataset.csv` defines **344 native-space CE-T1 scans and tumour
masks**, their order, volumes, quartile labels, and fixed cross-validation folds.
`reference_dataset.json` fingerprints the current study data. Preparation must
match those voxel values and geometry before publishing an output directory.

| Source | Public cases | Study cases | Selection |
| --- | ---: | ---: | --- |
| Queen Square / TCIA Vestibular-Schwannoma-SEG, version 2 | 242 | 239 | Exclude `vs_gk_8`, `vs_gk_73`, and `vs_gk_131` |
| crossMoDA 2022 / Tilburg | 105 | 105 | `training_source`, `center == Tilburg`, CE-T1 only |

`vs_gk_73` is a meningioma, as recorded in the historical combined-data notebook.
Cases 8 and 131 were removed following radiologist review in fastMONAI commit
`d8bd2de`; the specific findings are not recorded there. The eight other gaps in
Queen Square numbering are already absent from the public source. crossMoDA
London cases overlap Queen Square and are omitted. Target/validation images,
T2 images, cochlea labels, backup masks, HUS data, and other collections are
not included in the prepared study directory.

## Setup

Run these commands from `research/vestibular_schwannoma/`. Python 3.10+ and a CPU
are sufficient; preparation does not import fastMONAI or PyTorch or require a GPU.

```bash
pip install -r data/requirements.txt
python scripts/prepare_data.py --help
```

Allow roughly 80–100 GB free for downloads, extraction, and prepared copies.
Downloaded sources remain untouched. Existing output directories are refused;
use a new `--output` directory for another run.

## 1. Download by script

The default DICOM downloader is the pip package
[`idc-index`](https://github.com/ImagingDataCommons/idc-index), installed by
`data/requirements.txt`. The same TCIA collection is available in
[NCI Imaging Data Commons (IDC)](https://portal.imaging.datacommons.cancer.gov/collections/vestibular_schwannoma_seg/).
No Data Retriever installation, browser download, account, or cloud credentials
are needed. The tested package version is pinned to `0.12.5` with IDC v24 metadata.

```bash
python scripts/prepare_data.py download
```

This downloads and checksum-verifies the approximately 7.8 GB
[crossMoDA 2022 training archive](https://zenodo.org/records/6504722), extracts it,
downloads the official TCIA manifest and support files, and downloads the **239
Queen Square T1 MR series** from IDC (29,400 DICOM instances, approximately 15.5 GB).
The shared CSV selects cases; T2, radiation-therapy series, and the three excluded
cases are omitted from IDC downloads. Missing or ambiguous T1 series abort before
transferring data. Re-running resumes/reuses cached downloads. IDC's sync compares
existing files with remote sizes; preparation then checks exact voxel hashes and
geometry. `queen_square/idc_manifest.json` records the CSV checksum, IDC and package
versions, selected series UIDs, expected instance counts, and completion status.

Sources can be acquired separately:

```bash
python scripts/prepare_data.py download --dataset tilburg
python scripts/prepare_data.py download --dataset queen-square
```

Expected layout:

```text
data/raw/
├── queen_square/
│   ├── assets/                       # manifest, contours, matrices, mapping
│   ├── idc_manifest.json             # selected T1 series and download provenance
│   └── dicom/VS-SEG-001/<StudyUID>/MR_<SeriesUID>/...
├── crossmoda2022/
│   ├── crossmoda2022_training.zip
│   └── crossmoda2022_training/
│       ├── infos_source_training.csv
│       ├── training_source/
│       └── training_target/          # downloaded but not used
└── corrected_labels/                # extracted from the bundled 17-mask ZIP
```

The 17 replacement masks are included in this repository as
[`releases/vs_corrected_labels_v1.zip`](releases/vs_corrected_labels_v1.zip).
`download` checks and extracts this archive automatically; no separate hosting
account or download is needed. Use `download --dataset labels` to extract just
the masks. An alternative HTTPS location can be supplied with
`--corrected-label-url URL`; the archive must match `corrected_labels.json`.

## Alternatives: TCIA Data Retriever CLI or graphical app

To use the current [TCIA Data Retriever](https://github.com/TCIA/data-retriever)
binary instead of IDC:

```bash
python scripts/prepare_data.py download --dataset queen-square \
  --tcia-backend retriever --retriever /path/to/TCIA_Data_Retriever
```

This invokes `--cli`, `-i`, `-o`, `--skip-existing`,
`--directory-mode descriptive`, and `--accept-data-policy` to download the full
collection manifest. Providing `--retriever` also selects this backend when
`--tcia-backend` is omitted. The older Java NBIA Data Retriever uses different CLI
flags and is not interchangeable with this executable.

Download DICOM using the collection manifest and Data Retriever from the
[TCIA collection page](https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-seg/).
Classic and descriptive directory names are supported. Download support files:

```bash
python scripts/prepare_data.py download --dataset queen-square --assets-only
```

Supply the DICOM collection with `prepare --queen-square
/path/to/Vestibular-Schwannoma-SEG`. DICOM metadata selects the T1 MR series;
radiation-therapy files are not used as MR images.

## 2. Prepare and verify

For a fresh checkout without an existing `../nii_data/` directory:

```bash
python scripts/prepare_data.py prepare
python scripts/prepare_data.py verify
```

Default output is `../nii_data/`, matching the shared CSV and training notebook.
Alternate output writes its own CSV with updated paths and unchanged case order,
folds, volumes, and quartiles:

```bash
python scripts/prepare_data.py prepare --output data/prepared
python scripts/prepare_data.py verify --data-root data/prepared
python scripts/train_5fold.py --data-csv data/prepared/ml_dataset.csv --models unet
```

CLI paths resolve relative to this project, even when invoked from elsewhere.
The bundled masks are used automatically if they have not already been extracted.
`--workers` controls CPU parallelism; default 2. To reuse the existing VS_Seg data:

```bash
python scripts/prepare_data.py prepare \
  --from-existing /home/sathiesh/ml_projects/VS_Seg \
  --output data/prepared
```

The shortcut uses that checkout's `nii_data/queen_square_data`, `crossmoda_data`,
and `errors_fixed_nl`. Individual `--queen-square-nifti`, `--crossmoda`, and
`--corrected-labels` options accept equivalent data elsewhere.

Preparation selects only shared CSV cases, converts native Queen Square T1 DICOM,
applies 17 NL masks, retains Tilburg tumour label 1, and removes only these
verified six-neighbour components:

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

Unexpected signatures fail. Other masks retain every tumour region. The final
inventory is **344 images + 344 masks**, a manifest, and a generated CSV. Each
pair must match reference voxel fingerprints, geometry, and CSV volume. Missing
corrections or mismatched data abort and remove staged partial output.

Queen Square rasterization preserves the historical SimpleITK conversion,
rounded contour coordinates, tumour-name selection, and polygon filling used
in this study. It is not a general-purpose converter. Substituting Slicer or
changing rasterization requires a new dataset version and reference. Acquisition
does not register or resample; training preprocessing remains separate.

Fingerprints hash C-order uint16 images and uint8 integer masks. Compression and
mask storage dtype do not affect them; affine and spacing are checked separately.
The historical encoding decodes label 1 as `0.9999999997671694` in 103 Tilburg
masks. Integer rounding within `1e-6` accounts for this encoding; actual fractional
labels are rejected.

## Sharing the 17 replacement masks

Correction cases: **4, 5, 29, 35, 43, 44, 45, 47, 55, 56, 61, 63, 71, 74, 76,
96, 119**. These are new NL masks drawn for scans whose original tumour
annotations contained errors.

Contributors: **Njål Lura** and **Satheshkumar Kaliyugarassan**.
The release uses **CC BY 4.0** and asks readers to cite the accompanying study
paper when available.

The [`releases/`](releases/) folder contains one ZIP,
a [short README](releases/README.md), and a [license notice](releases/LICENSE.txt).
These small masks are versioned with the preparation code on GitHub. Model
weights are distributed separately on Hugging Face. Use a tagged repository
release to identify the code, case index, and masks used for a study.

Maintainer packaging command:

```bash
python scripts/prepare_data.py bundle-labels --data-root ../nii_data
```

The ZIP contains 17 masks, a TCIA case mapping, provenance manifest, README,
and license. Fresh NIfTI headers omit annotation-editor text and extensions
while preserving geometry and tumour voxels. Existing bundles and asset metadata
are refused. If the archive changes, update its checksum and size in
`corrected_labels.json`.

Original Queen Square data are
[CC BY 4.0](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053);
crossMoDA data are [CC BY-NC-SA 4.0](https://zenodo.org/records/6504722). Their
licenses are separate from the Apache-2.0 software license. This release contains
new tumour masks for Queen Square images.

## Maintaining the reference

The reference was frozen from the current dataset. Do not regenerate it to make
mismatched preparation pass. Fingerprint an explicitly reviewed future version
using a new path:

```bash
python scripts/prepare_data.py freeze-reference \
  --data-root /path/to/reviewed/data --output data/reference_dataset_v2.json
```

Audit indexed pairs in mixed legacy storage with `verify --data-root /path/to/data
--allow-extra-files`. Normal preparation/verification enforces exact inventory.

Download helpers were adapted from `VS_Seg/preprocess_data`; conversion reproduces
`VS_Seg/preprocessing/simple_data_conversion.py` (Apache-2.0).
Native preparation from the original local TCIA/crossMoDA downloads was checked
against all 344 reference pairs, including the 17-label ZIP and all eight
component cleanups. Standalone verification also checked the exact 688-file
inventory and the generated CSV's unchanged folds and metadata.

The pip-installed IDC route (`idc-index==0.12.5`, IDC v24) was tested with live
downloads of cases 1, 4, and 12: 360 DICOM instances, approximately 190 MB. All
three converted images and final masks matched the frozen reference's voxel
hashes, geometry, and CSV volumes, including case 4's human correction and case
12's 79-voxel cleanup. Removing a downloaded test slice and rerunning restored
it with the same SHA-256. IDC metadata covered all 239 study T1 series; the full
15.5 GB cohort was not redownloaded for this check.

```bash
python -m pytest -q tests/workflow/test_data_preparation.py \
  tests/workflow/test_data_download.py tests/workflow/test_data_conversion.py \
  tests/workflow/test_data_idc.py
```
