"""Authoritative metadata for the upstream vestibular schwannoma datasets.

The data licenses below apply to the downloaded datasets, independently of the
license used for this software repository.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceAsset:
    """A file published as part of an upstream dataset."""

    name: str
    filename: str
    url: str
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """Citation, license, and download metadata for an upstream dataset."""

    key: str
    title: str
    doi: str
    license_name: str
    license_url: str
    page_url: str
    assets: tuple[SourceAsset, ...]

    def asset(self, name: str) -> SourceAsset:
        """Return a named asset, raising a useful error for unknown names."""

        for asset in self.assets:
            if asset.name == name:
                return asset
        available = ", ".join(asset.name for asset in self.assets)
        raise KeyError(
            f"Unknown asset {name!r} for {self.key}; choose from: {available}"
        )


CROSSMODA_2022 = DatasetSource(
    key="crossmoda2022",
    title="Cross-Modality Domain Adaptation Challenge 2022 (crossMoDA)",
    doi="10.5281/zenodo.6504722",
    license_name="CC BY-NC-SA 4.0",
    license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    page_url="https://zenodo.org/records/6504722",
    assets=(
        SourceAsset(
            name="training",
            filename="crossmoda2022_training.zip",
            url=(
                "https://zenodo.org/records/6504722/files/crossmoda2022_training.zip?download=1"
            ),
            checksum="md5:8d68fcd44eaee6b0a4371ce344e775cd",
        ),
    ),
)


TCIA_VESTIBULAR_SCHWANNOMA_SEG = DatasetSource(
    key="vestibular_schwannoma_seg",
    title=(
        "Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: "
        "An Open Annotated Dataset and Baseline Algorithm"
    ),
    doi="10.7937/TCIA.9YTJ-5Q73",
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    page_url=(
        "https://www.cancerimagingarchive.net/collection/vestibular-schwannoma-seg/"
    ),
    assets=(
        SourceAsset(
            name="manifest",
            filename="Vestibular-Schwannoma-SEG-Feb-2021-manifest.tcia",
            url=(
                "https://www.cancerimagingarchive.net/wp-content/uploads/"
                "Vestibular-Schwannoma-SEG-Feb-2021-manifest.tcia"
            ),
        ),
        SourceAsset(
            name="contours",
            filename="Vestibular-Schwannoma-SEG-contours-Mar-2021.zip",
            url=(
                "https://www.cancerimagingarchive.net/wp-content/uploads/"
                "Vestibular-Schwannoma-SEG-contours-Mar-2021.zip"
            ),
        ),
        SourceAsset(
            name="matrices",
            filename="Vestibular-Schwannoma-SEG_matrices-Mar-2021.zip",
            url=(
                "https://www.cancerimagingarchive.net/wp-content/uploads/"
                "Vestibular-Schwannoma-SEG_matrices-Mar-2021.zip"
            ),
        ),
        SourceAsset(
            name="modality_mapping",
            filename="DirectoryNamesMappingModality.csv",
            url=(
                "https://www.cancerimagingarchive.net/wp-content/uploads/"
                "DirectoryNamesMappingModality.csv"
            ),
        ),
    ),
)

# Short aliases for callers that use cohort names rather than repository names.
TCIA_VS_SEG = TCIA_VESTIBULAR_SCHWANNOMA_SEG
QUEEN_SQUARE = TCIA_VESTIBULAR_SCHWANNOMA_SEG
TILBURG = CROSSMODA_2022

DATASET_SOURCES = {
    TCIA_VESTIBULAR_SCHWANNOMA_SEG.key: TCIA_VESTIBULAR_SCHWANNOMA_SEG,
    CROSSMODA_2022.key: CROSSMODA_2022,
}


__all__ = [
    "CROSSMODA_2022",
    "DATASET_SOURCES",
    "DatasetSource",
    "QUEEN_SQUARE",
    "SourceAsset",
    "TCIA_VESTIBULAR_SCHWANNOMA_SEG",
    "TCIA_VS_SEG",
    "TILBURG",
]
