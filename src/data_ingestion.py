"""Load the UCI Online Retail spreadsheet, or a local CSV with the same columns."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger("retention.ingestion")

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/352/online+retail"
REQUIRED_SOURCE_COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw file is missing and cannot be downloaded."""


def _looks_like_table(path: Path) -> bool:
    return path.suffix.lower() in {".xlsx", ".xls", ".csv"} and path.is_file()


def find_raw_file(raw_dir: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    if raw_dir.is_file() and _looks_like_table(raw_dir):
        return raw_dir
    if not raw_dir.exists():
        return None
    candidates = [
        p
        for p in raw_dir.rglob("*")
        if _looks_like_table(p) and "gitkeep" not in p.name.lower()
    ]
    if not candidates:
        return None
    # Prefer the canonical UCI workbook name, then the newest file.
    preferred = [p for p in candidates if "online" in p.name.lower() and p.suffix.lower() == ".xlsx"]
    pool = preferred or candidates
    return sorted(pool, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def download_uci_online_retail(dest_dir: Path, url: str) -> Path:
    """Download and extract the UCI Online Retail archive. Returns the table path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "online_retail.zip"
    logger.info("Downloading UCI Online Retail from %s", url)
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as exc:  # noqa: BLE001 - surface a single actionable error
        raise DatasetNotFoundError(
            "Automatic download failed. Download Online Retail.xlsx from "
            f"{UCI_DATASET_PAGE} and place it in {dest_dir}. Error: {exc}"
        ) from exc
    if zip_path.stat().st_size < 10_000:
        raise DatasetNotFoundError(
            f"Downloaded file at {zip_path} is too small to be the dataset. "
            f"Download it manually from {UCI_DATASET_PAGE}."
        )
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".xlsx", ".xls", ".csv")) and not name.endswith("/")
            ]
            if not members:
                raise DatasetNotFoundError(
                    f"The archive {zip_path} did not contain a spreadsheet or CSV."
                )
            extracted = archive.extract(members[0], dest_dir)
    except zipfile.BadZipFile as exc:
        raise DatasetNotFoundError(
            f"{zip_path} is not a valid zip file. Download the dataset manually "
            f"from {UCI_DATASET_PAGE}."
        ) from exc
    logger.info("Extracted raw dataset to %s", extracted)
    return Path(extracted)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_table(path: Path) -> pd.DataFrame:
    logger.info("Reading raw table from %s", path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, engine="openpyxl")
    else:
        raise DatasetNotFoundError(f"Unsupported raw file type: {path.suffix}")
    frame.columns = [str(column).strip() for column in frame.columns]
    logger.info("Raw table shape: %s rows, %s columns", len(frame), list(frame.columns))
    missing = [column for column in REQUIRED_SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "The raw file is missing required columns "
            f"{missing}. Found columns: {list(frame.columns)}. "
            "This project expects the UCI Online Retail schema "
            "(InvoiceNo, StockCode, Description, Quantity, InvoiceDate, "
            "UnitPrice, CustomerID, Country)."
        )
    return frame


def extract_local_archive(raw_dir: Path) -> Path | None:
    """Extract a spreadsheet from a zip that is already in the raw directory."""
    if not raw_dir.is_dir():
        return None
    for archive_path in sorted(raw_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith((".xlsx", ".xls", ".csv")) and not name.endswith("/")
                ]
                if not members:
                    continue
                return Path(archive.extract(members[0], raw_dir))
        except zipfile.BadZipFile:
            logger.warning("Skipping unreadable archive %s", archive_path)
    return None


def acquire_raw_table(raw_dir: Path, explicit: Path | None, download_url: str) -> tuple[pd.DataFrame, Path]:
    path = find_raw_file(raw_dir, explicit)
    if path is None:
        path = extract_local_archive(raw_dir if raw_dir.is_dir() else raw_dir.parent)
    if path is None:
        logger.info("No local raw file found under %s. Attempting download.", raw_dir)
        path = download_uci_online_retail(raw_dir, download_url)
    frame = load_raw_table(path)
    return frame, path
