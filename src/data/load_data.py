"""Portable readers for the supplied IMD Maitri source files.

The CSV intentionally has no header row.  Its column order is verified against
the overlapping NetCDF subset, rather than inferred from the first data row.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

CSV_COLUMNS = ("timestamp", "tempr", "rh", "ws", "wd", "ap")
REQUIRED_SOURCE_COLUMNS = ("timestamp", "tempr", "rh", "ap")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_csv_path() -> Path:
    """Find the supplied raw CSV without hard-coding a machine path."""
    root = project_root()
    candidates = (root / "Dataset" / "imd_maitri.csv", root / "data" / "raw" / "imd_maitri.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find imd_maitri.csv in Dataset/ or data/raw/.")


def load_imd_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Load the headerless IMD Maitri CSV and parse its timestamp column.

    Missing sensor readings remain as ``-999`` at this stage so validation can
    report the original data-quality markers before preprocessing transforms them.
    """
    source = Path(path) if path else default_csv_path()
    if not source.is_file():
        raise FileNotFoundError(f"CSV dataset not found: {source}")
    frame = pd.read_csv(source, header=None, names=CSV_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


def load_imd_netcdf(path: str | Path) -> pd.DataFrame:
    """Load the NetCDF subset when its optional ``h5py`` dependency is installed.

    NetCDF stores ``obstime`` as minutes since 2000-01-15. Its variable names
    follow the same verified order as the CSV. This reader is kept separate
    because the CSV is the fuller source used by the reproducible baseline.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - depends on installation
        raise ImportError("NetCDF support requires h5py; install requirements.txt.") from error
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"NetCDF dataset not found: {source}")
    with h5py.File(source, "r") as dataset:
        required = ("obstime", "tempr", "rh", "ws", "wd", "ap")
        absent = set(required).difference(dataset.keys())
        if absent:
            raise ValueError(f"Unexpected NetCDF schema; missing: {sorted(absent)}")
        origin = pd.Timestamp("2000-01-15 00:00:00")
        return pd.DataFrame({
            "timestamp": origin + pd.to_timedelta(dataset["obstime"][:], unit="m"),
            **{column: dataset[column][:] for column in CSV_COLUMNS[1:]},
        })
