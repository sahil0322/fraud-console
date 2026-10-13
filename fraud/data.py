"""Loading PaySim (or the synthetic stand-in) into the internal schema."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PAYSIM_RENAME


class SchemaError(ValueError):
    pass


def load_paysim(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a PaySim CSV, check its columns, drop isFlaggedFraud, rename to snake_case.

    The column check exists because the PRD's column names were written from memory.
    If Kaggle's file differs, this fails loudly and lists exactly what is missing.
    """
    raw = pd.read_csv(path, nrows=nrows)
    missing = [c for c in PAYSIM_RENAME if c not in raw.columns]
    if missing:
        raise SchemaError(
            f"{path} is missing expected PaySim columns: {missing}. "
            f"Found: {list(raw.columns)}. Update PAYSIM_RENAME in fraud/config.py."
        )
    df = raw[list(PAYSIM_RENAME)].rename(columns=PAYSIM_RENAME)  # isFlaggedFraud is not selected
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", range(1, len(df) + 1))
    return df
