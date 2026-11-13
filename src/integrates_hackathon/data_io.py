"""Read the organizer-owned public Excel inputs.

The files in ``data`` are the authoritative challenge inputs.  This module
keeps workbook parsing out of the notebook and exposes simple, testable rows.
"""

from __future__ import annotations

import json
import re
from math import isclose
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIRECTORY = Path(__file__).resolve().parents[2] / "data"
BUILDING_PARAMETERS_PATH = DATA_DIRECTORY / "building_parameters.xlsx"
ENERGY_CONSUMPTION_PATH = DATA_DIRECTORY / "energy_consumption_values.xlsx"
EPW_DIRECTORY = DATA_DIRECTORY / "epw"
AMY_EPW_FILENAME = "ita_39100_bolzano_local_{year}.epw"

MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def parse_geometry_rings(
    value: str,
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[tuple[float, float], ...], ...],
]:
    """Return an outer latitude/longitude ring and any courtyard holes.

    Public rows without courtyards retain the original single-ring JSON shape.
    Rows with voids use ``[outer_ring, hole_ring, ...]``.
    """

    raw = json.loads(str(value))
    if not raw:
        raise ValueError("Geometry must contain at least one closed ring")
    rings = [raw] if isinstance(raw[0][0], (int, float)) else raw
    normalized = tuple(
        tuple((float(latitude), float(longitude)) for latitude, longitude in ring)
        for ring in rings
    )
    for ring in normalized:
        if len(ring) < 4 or ring[0] != ring[-1]:
            raise ValueError("Every geometry ring must be closed and contain at least three vertices")
    return normalized[0], normalized[1:]


def load_building_parameter_frame(
    path: str | Path = BUILDING_PARAMETERS_PATH,
) -> pd.DataFrame:
    """Return the public building rows from the Buildings worksheet."""

    frame = pd.read_excel(path, sheet_name="Buildings", header=3, engine="openpyxl")
    if "building_id" not in frame:
        raise ValueError("Buildings worksheet does not contain building_id")
    frame = frame.loc[frame["building_id"].notna()].copy()
    frame["building_id"] = frame["building_id"].astype(str).str.strip()
    if frame["building_id"].duplicated().any():
        raise ValueError("Duplicate building_id in building_parameters.xlsx")
    return frame.reset_index(drop=True)


def load_energy_source_rows(
    path: str | Path = ENERGY_CONSUMPTION_PATH,
) -> list[dict[str, Any]]:
    """Parse reported monthly values from every building worksheet.

    Sheet names are the canonical public building IDs. Rows without a unit are
    intentionally blank template rows and are ignored. Both the heating-only
    and the combined heating+DHW blocks are returned; :mod:`energy_consumption`
    decides which one is scored and converts standard cubic metres to kWh.
    The optional Annual Total column is used only as a consistency check.
    """

    with pd.ExcelFile(path, engine="openpyxl") as workbook:
        return _energy_rows_from_workbook(workbook)


def _energy_rows_from_workbook(workbook: pd.ExcelFile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sheet_name in workbook.sheet_names:
        raw = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
            engine="openpyxl",
        )
        public_id = str(sheet_name).strip()
        for first_row, last_row, end_use_scope in (
            (7, 15, "heating_only"),
            (19, 27, "heating_plus_dhw"),
        ):
            for row_index in range(first_row, last_row + 1):
                year = raw.iat[row_index, 0]
                unit = raw.iat[row_index, 1]
                if pd.isna(year) or pd.isna(unit):
                    continue
                source_unit = _normalize_unit(str(unit))
                values = [raw.iat[row_index, column] for column in range(2, 14)]
                if any(pd.isna(value) for value in values):
                    # Formula cells read as blank when the workbook was saved
                    # without cached results (for example by openpyxl).
                    raise ValueError(
                        f"{public_id} {int(year)} has an incomplete monthly series. "
                        "If the cells are formulas, open and re-save the workbook "
                        "in Excel so their calculated values are stored."
                    )
                annual_total = raw.iat[row_index, 14] if raw.shape[1] > 14 else None
                if not pd.isna(annual_total) and not isclose(
                    float(annual_total), sum(float(value) for value in values),
                    rel_tol=1e-9, abs_tol=1e-6,
                ):
                    raise ValueError(
                        f"{public_id} {int(year)} {end_use_scope} months do not sum "
                        f"to the stored Annual Total {float(annual_total)}"
                    )
                for month, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "building_id": public_id,
                            "year": int(year),
                            "month": month,
                            "source_value": float(value),
                            "source_unit": source_unit,
                            "end_use_scope": end_use_scope,
                            "quality_flag": "reported",
                        }
                    )
    if not rows:
        raise ValueError("No reported energy rows found")
    return rows


def amy_epw_path(year: int) -> Path:
    """Return the organizer-provided actual meteorological year EPW path."""

    return EPW_DIRECTORY / AMY_EPW_FILENAME.format(year=int(year))


def available_amy_years() -> tuple[int, ...]:
    """Discover AMY years from the EPW files shipped in ``data/epw``."""

    pattern = re.compile(r"^ita_39100_bolzano_local_(\d{4})\.epw$")
    years = sorted(
        int(match.group(1))
        for path in EPW_DIRECTORY.glob("ita_39100_bolzano_local_*.epw")
        if (match := pattern.match(path.name))
    )
    if not years:
        raise ValueError(f"No actual-year EPW files found in {EPW_DIRECTORY}")
    return tuple(years)


def _normalize_unit(value: str) -> str:
    compact = value.strip().lower().replace("³", "3")
    if "smc" in compact or "standard m3" in compact:
        return "Smc"
    if compact == "kwh":
        return "kWh"
    if compact == "m3":
        return "m3"
    raise ValueError(f"Unsupported energy unit: {value!r}")
