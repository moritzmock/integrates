"""Minimal EPW table I/O used by the shortened simulation window."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


EPW_HEADER_ROWS = 8
EPW_COLUMNS = (
    "Year", "Month", "Day", "Hour", "Minute",
    "Data Source and Uncertainty Flags", "Dry Bulb Temperature",
    "Dew Point Temperature", "Relative Humidity",
    "Atmospheric Station Pressure", "Extraterrestrial Horizontal Radiation",
    "Extraterrestrial Direct Normal Radiation",
    "Horizontal Infrared Radiation Intensity", "Global Horizontal Radiation",
    "Direct Normal Radiation", "Diffuse Horizontal Radiation",
    "Global Horizontal Illuminance", "Direct Normal Illuminance",
    "Diffuse Horizontal Illuminance", "Zenith Luminance", "Wind Direction",
    "Wind Speed", "Total Sky Cover", "Opaque Sky Cover", "Visibility",
    "Ceiling Height", "Present Weather Observation", "Present Weather Codes",
    "Precipitable Water", "Aerosol Optical Depth", "Snow Depth",
    "Days Since Last Snowfall", "Albedo", "Liquid Precipitation Depth",
    "Liquid Precipitation Quantity",
)


def read_epw_table(epw_path: str | Path) -> tuple[list[str], pd.DataFrame]:
    path = Path(epw_path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) <= EPW_HEADER_ROWS:
        raise ValueError(f"EPW file is too short: {path}")
    header = lines[:EPW_HEADER_ROWS]
    rows = [line.split(",") for line in lines[EPW_HEADER_ROWS:] if line.strip()]
    frame = pd.DataFrame(rows, columns=EPW_COLUMNS)
    for column in EPW_COLUMNS:
        if column != "Data Source and Uncertainty Flags":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return header, frame


def write_epw_table(
    header: Iterable[str], frame: pd.DataFrame, output_path: str | Path
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        for line in header:
            stream.write(str(line).rstrip("\n") + "\n")
        frame.to_csv(stream, header=False, index=False)
    return path
