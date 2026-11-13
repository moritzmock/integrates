"""Validated organizer-generated baseline results.

The expensive pyBuildingEnergy baseline is generated once by the organizers
and shipped as challenge data.  Participant notebooks only read and validate
the cache; changing ``calibration.py`` cannot invalidate it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from math import isclose
from pathlib import Path
from typing import Mapping, Sequence

from .data_io import BUILDING_PARAMETERS_PATH, DATA_DIRECTORY, ENERGY_CONSUMPTION_PATH
from .models import BuildingCase, MonthlyConsumption


BASELINE_CACHE_PATH = DATA_DIRECTORY / "precomputed_baseline.csv"
BASELINE_METADATA_PATH = DATA_DIRECTORY / "precomputed_baseline_metadata.json"
BASELINE_SCHEMA_VERSION = 1

# Git may check text files out with CRLF (Windows) or LF (macOS, Linux, ZIP
# downloads). These are hashed with normalized line endings so every clean
# copy of the repository produces the same signature.
TEXT_SUFFIXES = frozenset({".py", ".epw"})

# Packages whose versions can change simulated energy. They are pinned in
# environment.yml and recorded with the baseline so drift is reported.
NUMERIC_PACKAGES = (
    "pybuildingenergy",
    "numpy",
    "pandas",
    "scipy",
    "pvlib",
    "holidays",
    "workalendar",
    "openpyxl",
    "python-dateutil",
    "pytz",
    "tzdata",
)


def installed_package_versions() -> dict[str, str]:
    versions = {}
    for name in NUMERIC_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _signature_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        content = content.replace(b"\r\n", b"\n")
    return content


def baseline_input_signature() -> str:
    """Hash every organizer input or source file that affects the baseline."""

    package_dir = Path(__file__).resolve().parent
    paths = [
        BUILDING_PARAMETERS_PATH,
        ENERGY_CONSUMPTION_PATH,
        package_dir / "bui.py",
        package_dir / "building_inputs.py",
        package_dir / "challenge_inputs.py",
        package_dir / "data_io.py",
        package_dir / "energy_consumption.py",
        package_dir / "heating_schedule.py",
        package_dir / "simulation.py",
        package_dir / "epw_io.py",
        package_dir / "pbe_compat.py",
        package_dir / "simulation_workers.py",
        package_dir / "window_simulation.py",
        *sorted((DATA_DIRECTORY / "epw").glob("*.epw")),
    ]
    digest = hashlib.sha256()
    digest.update(f"baseline-schema:{BASELINE_SCHEMA_VERSION}\n".encode())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Baseline input is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_signature_bytes(path))
    return digest.hexdigest()


def save_precomputed_baseline(
    cases: Sequence[BuildingCase],
    baseline: Mapping[str, Sequence[MonthlyConsumption]],
    *,
    cache_path: str | Path = BASELINE_CACHE_PATH,
    metadata_path: str | Path = BASELINE_METADATA_PATH,
) -> tuple[Path, Path]:
    """Write the complete monthly cache and provenance metadata."""

    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)
    rows = []
    for case in cases:
        if case.building_id not in baseline:
            raise ValueError(f"Missing baseline for {case.building_id}")
        expected = {(row.year, row.month) for row in case.measurements}
        values = tuple(baseline[case.building_id])
        received = {(row.year, row.month) for row in values}
        if received != expected or len(values) != len(expected):
            raise ValueError(
                f"Baseline periods for {case.building_id} do not match measurements"
            )
        rows.extend(
            {
                "building_id": case.building_id,
                "year": int(row.year),
                "month": int(row.month),
                "baseline_kwh": float(row.consumption_kwh),
            }
            for row in sorted(values)
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("building_id", "year", "month", "baseline_kwh"),
        )
        writer.writeheader()
        writer.writerows(rows)

    try:
        pbe_version = version("pybuildingenergy")
    except PackageNotFoundError:
        pbe_version = "unknown"
    metadata = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "pybuildingenergy",
        "pybuildingenergy_version": pbe_version,
        "package_versions": installed_package_versions(),
        "input_signature": baseline_input_signature(),
        "row_count": len(rows),
        "building_ids": [case.building_id for case in cases],
        "measurement_period_totals_kwh": {
            case.building_id: sum(
                float(row.consumption_kwh) for row in baseline[case.building_id]
            )
            for case in cases
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return cache_path, metadata_path


def load_precomputed_baseline(
    cases: Sequence[BuildingCase],
    *,
    validate_signature: bool = True,
    cache_path: str | Path = BASELINE_CACHE_PATH,
    metadata_path: str | Path = BASELINE_METADATA_PATH,
) -> dict[str, tuple[MonthlyConsumption, ...]]:
    """Load the organizer cache and reject stale or incomplete results."""

    cache_path = Path(cache_path)
    metadata_path = Path(metadata_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Precomputed baseline is not available. Organizers must run "
            "scripts/generate_baseline_cache.py before publishing the challenge."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != BASELINE_SCHEMA_VERSION:
        raise ValueError("Precomputed baseline schema version is not supported")
    try:
        installed_pbe_version = version("pybuildingenergy")
    except PackageNotFoundError:
        installed_pbe_version = "unknown"
    if metadata.get("pybuildingenergy_version") != installed_pbe_version:
        raise ValueError(
            "Precomputed baseline uses pybuildingenergy "
            f"{metadata.get('pybuildingenergy_version')}, but this environment has "
            f"{installed_pbe_version}. Use the challenge environment before running."
        )
    recorded_versions = metadata.get("package_versions", {})
    installed_versions = installed_package_versions()
    drift = {
        name: (recorded, installed_versions.get(name, "unknown"))
        for name, recorded in recorded_versions.items()
        if installed_versions.get(name, "unknown") != recorded
    }
    if drift:
        details = ", ".join(
            f"{name} {installed} (baseline used {recorded})"
            for name, (recorded, installed) in sorted(drift.items())
        )
        warnings.warn(
            f"Package versions differ from the baseline environment: {details}. "
            "Simulated energy may not match official results. Run "
            "'conda env update -f environment.yml --prune' and restart the kernel.",
            stacklevel=2,
        )
    expected_ids = [case.building_id for case in cases]
    cached_ids = list(metadata.get("building_ids", ()))
    if not set(expected_ids).issubset(cached_ids):
        raise ValueError("Precomputed baseline does not contain every selected building")
    if validate_signature:
        current = baseline_input_signature()
        if metadata.get("input_signature") != current:
            raise ValueError(
                "Precomputed baseline is stale because organizer inputs or simulation "
                "code changed. Regenerate it before distributing the notebook."
            )

    by_building: dict[str, list[MonthlyConsumption]] = {}
    with cache_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_building.setdefault(row["building_id"], []).append(
                MonthlyConsumption(
                    int(row["year"]),
                    int(row["month"]),
                    float(row["baseline_kwh"]),
                )
            )

    row_count = sum(len(values) for values in by_building.values())
    if row_count != int(metadata.get("row_count", -1)):
        raise ValueError("Precomputed baseline row count does not match its metadata")
    if set(by_building) != set(cached_ids):
        raise ValueError("Precomputed baseline building list does not match its metadata")

    result = {}
    for case in cases:
        expected = {(row.year, row.month) for row in case.measurements}
        values = tuple(
            row for row in sorted(by_building.get(case.building_id, ()))
            if (row.year, row.month) in expected
        )
        received = {(row.year, row.month) for row in values}
        if received != expected or len(values) != len(expected):
            raise ValueError(
                f"Cached baseline periods for {case.building_id} do not match measurements"
            )
        if len(values) == len(by_building[case.building_id]):
            expected_total = float(
                metadata["measurement_period_totals_kwh"][case.building_id]
            )
            received_total = sum(row.consumption_kwh for row in values)
            if not isclose(received_total, expected_total, rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError(
                    f"Cached baseline values for {case.building_id} do not match metadata"
                )
        result[case.building_id] = values
    return result
