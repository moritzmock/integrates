"""Stable participant submission contract and validation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from .models import BuildingCase, ParameterSpec, finite_real


FORBIDDEN_SUBMISSION_IMPORTS = frozenset(
    {
        "integrates_hackathon",
        "pybuildingenergy",
        "multiprocessing",
        "concurrent.futures",
        "subprocess",
    }
)

CalibrationFunction = Callable[
    [Callable[[Sequence[float]], float], Sequence[ParameterSpec], int, int],
    Mapping[str, float],
]


@dataclass(frozen=True)
class Submission:
    team_name: str
    algorithm_name: str
    calibrate: CalibrationFunction
    source_path: Path
    source_sha256: str


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_submission(path: str | Path | None = None) -> Submission:
    source_path = Path(path) if path is not None else repository_root() / "calibration.py"
    source_path = source_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Participant file not found: {source_path}. Copy or create calibration.py."
        )
    source_bytes = source_path.read_bytes()
    source_sha256 = sha256(source_bytes).hexdigest()
    spec = spec_from_file_location("challenge_participant_calibration", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {source_path}")
    module = module_from_spec(spec)
    check_submission_imports(source_bytes, source_path)
    # Compile the same byte sequence that was hashed, so the exported digest
    # identifies the exact participant code used by this run.
    exec(compile(source_bytes, str(source_path), "exec"), module.__dict__)
    return _submission_from_module(module, source_path, source_sha256)


def check_submission_imports(source: bytes | str, source_path: str | Path = "calibration.py") -> None:
    """Reject imports that would bypass the budgeted objective.

    Participant code may only evaluate candidates through the objective it is
    given. Importing the challenge package or simulator, or starting processes,
    could run simulations outside the evaluation budget.
    """

    tree = ast.parse(source, filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # Include "concurrent.futures" for "from concurrent import futures".
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        else:
            continue
        for name in names:
            parts = name.split(".")
            prefixes = {".".join(parts[: index + 1]) for index in range(len(parts))}
            if prefixes & FORBIDDEN_SUBMISSION_IMPORTS:
                raise ImportError(
                    f"{Path(source_path).name} line {node.lineno} imports {name!r}. "
                    "Submissions must evaluate candidates only through the objective "
                    f"they receive; these imports are not allowed: "
                    f"{', '.join(sorted(FORBIDDEN_SUBMISSION_IMPORTS))}."
                )


def _submission_from_module(
    module: ModuleType,
    source_path: Path,
    source_sha256: str,
) -> Submission:
    team_name = str(getattr(module, "TEAM_NAME", "")).strip()
    algorithm_name = str(getattr(module, "ALGORITHM_NAME", "")).strip()
    calibrate = getattr(module, "calibrate", None)
    if not team_name:
        raise ValueError("calibration.py must define a non-empty TEAM_NAME")
    if not algorithm_name:
        raise ValueError("calibration.py must define a non-empty ALGORITHM_NAME")
    if not callable(calibrate):
        raise ValueError("calibration.py must define callable calibrate(...)")
    return Submission(
        team_name,
        algorithm_name,
        calibrate,
        source_path,
        source_sha256,
    )


def validate_parameter_result(
    case: BuildingCase, result: Any
) -> dict[str, float]:
    if not isinstance(result, Mapping):
        raise TypeError("calibrate(...) must return a mapping of parameter names to values")
    expected = set(case.parameter_names)
    received = set(result)
    if received != expected:
        raise ValueError(
            f"Returned parameter keys do not match the contract. "
            f"Missing={sorted(expected - received)}; extra={sorted(received - expected)}"
        )
    validated: dict[str, float] = {}
    by_name = {parameter.name: parameter for parameter in case.parameters}
    for name in case.parameter_names:
        value = finite_real(result[name], f"Returned value for {name!r}")
        spec = by_name[name]
        if not spec.lower <= value <= spec.upper:
            raise ValueError(
                f"{name!r}={value} is outside [{spec.lower}, {spec.upper}]"
            )
        validated[name] = value
    return validated

