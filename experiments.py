#!/usr/bin/env python
"""Run, save, and compare calibration experiments from a JSON settings file.

This is the command-line counterpart of the challenge notebook. It drives the
organizer workflow (simulation, budget enforcement, scoring, export) for every
experiment in a settings file and never modifies ``calibration.py``.

Commands
--------
    python experiments.py [run]   [--settings experiments.json] [--index N] [--only TEXT]
                                  [--force] [--strict]
    python experiments.py status  [--settings experiments.json] [--strict]
    python experiments.py compare [--settings experiments.json] [--all] [--runs TEXT]
                                  [--out DIR] [--no-plots]

``run`` (the default) calibrates, scores, and saves each experiment under
``outputs/runs/<run-id>/``. An experiment whose saved result is still valid is
skipped. ``status`` shows what ``run`` would do. ``compare`` only reads saved
results and writes tables and plots; it never simulates.

Settings file
-------------
    {
      "defaults": {
        "buildings": ["building_01", "building_02", "building_03"],
        "calibration_years": [2018, 2019],
        "evaluation_budget": 48,
        "parallel_workers": 12,
        "seed": 2026,
        "parameters": "all"
      },
      "experiments": [
        {"algorithm": ["random_search", "bayesian_optimization"]},
        {"algorithm": "bayesian_optimization", "buildings": ["building_02"],
         "calibration_years": [2018], "evaluation_budget": [12, 24]}
      ]
    }

``algorithm``, ``evaluation_budget``, ``parallel_workers``, and ``seed`` may be
lists; every combination becomes one experiment. ``parameters`` is ``"all"``
(all twelve active, starting at the organizer defaults), a list of names, or
an object ``{name: start value or null}``.

How an algorithm is selected
----------------------------
``calibration.py`` chooses its method with the module constant
``ALGORITHM_NAME``. For each experiment the runner writes a copy of
``calibration.py`` in which only that line is changed and hands the copy to
the organizer workflow. The copy used is stored with the result as
``calibration_snapshot.py``.

When is a saved result reused?
------------------------------
A saved result is reused when the settings and the organizer inputs are the
same and the code the selected algorithm runs is unchanged. Two checks decide
that (see ``analyse_source`` and ``run_probe``):

1. Static fingerprint. Starting at ``calibrate``, every function, class,
   constant, and import the algorithm can reach is hashed from the parsed
   code, so comments, formatting, and docstrings do not count. A dict keyed
   by algorithm name (``_ALGORITHMS``) and ``if ALGORITHM_NAME == "..."``
   branches are followed only for the selected algorithm, so other algorithms
   can be added or edited freely. An equal fingerprint means the result is
   reused immediately.
2. Behaviour probe. If the fingerprint differs (for example a shared helper
   was extended for a new algorithm), the algorithm is run against three cheap
   synthetic objectives with the experiment's real parameter space, budget,
   batch size, and seed. If it proposes exactly the same candidates as the
   saved run's code did, and every changed function was executed by the
   probe, the result is reused. A changed constant or a changed function the
   probe never reached forces a rerun, because the probe cannot vouch for it.

``--strict`` uses check 1 only. ``--force`` always reruns. A rerun moves the
old result to ``outputs/runs/_superseded/`` instead of deleting it.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import re
import shutil
import socket
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent
if (REPO / "src").is_dir() and str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))  # also runs in spawned worker processes
os.environ.setdefault("MPLBACKEND", "Agg")  # servers have no display

CALIBRATION_PATH = REPO / "calibration.py"
DEFAULT_SETTINGS = REPO / "experiments.json"
DEFAULT_RUNS_DIR = REPO / "outputs" / "runs"
SELECTOR = "ALGORITHM_NAME"
RECORD_FILE = "experiment.json"
SNAPSHOT_FILE = "calibration_snapshot.py"
EXPORT_FILES = ("summary.csv", "monthly_results.csv", "evaluation_trace.csv", "submission_result.json")
RECORD_SCHEMA = 2
OFFICIAL_SEED = 2026

DEFAULTS = {
    "buildings": ["building_01", "building_02", "building_03"],
    "calibration_years": [2018, 2019],
    "evaluation_budget": 48,
    "parallel_workers": 12,
    "seed": OFFICIAL_SEED,
    "parameters": "all",
}
GRID_KEYS = ("algorithm", "evaluation_budget", "parallel_workers", "seed")
ENTRY_KEYS = {"name", "algorithm", *DEFAULTS}


class ExperimentError(Exception):
    """A problem with one experiment that should not stop the others."""


def _sha(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Experiment:
    index: int
    name: str | None
    algorithm: str
    buildings: tuple[str, ...]
    calibration_years: tuple[int, ...]
    evaluation_budget: int
    parallel_workers: int
    seed: int
    parameters: tuple[tuple[str, float | None], ...]  # organizer order

    def settings(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "buildings": list(self.buildings),
            "calibration_years": list(self.calibration_years),
            "evaluation_budget": self.evaluation_budget,
            "parallel_workers": self.parallel_workers,
            "seed": self.seed,
            "parameters": dict(self.parameters),
        }

    @property
    def run_id(self) -> str:
        """Folder name: ``[<name>__]<algorithm>__budget<N>__<years>__<buildings>__w<workers>[...]``.

        A named experiment starts with its name and needs no parameter hash,
        because its name identifies the parameter selection.
        """
        from integrates_hackathon.challenge_inputs import parameter_defaults

        safe = lambda text: re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("_")
        parts = [safe(self.name)] if self.name else []
        parts += [
            safe(self.algorithm),
            f"budget{self.evaluation_budget}",
            "+".join(str(year) for year in self.calibration_years),
            "+".join(b.replace("building_", "b") for b in self.buildings),
            f"w{self.parallel_workers}",
        ]
        if self.seed != OFFICIAL_SEED:
            parts.append(f"seed{self.seed}")
        if not self.name and dict(self.parameters) != parameter_defaults():
            parts.append("p" + _sha(json.dumps(dict(self.parameters), sort_keys=True), 6))
        return "__".join(parts)


def _resolve_parameters(value) -> tuple[tuple[str, float | None], ...]:
    from integrates_hackathon.challenge_inputs import parameter_defaults

    defaults = parameter_defaults()
    if value == "all":
        selection = dict(defaults)
    elif isinstance(value, list):
        selection = {name: (defaults[name] if name in value else None) for name in defaults}
        unknown = set(value) - set(defaults)
    elif isinstance(value, dict):
        selection = {name: value.get(name) for name in defaults}
        unknown = set(value) - set(defaults)
    else:
        raise ValueError('"parameters" must be "all", a list of names, or an object')
    if value != "all" and unknown:
        raise ValueError(f"Unknown parameter names: {sorted(unknown)}; known: {sorted(defaults)}")
    return tuple((name, None if v is None else float(v)) for name, v in selection.items())


def load_experiments(path: Path) -> list[Experiment]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Settings file not found: {path}")
    except json.JSONDecodeError as error:
        raise SystemExit(f"Settings file {path} is not valid JSON: {error}")
    unknown = set(raw) - {"defaults", "experiments"}
    if unknown or not isinstance(raw.get("experiments"), list) or not raw["experiments"]:
        raise SystemExit(
            f'{path}: expected an object with "experiments" (non-empty list) and optional '
            f'"defaults"; unexpected keys: {sorted(unknown)}'
        )
    defaults = {**DEFAULTS, **raw.get("defaults", {})}
    if set(defaults) - set(DEFAULTS):
        raise SystemExit(f'{path}: unknown keys in "defaults": {sorted(set(defaults) - set(DEFAULTS))}')

    experiments: list[Experiment] = []
    seen: dict[str, tuple[int, dict]] = {}  # run id -> (entry position, settings)
    for position, entry in enumerate(raw["experiments"]):
        where = f"{path}: experiments[{position}]"
        if not isinstance(entry, dict) or "algorithm" not in entry:
            raise SystemExit(f'{where} must be an object with an "algorithm"')
        if set(entry) - ENTRY_KEYS:
            raise SystemExit(f"{where}: unknown keys {sorted(set(entry) - ENTRY_KEYS)}; allowed: {sorted(ENTRY_KEYS)}")
        if "name" in entry and (not isinstance(entry["name"], str) or not re.search(r"[A-Za-z0-9]", entry["name"])):
            raise SystemExit(f'{where}: "name" must be a non-empty string')
        merged = {**defaults, **entry}
        grid =[merged[key] if isinstance(merged[key], list) else [merged[key]] for key in GRID_KEYS]
        try:
            parameters = _resolve_parameters(merged["parameters"])
            for algorithm, budget, workers, seed in product(*grid):
                experiment = Experiment(
                    index=len(experiments),
                    name=entry.get("name"),
                    algorithm=str(algorithm),
                    buildings=tuple(sorted(str(b) for b in merged["buildings"])),
                    calibration_years=tuple(sorted(int(y) for y in merged["calibration_years"])),
                    evaluation_budget=int(budget),
                    parallel_workers=int(workers),
                    seed=int(seed),
                    parameters=parameters,
                )
                if min(experiment.evaluation_budget, experiment.parallel_workers) < 1:
                    raise ValueError("evaluation_budget and parallel_workers must be at least 1")
                if not experiment.buildings or not experiment.calibration_years:
                    raise ValueError("buildings and calibration_years must not be empty")
                if experiment.run_id in seen:
                    first, settings = seen[experiment.run_id]
                    if settings != experiment.settings():
                        raise ValueError(
                            f"same folder name {experiment.run_id!r} as experiments[{first}] but different "
                            "settings (usually the same name for different parameter selections); "
                            "give each experiment its own name"
                        )
                    print(f"Note: {where} repeats experiment {experiment.run_id}; listed once.")
                    continue
                seen[experiment.run_id] = (position, experiment.settings())
                experiments.append(experiment)
        except (ValueError, TypeError) as error:
            raise SystemExit(f"{where}: {error}")
    return experiments


def available_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):  # respects the Slurm CPU allocation on Linux
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


# --------------------------------------------------------------------------
# Static fingerprint of the code one algorithm can reach
# --------------------------------------------------------------------------


class _Specializer(ast.NodeTransformer):
    """Resolve ``ALGORITHM_NAME`` comparisons for one algorithm and drop dead branches."""

    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm

    @staticmethod
    def _is_selector(node) -> bool:
        return isinstance(node, ast.Name) and node.id == SELECTOR

    @staticmethod
    def _text(node):
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    def _texts(self, node):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values = [self._text(item) for item in node.elts]
            if all(value is not None for value in values):
                return set(values)
        return None

    def _fold(self, test) -> bool | None:
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left, op, right = test.left, test.ops[0], test.comparators[0]
            if isinstance(op, (ast.Eq, ast.NotEq)):
                other = right if self._is_selector(left) else left if self._is_selector(right) else None
                if other is not None and self._text(other) is not None:
                    return (self._text(other) == self.algorithm) == isinstance(op, ast.Eq)
            if isinstance(op, (ast.In, ast.NotIn)) and self._is_selector(left):
                options = self._texts(right)
                if options is not None:
                    return (self.algorithm in options) == isinstance(op, ast.In)
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            inner = self._fold(test.operand)
            return None if inner is None else not inner
        if isinstance(test, ast.BoolOp):
            values = [self._fold(value) for value in test.values]
            decisive = not isinstance(test.op, ast.And)  # one True decides "or"; one False decides "and"
            if any(value is decisive for value in values):
                return decisive
            if all(value is (not decisive) for value in values):
                return not decisive
        return None

    def visit_If(self, node):
        self.generic_visit(node)
        taken = self._fold(node.test)
        if taken is None:
            return node
        return node.body if taken else node.orelse

    def visit_IfExp(self, node):
        self.generic_visit(node)
        taken = self._fold(node.test)
        if taken is None:
            return node
        return node.body if taken else node.orelse

    def visit_Match(self, node):
        self.generic_visit(node)
        if not self._is_selector(node.subject):
            return node
        for case in node.cases:
            if case.guard is not None:
                return node
            patterns = case.pattern.patterns if isinstance(case.pattern, ast.MatchOr) else [case.pattern]
            values = []
            for pattern in patterns:
                if isinstance(pattern, ast.MatchAs) and pattern.pattern is None:
                    return case.body  # wildcard: always matches
                if not isinstance(pattern, ast.MatchValue) or self._text(pattern.value) is None:
                    return node
                values.append(self._text(pattern.value))
            if self.algorithm in values:
                return case.body
        return []


def _selector_is_constant(tree: ast.Module) -> bool:
    """True when ALGORITHM_NAME is bound exactly once, at module level, to a string."""

    stores = [
        node
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == SELECTOR and not isinstance(node.ctx, ast.Load))
        or (isinstance(node, (ast.Global, ast.Nonlocal)) and SELECTOR in node.names)
    ]
    top = [
        stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.Assign, ast.AnnAssign))
        and any(isinstance(t, ast.Name) and t.id == SELECTOR for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]))
    ]
    return (
        len(stores) == 1
        and len(top) == 1
        and isinstance(top[0].value, ast.Constant)
        and isinstance(top[0].value.value, str)
    )


def _is_docstring(stmt) -> bool:
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


@dataclass
class CodeInfo:
    fingerprint: str
    parts: dict[str, dict]  # name -> {"kind": ..., "hash": ...}
    line_index: list[tuple[int, int, tuple[str, ...]]]  # top-level statement line ranges
    modules: set[str]  # top-level modules imported by the reachable code
    algorithms: list[str] | None  # names in the algorithm registry, if one was found
    current_algorithm: str | None  # value of ALGORITHM_NAME in the source


def analyse_source(text: str, algorithm: str) -> CodeInfo:
    """Fingerprint the code ``algorithm`` can reach from ``calibrate``.

    Each reachable top-level name gets its own hash, so a later comparison can
    say exactly what changed. Hashes come from the parsed code after removing
    docstrings and annotations and after resolving ``ALGORITHM_NAME`` for this
    algorithm, which makes them independent of comments and formatting.
    """

    tree = ast.parse(text)
    fold = _selector_is_constant(tree)

    def normalise(nodes) -> ast.Module:
        module = ast.Module(body=copy.deepcopy(list(nodes)), type_ignores=[])
        if fold:
            module = _Specializer(algorithm).visit(module)
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and _is_docstring(node.body[0]):
                    node.body = node.body[1:] or [ast.Pass()]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.returns = None
                arguments = node.args
                for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg):
                    if arg is not None:
                        arg.annotation = None
        return module

    def names_in(module) -> set[str]:
        found = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
        for node in ast.walk(module):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                found.update(node.names)
        return found

    # Index every top-level statement by the names it binds.
    index: dict[str, list[tuple[str, ast.AST, object]]] = {}
    line_index: list[tuple[int, int, tuple[str, ...]]] = []
    always: list[ast.AST] = []  # module-level code that runs regardless of the algorithm
    future: list[str] = []
    for position, stmt in enumerate(tree.body):
        if position == 0 and _is_docstring(stmt):
            continue
        bound: list[str] = []
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(stmt, ast.ClassDef) else "def"
            index.setdefault(stmt.name, []).append((kind, stmt, None))
            bound = [stmt.name]
        elif isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            future += sorted(alias.name for alias in stmt.names)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                if alias.name == "*":
                    always.append(stmt)
                    continue
                name = (alias.asname or alias.name).split(".")[0]
                index.setdefault(name, []).append(("import", stmt, alias))
                bound.append(name)
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            bound = sorted({n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)})
            for name in bound:
                index.setdefault(name, []).append(("assign", stmt, None))
        else:
            # if/try/with/for blocks bind names; plain expressions are side effects.
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                    bound.append(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.append(node.name)
                elif isinstance(node, ast.alias):
                    bound.append((node.asname or node.name).split(".")[0])
            bound = sorted(set(bound))
            for name in bound:
                index.setdefault(name, []).append(("block", stmt, None))
            always.append(stmt)
        first = min([stmt.lineno, *(d.lineno for d in getattr(stmt, "decorator_list", []))])
        line_index.append((first, stmt.end_lineno or stmt.lineno, tuple(bound)))

    def registry_entry(stmt):
        """The value registered for ``algorithm`` if stmt is a dict keyed by algorithm name."""
        value = getattr(stmt, "value", None)
        if isinstance(value, ast.Dict) and value.keys and all(
            isinstance(key, ast.Constant) and isinstance(key.value, str) for key in value.keys
        ):
            for key, item in zip(value.keys, value.values):
                if key.value == algorithm:
                    return item
        return None

    if "calibrate" not in index:
        raise ExperimentError("calibration.py defines no calibrate function")

    parts: dict[str, dict] = {}
    modules: set[str] = set()
    pending = ["calibrate"]
    if always:
        module = normalise(always)
        parts["<module code>"] = {"kind": "module", "hash": _sha(ast.dump(module))}
        pending += sorted(names_in(module))
    if future:
        parts["<__future__>"] = {"kind": "module", "hash": _sha(",".join(future))}
    while pending:
        name = pending.pop()
        if name in parts or name not in index:
            continue
        kinds, dumps, references = [], [], set()
        for kind, stmt, alias in index[name]:
            if kind == "import":
                origin = "." * getattr(stmt, "level", 0) + (getattr(stmt, "module", None) or "")
                dumps.append(f"from {origin} import {alias.name} as {name}" if isinstance(stmt, ast.ImportFrom) else f"import {alias.name} as {name}")
                root = (getattr(stmt, "module", None) or alias.name).split(".")[0]
                if not getattr(stmt, "level", 0):
                    modules.add(root)
            elif kind == "assign" and name == SELECTOR and fold:
                kind = "selector"
                dumps.append(f"selected:{algorithm}")
            elif kind == "assign" and registry_entry(stmt) is not None:
                kind = "registry"
                module = normalise([ast.Expr(value=registry_entry(stmt))])
                dumps.append(f"{algorithm} -> {ast.dump(module)}")
                references |= names_in(module)
            else:
                module = normalise([stmt])
                dumps.append(ast.dump(module))
                references |= names_in(module)
            kinds.append(kind)
        parts[name] = {"kind": kinds[0] if len(set(kinds)) == 1 else "block", "hash": _sha("\n".join(dumps))}
        pending += sorted(references - {name})

    algorithms = None
    for entries in index.values():
        for kind, stmt, _ in entries:
            value = getattr(stmt, "value", None)
            if (
                kind == "assign"
                and isinstance(value, ast.Dict)
                and value.keys
                and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in value.keys)
                and all(isinstance(v, ast.Name) and v.id in index for v in value.values)
            ):
                algorithms = [k.value for k in value.keys]
    current = None
    for kind, stmt, _ in index.get(SELECTOR, []):
        if kind == "assign" and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            current = stmt.value.value

    fingerprint = _sha(json.dumps({k: v["hash"] for k, v in parts.items()}, sort_keys=True), 64)
    return CodeInfo(fingerprint, dict(sorted(parts.items())), line_index, modules, algorithms, current)


def variant_source(original: bytes, algorithm: str) -> bytes:
    """Return calibration.py with only the ALGORITHM_NAME line set to ``algorithm``.

    If the file already selects ``algorithm`` the original bytes are returned,
    so that run uses (and records the hash of) the exact file to be submitted.
    """

    text = original.decode("utf-8")
    info = analyse_source(text, algorithm)
    if info.current_algorithm == algorithm:
        return original
    pattern = re.compile(rf"^{SELECTOR}[ \t]*(?::[^=\r\n]+)?=[^\r\n]*", re.M)
    if len(pattern.findall(text)) != 1:
        raise ExperimentError(
            f"calibration.py must contain exactly one module-level '{SELECTOR} = \"...\"' line "
            "so the runner can select an algorithm"
        )
    replaced = pattern.sub(lambda _: f"{SELECTOR} = {json.dumps(algorithm)}", text)
    return replaced.encode("utf-8")


# --------------------------------------------------------------------------
# Behaviour probe: which candidates does the algorithm propose?
# --------------------------------------------------------------------------

PROBE_LANDSCAPES = ("smooth", "corner", "degenerate")


def _round(value: float) -> float:
    return float(f"{float(value):.9g}")


class _ProbeObjective:
    """A cheap stand-in for the organizer objective with the same interface and rules.

    It clips to bounds, rejects non-finite values, caches duplicates (whole-hour
    schedule values count as the same candidate), and enforces the budget.

    - smooth: optimum inside the bounds.
    - corner: optimum near the lower bounds and a large loss at the defaults,
      like the real buildings.
    - degenerate: only nine distinct candidates exist, so the algorithm's
      handling of duplicates and stalled batches is exercised.
    """

    def __init__(self, specs, budget: int, batch_size: int, landscape: str) -> None:
        import numpy as np
        from integrates_hackathon.objective import SCHEDULE_PARAMETERS

        self._np = np
        self.specs = tuple(specs)
        self.landscape = landscape
        self._budget = int(budget)
        self._batch_size = int(batch_size)
        self._count = 0
        self._cache: dict[tuple, float] = {}
        self.log: list = []
        size = len(self.specs)
        rng = np.random.default_rng([20260, PROBE_LANDSCAPES.index(landscape), size])
        self._lower = np.array([float(s.lower) for s in self.specs])
        self._span = np.array([float(s.upper) - float(s.lower) for s in self.specs])
        self._weights = rng.uniform(0.2, 1.0, size=(4, size))
        target = rng.uniform(0.25, 0.75, size) if landscape == "smooth" else rng.uniform(0.0, 0.08, size)
        self._hours = [s.name in SCHEDULE_PARAMETERS and "fraction" not in s.name for s in self.specs]
        self._eighths = [s.name in SCHEDULE_PARAMETERS and "fraction" in s.name for s in self.specs]
        self._measured = self._energy(target)

    def _energy(self, unit):
        if self.landscape == "degenerate":
            unit = self._np.round(unit[:2] * 2) / 2
            return 1.0 + self._weights[:, : len(unit)] @ unit
        return 1.0 + self._weights @ unit + 0.5 * (self._weights[:, :3] @ unit[:3]) ** 2

    def _key(self, clipped) -> tuple:
        if self.landscape == "degenerate":
            unit = (self._np.array(clipped) - self._lower) / self._span
            return tuple(float(v) for v in self._np.round(unit[:2] * 2))
        return tuple(
            round(v) if hours else round(v * 8) if eighths else round(v, 12)
            for v, hours, eighths in zip(clipped, self._hours, self._eighths)
        )

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    @property
    def remaining_budget(self) -> int:
        return max(0, self._budget - self._count)

    def __call__(self, vector) -> float:
        return self.evaluate_many([vector])[0]

    def evaluate_many(self, vectors) -> tuple[float, ...]:
        np = self._np
        prepared = []
        for vector in vectors:
            values = tuple(vector)
            if len(values) != len(self.specs):
                raise ValueError(f"Expected {len(self.specs)} values, received {len(values)}")
            for value in values:
                if isinstance(value, (bool, str)) or not np.isfinite(float(value)):
                    raise ValueError(f"Candidate value {value!r} must be a finite real number")
            clipped = tuple(spec.clip(float(v)) for spec, v in zip(self.specs, values))
            prepared.append((clipped, self._key(clipped)))
        unseen = {}
        for clipped, key in prepared:
            if key not in self._cache and key not in unseen:
                unseen[key] = clipped
        if self._count + len(unseen) > self._budget:
            raise RuntimeError(
                f"Simulation evaluation budget exhausted: requested {len(unseen)} "
                f"new candidates with {self.remaining_budget} remaining."
            )
        for key, clipped in unseen.items():
            unit = (np.array(clipped) - self._lower) / self._span
            energy = self._energy(unit)
            self._cache[key] = float(100.0 * np.abs(energy - self._measured).sum() / self._measured.sum())
            self._count += 1
        self.log.append([[_round(v) for v in clipped] for clipped, _ in prepared])
        return tuple(self._cache[key] for _, key in prepared)


def run_probe(calibrate, source_path: Path, code: CodeInfo, specs, budget: int, batch_size: int, seed: int) -> dict:
    """Run ``calibrate`` on the probe objectives and record what it does.

    Returns one signature per landscape (a hash of every candidate proposed, in
    order, and of the returned parameters) plus the top-level names of
    ``source_path`` whose code actually executed.
    """

    target = os.path.normcase(str(source_path))
    executed: set[str] = set()

    def tracer(frame, event, _arg):
        if event == "call" and os.path.normcase(frame.f_code.co_filename) == target:
            line = frame.f_code.co_firstlineno
            for first, last, names in code.line_index:
                if first <= line <= last:
                    executed.update(names)
                    break
        return None  # no line tracing: only function entries are needed

    signatures, errors, evaluations = {}, {}, 0
    for landscape in PROBE_LANDSCAPES:
        objective = _ProbeObjective(specs, budget, batch_size, landscape)
        previous = sys.gettrace()
        sys.settrace(tracer)
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = calibrate(objective, specs, budget, seed)
            outcome = sorted((str(k), _round(v)) for k, v in dict(result).items())
        except Exception as error:  # the failure itself is part of the behaviour
            outcome = ["error", type(error).__name__, str(error)[:300]]
            errors[landscape] = f"{type(error).__name__}: {error}"
        finally:
            sys.settrace(previous)
        signatures[landscape] = _sha(json.dumps([objective.log, outcome]), 32)
        evaluations += objective._count
    return {"signatures": signatures, "executed": sorted(executed), "evaluations": evaluations, "errors": errors}


# --------------------------------------------------------------------------
# Deciding whether a saved result is still valid
# --------------------------------------------------------------------------


def organizer_identity() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    from integrates_hackathon.baseline_cache import baseline_input_signature

    try:
        simulator = version("pybuildingenergy")
    except PackageNotFoundError:
        simulator = "unknown"
    return {"input_signature": baseline_input_signature(), "pybuildingenergy": simulator}


def environment_identity(modules: set[str]) -> dict:
    """Versions of Python and of the third-party packages the algorithm imports."""

    from importlib.metadata import PackageNotFoundError, packages_distributions, version

    distributions = packages_distributions()
    packages = {}
    for module in sorted(modules):
        for distribution in distributions.get(module, []):
            try:
                packages[distribution] = version(distribution)
            except PackageNotFoundError:
                pass
    return {"python": platform.python_version(), "packages": packages}


class Candidate:
    """The current code for one experiment: snapshot, fingerprint, and a lazy probe."""

    def __init__(self, experiment: Experiment, work_dir: Path) -> None:
        self.experiment = experiment
        original = CALIBRATION_PATH.read_bytes()
        self.original_sha256 = hashlib.sha256(original).hexdigest()
        self.source = variant_source(original, experiment.algorithm)
        self.code = analyse_source(self.source.decode("utf-8"), experiment.algorithm)
        if self.code.algorithms is not None and experiment.algorithm not in self.code.algorithms:
            raise ExperimentError(
                f"Unknown algorithm {experiment.algorithm!r}; calibration.py registers {self.code.algorithms}"
            )
        self.environment = environment_identity(self.code.modules)
        self.work_dir = work_dir
        self.snapshot_path = work_dir / "calibration.py"
        self._probe: dict | None = None

    def write_snapshot(self) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_bytes(self.source)
        return self.snapshot_path

    def probe(self) -> dict:
        if self._probe is None:
            from integrates_hackathon.challenge_inputs import parameter_specs
            from integrates_hackathon.submission_api import load_submission

            experiment = self.experiment
            submission = load_submission(self.write_snapshot())
            specs = parameter_specs(dict(experiment.parameters), announce=False)
            started = time.perf_counter()
            self._probe = run_probe(
                submission.calibrate,
                self.snapshot_path.resolve(),
                self.code,
                specs,
                experiment.evaluation_budget,
                experiment.parallel_workers,
                experiment.seed,
            )
            self._probe["seconds"] = round(time.perf_counter() - started, 1)
        return self._probe


def _complete(folder: Path) -> bool:
    return all((folder / name).is_file() for name in (*EXPORT_FILES, RECORD_FILE))


def decide(candidate: Candidate, folder: Path, organizer: dict, *, force: bool, strict: bool) -> tuple[bool, str, dict | None]:
    """Return (reuse, reason, saved record) for one experiment."""

    if force:
        return False, "--force", None
    if not folder.is_dir():
        return False, "no saved result", None
    if not _complete(folder):
        return False, "saved result is incomplete", None
    try:
        record = json.loads((folder / RECORD_FILE).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, "saved record is unreadable", None
    if record.get("schema_version") != RECORD_SCHEMA:
        return False, "saved by an older version of the runner", record
    if record.get("settings") != candidate.experiment.settings():
        return False, "saved settings differ", record
    if record.get("organizer") != organizer:
        return False, "organizer inputs or simulator changed", record

    saved_code = record.get("code", {})
    accepted = {saved_code.get("fingerprint"), *(item["fingerprint"] for item in saved_code.get("equivalent", []))}
    same_code = candidate.code.fingerprint in accepted
    same_environment = record.get("environment") == candidate.environment
    if same_code and same_environment:
        return True, "algorithm code unchanged", record

    saved_parts, parts = saved_code.get("parts", {}), candidate.code.parts
    changed = sorted(n for n in set(saved_parts) & set(parts) if saved_parts[n]["hash"] != parts[n]["hash"])
    added = sorted(set(parts) - set(saved_parts))
    removed = sorted(set(saved_parts) - set(parts))
    what = []
    if not same_code:
        what.append("code changed: " + ", ".join([*changed, *(f"+{n}" for n in added), *(f"-{n}" for n in removed)]))
    if not same_environment:
        what.append("package versions changed")
    what = "; ".join(what)
    if strict:
        return False, f"{what} (--strict)", record

    if not same_code:
        # Identical behaviour can only vouch for code the probe exercises.
        unverifiable = [n for n in changed if parts[n]["kind"] not in ("def", "class", "registry")]
        if unverifiable:
            return False, f"{what}; a changed constant or import cannot be verified ({', '.join(unverifiable)})", record
    probe = candidate.probe()
    if probe["signatures"] != record.get("probe", {}).get("signatures"):
        differing = [k for k, v in probe["signatures"].items() if v != record.get("probe", {}).get("signatures", {}).get(k)]
        return False, f"{what}; the algorithm now proposes different candidates (probe: {', '.join(differing)})", record
    if not same_code:
        unexecuted = [
            n for n in [*changed, *added]
            if parts[n]["kind"] in ("def", "class") and n not in probe["executed"]
        ]
        if unexecuted:
            return False, f"{what}; changed code was not reached by the probe ({', '.join(unexecuted)})", record
    return True, f"{what}; behaviour identical on the probe ({probe['evaluations']} evaluations)", record


def remember_equivalent(folder: Path, record: dict, candidate: Candidate) -> None:
    """Record that the current code was verified equivalent, so the next check is instant."""

    code = record.setdefault("code", {})
    known = {code.get("fingerprint"), *(item["fingerprint"] for item in code.get("equivalent", []))}
    changed = False
    if candidate.code.fingerprint not in known:
        code.setdefault("equivalent", []).append(
            {"fingerprint": candidate.code.fingerprint, "verified_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        )
        changed = True
    if record.get("environment") != candidate.environment:
        record["environment"] = candidate.environment
        changed = True
    if changed:
        temporary = folder / (RECORD_FILE + ".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, folder / RECORD_FILE)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _simulate_and_export(candidate: Candidate) -> tuple[dict, dict]:
    """Run the organizer workflow for one experiment; return (export paths, result summary)."""

    import integrates_hackathon.notebook_workflow as workflow

    experiment = candidate.experiment
    context = workflow.initialize_challenge(
        backend="pbe",
        building_ids=experiment.buildings,
        calibration_years=experiment.calibration_years,
        score_months=(2, 3),
        warmup_month=1,
        evaluation_budget=experiment.evaluation_budget,
        parallel_workers=experiment.parallel_workers,
        seed=experiment.seed,
        output_dir=candidate.work_dir,
    )
    try:
        if context.parallel_workers != experiment.parallel_workers:
            raise ExperimentError(
                f"the simulator would use {context.parallel_workers} workers instead of {experiment.parallel_workers}"
            )
        with _quiet():
            context = workflow.configure_parameter_contract(context, dict(experiment.parameters))
            baseline = workflow.load_baseline(context)
        started = time.perf_counter()
        run = workflow.run_calibration(context, submission_path=candidate.write_snapshot())
        runtime = time.perf_counter() - started
        with _quiet():
            report = workflow.score_and_visualize(context, baseline, run)
            import matplotlib.pyplot as plt

            plt.close("all")
        paths = workflow.export_standardized_results(context, run, report)
    finally:
        close = getattr(context.simulator, "close", None)
        if close is not None:
            close()
    result = {
        "overall_score": report.overall_score,
        "building_scores": {score.building_id: score.calibrated_score for score in report.building_scores},
        "runtime_seconds": round(runtime, 1),
    }
    return paths, result


def _store(candidate: Candidate, runs_dir: Path, paths: dict, record: dict) -> Path:
    """Move a finished run into outputs/runs/<run-id>/, keeping any older result."""

    final = runs_dir / candidate.experiment.run_id
    staging = runs_dir / f".staging_{candidate.experiment.run_id}_{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for path in paths.values():
        shutil.copy2(path, staging / Path(path).name)
    shutil.copy2(candidate.snapshot_path, staging / SNAPSHOT_FILE)
    (staging / RECORD_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    if final.exists():
        old = "unknown"
        with contextlib.suppress(Exception):
            old = json.loads((final / RECORD_FILE).read_text(encoding="utf-8"))["code"]["fingerprint"][:8]
        archive = runs_dir / "_superseded"
        archive.mkdir(exist_ok=True)
        destination = archive / f"{final.name}__{old}"
        suffix = 1
        while destination.exists():
            suffix += 1
            destination = archive / f"{final.name}__{old}_{suffix}"
        shutil.move(str(final), str(destination))
        print(f"  Previous result kept in {destination.relative_to(runs_dir.parent)}")
    os.replace(staging, final)
    return final


def run_experiment(experiment: Experiment, runs_dir: Path, organizer: dict, *, force: bool, strict: bool, dry_run: bool) -> dict:
    """Run or reuse one experiment. Returns a row for the final summary."""

    row = {"index": experiment.index, "run_id": experiment.run_id, "action": "", "score": None, "detail": ""}
    work_dir = runs_dir.parent / "work" / f"{experiment.run_id}__{os.getpid()}"
    try:
        candidate = Candidate(experiment, work_dir)
        folder = runs_dir / experiment.run_id
        reuse, reason, record = decide(candidate, folder, organizer, force=force, strict=strict)
        row["detail"] = reason
        if reuse:
            row["action"] = "saved"
            row["score"] = record.get("result", {}).get("overall_score")
            if not dry_run:
                remember_equivalent(folder, record, candidate)
            return row
        row["action"] = "needs run"
        if dry_run:
            return row
        if experiment.parallel_workers > available_cpus():
            raise ExperimentError(
                f"needs {experiment.parallel_workers} CPUs for parallel_workers, but only {available_cpus()} are "
                "available here. The worker count is the algorithm's batch size, so it is part of the experiment; "
                "run it on a larger machine or add an experiment with fewer workers."
            )
        print(f"  Running ({reason}).", flush=True)
        probe = candidate.probe()  # also a seconds-long smoke test before the long simulation
        if "smooth" in probe["errors"]:
            raise ExperimentError(f"calibrate() failed on the probe objective: {probe['errors']['smooth']}")
        for landscape, message in probe["errors"].items():
            print(f"  Warning: calibrate() failed on the '{landscape}' probe: {message}")
        paths, result = _simulate_and_export(candidate)
        new_record = {
            "schema_version": RECORD_SCHEMA,
            "run_id": experiment.run_id,
            "name": experiment.name,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "settings": experiment.settings(),
            "organizer": organizer,
            "environment": candidate.environment,
            "code": {
                "fingerprint": candidate.code.fingerprint,
                "parts": candidate.code.parts,
                "calibration_py_sha256": candidate.original_sha256,
                "snapshot_sha256": hashlib.sha256(candidate.source).hexdigest(),
                "equivalent": [],
            },
            "probe": {key: probe[key] for key in ("signatures", "executed", "evaluations")},
            "result": result,
        }
        final = _store(candidate, runs_dir, paths, new_record)
        row.update(action="ran", score=result["overall_score"], detail=f"{reason}; saved to {final.relative_to(runs_dir.parent)}")
    except ExperimentError as error:
        row.update(action="FAILED", detail=str(error))
    except Exception as error:
        traceback.print_exc()
        row.update(action="FAILED", detail=f"{type(error).__name__}: {error}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return row


def _select(experiments: list[Experiment], args) -> list[Experiment]:
    chosen = experiments
    if getattr(args, "index", None) is not None:
        if not 0 <= args.index < len(experiments):
            raise SystemExit(f"--index {args.index} is out of range; the settings define indices 0..{len(experiments) - 1}")
        chosen = [experiments[args.index]]
    for text in getattr(args, "only", None) or []:
        chosen = [e for e in chosen if text in e.run_id or text == e.name]
    if not chosen:
        raise SystemExit("No experiment matches the selection.")
    return chosen


def _print_rows(rows: list[dict]) -> None:
    width = max(len(row["run_id"]) for row in rows)
    print(f"\n{'#':>3}  {'experiment':<{width}}  {'status':<9}  {'score':>6}  detail")
    for row in rows:
        score = "" if row["score"] is None else f"{row['score']:.2f}"
        print(f"{row['index']:>3}  {row['run_id']:<{width}}  {row['action']:<9}  {score:>6}  {row['detail']}")


def command_run(args, *, dry_run: bool = False) -> int:
    experiments = _select(load_experiments(args.settings), args)
    organizer = organizer_identity()
    rows = []
    for experiment in experiments:
        print(f"[{experiment.index}] {experiment.run_id}", flush=True)
        row = run_experiment(
            experiment, args.runs_dir, organizer,
            force=getattr(args, "force", False), strict=args.strict, dry_run=dry_run,
        )
        print(f"  {row['action']}: {row['detail']}", flush=True)
        rows.append(row)
    _print_rows(rows)
    if dry_run:
        pending = sum(row["action"] == "needs run" for row in rows)
        print(f"\n{pending} of {len(rows)} experiment(s) would run. For a Slurm array use --array=0-{len(load_experiments(args.settings)) - 1}.")
    return 1 if any(row["action"] == "FAILED" for row in rows) else 0


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def load_run(folder: Path) -> dict | None:
    import pandas as pd

    if not all((folder / name).is_file() for name in EXPORT_FILES):
        return None
    read = lambda name: pd.read_csv(folder / name, encoding="utf-8-sig", float_precision="round_trip")
    record = {}
    with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
        record = json.loads((folder / RECORD_FILE).read_text(encoding="utf-8"))
    return {
        "name": record.get("name") or "-",  # runs saved by the notebook have no name
        "meta":json.loads((folder / "submission_result.json").read_text(encoding="utf-8")),
        "summary": read("summary.csv"),
        "monthly": read("monthly_results.csv"),
        "trace": read("evaluation_trace.csv"),
    }


def command_compare(args) -> int:
    import pandas as pd

    runs_dir: Path = args.runs_dir
    folders = sorted(d for d in runs_dir.glob("*") if d.is_dir() and not d.name.startswith(("_", ".")))
    if args.all or args.runs:
        selected = [d for d in folders if not args.runs or any(text in d.name for text in args.runs)]
    else:
        wanted = [experiment.run_id for experiment in load_experiments(args.settings)]
        missing = [run_id for run_id in wanted if not (runs_dir / run_id).is_dir()]
        if missing:
            print("Not run yet (skipped):", *missing, sep="\n  ")
        selected = [runs_dir / run_id for run_id in wanted if (runs_dir / run_id).is_dir()]
    runs = {folder.name: load_run(folder) for folder in selected}
    runs = {name: run for name, run in runs.items() if run is not None}
    if not runs:
        print(f"No saved runs to compare in {runs_dir}. Run experiments first, or use --all.")
        return 1

    scores = pd.DataFrame({name: run["summary"].set_index("building_id")["calibrated_score"] for name, run in runs.items()}).T
    scores["OVERALL"] = [run["meta"]["overall_score"] for run in runs.values()]
    scores.insert(0, "name", [run["name"] for run in runs.values()])
    scores.index.name = "run"
    scores = scores.sort_values("OVERALL", ascending=False)

    details, parameters = [], []
    for name, run in runs.items():
        final = run["trace"][run["trace"]["is_final_candidate"]].set_index("building_id")["evaluation_number"]
        for row in run["summary"].itertuples():
            details.append({
                "building_id": row.building_id,
                "run": name,
                "name": run["name"],
                "baseline_score": row.baseline_score,
                "calibrated_score": row.calibrated_score,
                "wape_pct": row.calibrated_weighted_absolute_percentage_error,
                "nmbe_pct": row.calibrated_nmbe_pct,
                "cv_rmse_pct": row.calibrated_cv_rmse_pct,
                "evaluations": row.evaluations,
                "best_found_at_eval": final.get(row.building_id),
                "runtime_min": row.runtime_seconds / 60,
            })
        for building, values in run["meta"]["parameters"].items():
            parameters.append({"building_id": building, "run": name, **values})
    details = pd.DataFrame(details).set_index(["building_id", "run"]).sort_index()
    parameters = pd.DataFrame(parameters).set_index(["building_id", "run"]).sort_index()

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print("\nCalibrated score (higher is better)\n" + scores.to_string(na_rep="-"))
        print("\nDetails\n" + details.to_string(na_rep="-"))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print("\nBest parameters found\n" + parameters.T.to_string(na_rep="-"))
    scopes = {(tuple(sorted(run["summary"]["building_id"])), tuple(sorted(run["monthly"]["year"].unique()))) for run in runs.values()}
    if len(scopes) > 1:
        print("\nNote: these runs cover different buildings or years, so their OVERALL scores are not directly comparable.")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv")
    details.to_csv(out / "details.csv")
    parameters.to_csv(out / "parameters.csv")
    written = ["scores.csv", "details.csv", "parameters.csv"]
    if not args.no_plots:
        written += _plots(runs, [d.name for d in folders], out)
    print(f"\nWrote {', '.join(written)} to {out}")
    return 0


def _plots(runs: dict, all_names: list[str], out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    import pandas as pd

    palette = plt.get_cmap("tab10").colors
    # One fixed color per run in both plots, stable when the selection changes.
    colors = {name: palette[i % len(palette)] for i, name in enumerate(sorted(set(all_names) | set(runs)))}
    reference = {"measured": "#222222", "baseline": "#b0b0b0"}
    buildings = sorted({b for run in runs.values() for b in run["summary"]["building_id"]})
    width = max(8, 5.5 * len(buildings))

    figure, axes = plt.subplots(1, len(buildings), figsize=(width, 4.2), squeeze=False)
    for axis, building in zip(axes[0], buildings):
        baseline_wape = None
        for name, run in runs.items():
            trace = run["trace"][run["trace"]["building_id"] == building].sort_values("evaluation_number")
            if trace.empty:
                continue
            axis.plot(trace["evaluation_number"], trace["loss_wape_pct"].cummin(), marker=".", label=name, color=colors[name])
            axis.scatter(trace["evaluation_number"], trace["loss_wape_pct"], s=8, alpha=0.25, color=colors[name])
            baseline_wape = run["summary"].set_index("building_id").loc[building, "baseline_weighted_absolute_percentage_error"]
        if baseline_wape is not None:
            axis.axhline(baseline_wape, color=reference["baseline"], linestyle="--", linewidth=1, label="baseline")
        axis.set(title=building, xlabel="evaluation #", ylabel="WAPE % (lower is better)", yscale="log")
        axis.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    figure.suptitle("Convergence: best-so-far WAPE (line) and every candidate (dots)")
    figure.tight_layout()
    figure.savefig(out / "convergence.png", dpi=130)

    figure, axes = plt.subplots(1, len(buildings), figsize=(width, 4.2), squeeze=False)
    for axis, building in zip(axes[0], buildings):
        frame = pd.DataFrame()
        for name, run in runs.items():
            monthly = run["monthly"][run["monthly"]["building_id"] == building].copy()
            if monthly.empty:
                continue
            monthly.index = monthly["year"].astype(str) + "-" + monthly["month"].map("{:02d}".format)
            both = monthly[["actual_kwh", "baseline_kwh"]].rename(columns={"actual_kwh": "measured", "baseline_kwh": "baseline"})
            frame = both if frame.empty else frame.combine_first(both)
            frame[name] = monthly["calibrated_kwh"]
        frame = frame[["measured", "baseline", *[c for c in frame.columns if c not in reference]]].sort_index()
        (frame / 1000).plot.bar(ax=axis, rot=0, width=0.8, color=[reference.get(c) or colors[c] for c in frame.columns])
        axis.set(title=building, ylabel="MWh", xlabel="")
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Monthly heating energy: measured vs. simulated")
    figure.tight_layout()
    figure.savefig(out / "monthly_energy.png", dpi=130)
    plt.close("all")
    return ["convergence.png", "monthly_energy.png"]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ("run", "status", "compare")
    if not argv or (argv[0] not in commands and argv[0] not in ("-h", "--help")):
        argv.insert(0, "run")  # running experiments is the default action

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS, help="JSON settings file (default: experiments.json)")
        p.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="where results are saved (default: outputs/runs)")

    def selection(p):
        p.add_argument("--index", type=int, help="run only the experiment with this index (for Slurm job arrays)")
        p.add_argument("--only", action="append", help="run only experiments whose run id contains this text, or whose name equals it")
        p.add_argument("--strict", action="store_true", help="rerun whenever the algorithm's code changed at all (skip the behaviour probe)")

    run = sub.add_parser("run", help="run and save the experiments that have no valid saved result (default)")
    common(run), selection(run)
    run.add_argument("--force", action="store_true", help="rerun even if a valid saved result exists")
    status = sub.add_parser("status", help="show which experiments are saved and which would run")
    common(status), selection(status)
    compare = sub.add_parser("compare", help="compare saved results; never simulates")
    common(compare)
    compare.add_argument("--all", action="store_true", help="compare every saved run instead of those in the settings file")
    compare.add_argument("--runs", action="append", help="compare saved runs whose folder name contains this text")
    compare.add_argument("--out", type=Path, default=REPO / "outputs" / "comparison", help="folder for tables and plots")
    compare.add_argument("--no-plots", action="store_true", help="write only the CSV tables")

    args = parser.parse_args(argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "status":
        return command_run(args, dry_run=True)
    return command_compare(args)


if __name__ == "__main__":  # required: simulation workers are spawned processes
    sys.exit(main())
