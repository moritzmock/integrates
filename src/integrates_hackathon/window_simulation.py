"""Process-parallel January-warm-up, February/March simulations."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
import os
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .models import BuildingCase, MonthlyConsumption
from .epw_io import read_epw_table, write_epw_table
from .simulation_workers import ParallelYearTask, simulate_year_worker


@dataclass(frozen=True)
class WindowEpw:
    year: int
    path: Path
    months: tuple[int, ...]
    hours: int
    warmup_hours: int


def create_window_epw(
    source_path: str | Path,
    *,
    year: int,
    output_dir: str | Path,
    warmup_month: int = 1,
    score_months: Sequence[int] = (2, 3),
) -> WindowEpw:
    """Create a contiguous EPW containing the warm-up and scored months."""

    months = (int(warmup_month), *(int(value) for value in score_months))
    if months != tuple(range(months[0], months[-1] + 1)):
        raise ValueError("Warm-up and scored months must form one contiguous window")
    header, table = read_epw_table(Path(source_path))
    window = table[table["Month"].astype(int).isin(months)].copy()
    warmup_hours = int((window["Month"].astype(int) == warmup_month).sum())
    if warmup_hours == 0 or window.empty:
        raise ValueError(f"EPW does not contain the requested months {months}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{Path(source_path).stem}_months_{months[0]:02d}-{months[-1]:02d}.epw"
    write_epw_table(header, window, path)
    return WindowEpw(int(year), path, months, len(window), warmup_hours)


class ParallelWindowedPyBuildingEnergySimulator:
    """Run independent candidate/year simulations in a local process pool."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        warmup_month: int = 1,
        score_months: Sequence[int] = (2, 3),
        max_workers: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.warmup_month = int(warmup_month)
        self.score_months = tuple(int(value) for value in score_months)
        available = os.cpu_count() or 1
        self.max_workers = max(1, min(int(max_workers), available))
        self._progress_batch: tuple[tuple[int, int], ...] = ((1, 1),)
        self._windows: dict[tuple[str, int], WindowEpw] = {}
        self._executor: ProcessPoolExecutor | None = None

    def set_calibration_progress(self, index: int, total: int) -> None:
        self._progress_batch = ((int(index), int(total)),)

    def set_calibration_batch_progress(
        self, indices: Sequence[int], total: int
    ) -> None:
        self._progress_batch = tuple((int(index), int(total)) for index in indices)

    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            # Spawned workers inherit these limits. Without them, every worker
            # may ask BLAS/OpenMP to use all laptop cores at the same time.
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ.setdefault(name, "1")
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=get_context("spawn"),
            )
        return self._executor

    def _window_for(self, year: int, source: str | Path) -> WindowEpw:
        key = (str(Path(source).resolve()), int(year))
        if key not in self._windows:
            # Only the parent process writes this derived file. All child
            # processes receive the completed path as read-only input.
            self._windows[key] = create_window_epw(
                source,
                year=int(year),
                output_dir=self.output_dir / "window_epw",
                warmup_month=self.warmup_month,
                score_months=self.score_months,
            )
        return self._windows[key]

    def simulate(
        self, case: BuildingCase, parameters: Mapping[str, float]
    ) -> tuple[MonthlyConsumption, ...]:
        return self.simulate_many(case, (parameters,))[0]

    def simulate_many(
        self,
        case: BuildingCase,
        parameter_sets: Sequence[Mapping[str, float]],
    ) -> tuple[tuple[MonthlyConsumption, ...], ...]:
        """Evaluate a batch while preserving candidate and period order."""

        candidates = tuple(dict(values) for values in parameter_sets)
        if not candidates:
            return ()
        years = tuple(sorted(case.weather_files.items()))
        windows = {
            int(year): self._window_for(int(year), source) for year, source in years
        }
        progress = self._progress_batch
        if len(progress) != len(candidates):
            progress = tuple((index + 1, len(candidates)) for index in range(len(candidates)))

        tasks: list[ParallelYearTask] = []
        for candidate_index, parameters in enumerate(candidates):
            progress_index, progress_total = progress[candidate_index]
            for year, _source in years:
                window = windows[int(year)]
                for month in self.score_months:
                    print(
                        f"Now running {case.building_id}, "
                        f"{pd.Timestamp(2001, month, 1).month_name()} {year}, "
                        f"calibration set {progress_index} of {progress_total}",
                        flush=True,
                    )
                tasks.append(
                    ParallelYearTask(
                        case=case,
                        parameters=parameters,
                        year=int(year),
                        window_epw_path=window.path,
                        window_hours=window.hours,
                        warmup_hours=window.warmup_hours,
                        score_months=self.score_months,
                    )
                )

        if self.max_workers == 1:
            year_results = tuple(simulate_year_worker(task) for task in tasks)
        else:
            year_results = tuple(self._pool().map(simulate_year_worker, tasks, chunksize=1))

        combined: list[tuple[MonthlyConsumption, ...]] = []
        year_count = len(years)
        for candidate_index in range(len(candidates)):
            start = candidate_index * year_count
            rows = tuple(
                row
                for result in year_results[start : start + year_count]
                for row in result
            )
            combined.append(tuple(sorted(rows)))
        return tuple(combined)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
