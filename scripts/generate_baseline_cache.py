"""Generate the shared exact-window baseline with local process workers."""

from __future__ import annotations

from time import perf_counter

from integrates_hackathon.baseline_cache import save_precomputed_baseline
from integrates_hackathon.notebook_workflow import initialize_challenge


def main() -> None:
    started = perf_counter()
    context = initialize_challenge(
        backend="pbe",
        building_ids=("building_01", "building_02", "building_03"),
        calibration_years=(2018, 2019),
        score_months=(2, 3),
        warmup_month=1,
        parallel_workers=4,
    )
    baseline = {}
    try:
        for index, case in enumerate(context.cases, start=1):
            context.simulator.set_calibration_progress(1, 1)
            print(
                f"[{index}/{len(context.cases)}] Simulating {case.building_id}",
                flush=True,
            )
            baseline[case.building_id] = context.simulator.simulate(
                case, case.defaults
            )
            total = sum(
                row.consumption_kwh for row in baseline[case.building_id]
            )
            print(f"{case.building_id}: {total:,.3f} kWh", flush=True)
    finally:
        context.simulator.close()

    cache_path, metadata_path = save_precomputed_baseline(
        context.cases, baseline
    )
    print(f"Saved {cache_path}", flush=True)
    print(f"Saved {metadata_path}", flush=True)
    print(
        f"Completed in {(perf_counter() - started) / 60.0:.1f} minutes",
        flush=True,
    )


if __name__ == "__main__":
    main()
