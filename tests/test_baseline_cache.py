import json
import tempfile
import unittest
from pathlib import Path

from integrates_hackathon.baseline_cache import (
    _signature_bytes,
    load_precomputed_baseline,
    save_precomputed_baseline,
)
from integrates_hackathon.challenge_inputs import BUILDINGS
from integrates_hackathon.models import MonthlyConsumption
from integrates_hackathon.notebook_workflow import initialize_challenge


class BaselineCacheTests(unittest.TestCase):
    def _example_baseline(self):
        return {
            case.building_id: tuple(
                MonthlyConsumption(row.year, row.month, float(index + 1))
                for index, row in enumerate(case.measurements)
            )
            for case in BUILDINGS
        }

    def test_round_trip_and_signature_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "baseline.csv"
            metadata = Path(directory) / "baseline.json"
            expected = self._example_baseline()
            save_precomputed_baseline(
                BUILDINGS, expected, cache_path=cache, metadata_path=metadata
            )

            loaded = load_precomputed_baseline(
                BUILDINGS, cache_path=cache, metadata_path=metadata
            )
            self.assertEqual(loaded, expected)

            details = json.loads(metadata.read_text(encoding="utf-8"))
            details["input_signature"] = "stale"
            metadata.write_text(json.dumps(details), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                load_precomputed_baseline(
                    BUILDINGS, cache_path=cache, metadata_path=metadata
                )

    def test_package_version_drift_warns(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "baseline.csv"
            metadata = Path(directory) / "baseline.json"
            save_precomputed_baseline(
                BUILDINGS, self._example_baseline(),
                cache_path=cache, metadata_path=metadata,
            )
            details = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertIn("numpy", details["package_versions"])
            details["package_versions"]["numpy"] = "0.0.1"
            metadata.write_text(json.dumps(details), encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, r"numpy .*baseline used 0\.0\.1"):
                load_precomputed_baseline(
                    BUILDINGS, cache_path=cache, metadata_path=metadata
                )

    def test_signature_ignores_text_line_endings_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in (".py", ".epw"):
                lf = root / f"lf{suffix}"
                crlf = root / f"crlf{suffix}"
                lf.write_bytes(b"a,1\nb,2\n")
                crlf.write_bytes(b"a,1\r\nb,2\r\n")
                self.assertEqual(_signature_bytes(lf), _signature_bytes(crlf))
            binary = root / "book.xlsx"
            binary.write_bytes(b"PK\r\n\x00")
            self.assertEqual(_signature_bytes(binary), b"PK\r\n\x00")

    def test_shipped_cache_matches_current_inputs(self):
        context = initialize_challenge(backend="demo")
        loaded = load_precomputed_baseline(context.cases)
        self.assertEqual(set(loaded), {case.building_id for case in context.cases})


if __name__ == "__main__":
    unittest.main()
