from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from integrates_hackathon.submission_api import load_submission


class SubmissionApiTests(unittest.TestCase):
    def test_hash_identifies_exact_calibration_source_bytes(self) -> None:
        source = (
            b'TEAM_NAME = "Hash Test"\n'
            b'ALGORITHM_NAME = "Test"\n'
            b'def calibrate(objective, parameter_space, budget, seed):\n'
            b'    return {}\n'
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.py"
            path.write_bytes(source)

            submission = load_submission(path)

        self.assertEqual(submission.source_sha256, sha256(source).hexdigest())
        self.assertEqual(submission.team_name, "Hash Test")


if __name__ == "__main__":
    unittest.main()
