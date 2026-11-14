"""Check equal-weight team selection and stable identity across sync retries."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import sync_author


GIT = shutil.which("git")
SCRIPT_DIRECTORY = Path(__file__).resolve().parent


class SyncAuthorTests(unittest.TestCase):
    def test_four_requested_members_have_one_entry_each(self):
        names = [name for name, _ in sync_author.TEAM_MEMBERS]
        self.assertCountEqual(names, [
            "moritz-mock", "daniel-planotscher", "jonas-rabensteiner", "thomas-borsani",
        ])
        self.assertEqual(len(names), len(set(names)))
        for _, email in sync_author.TEAM_MEMBERS:
            self.assertIn("@", email)
            self.assertFalse(any(character.isspace() for character in email))

    def test_one_random_draw_keeps_each_name_with_its_own_email(self):
        for member in sync_author.TEAM_MEMBERS:
            with self.subTest(member=member), patch.object(
                sync_author.secrets, "choice", return_value=member,
            ) as draw:
                self.assertEqual(sync_author.choose_author(), member)
                draw.assert_called_once_with(sync_author.TEAM_MEMBERS)

    def test_cli_prints_one_tab_separated_identity(self):
        with patch.object(sync_author.secrets, "choice", return_value=sync_author.TEAM_MEMBERS[1]):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                sync_author.main()
        self.assertEqual(output.getvalue(), "\t".join(sync_author.TEAM_MEMBERS[1]) + "\n")

    def test_sync_retains_selected_identity_after_a_competing_push(self):
        def run(*args, cwd=None, env=None):
            result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout.strip()

        def git(path, *args):
            return run(GIT, *args, cwd=path)

        def configure(path):
            git(path, "config", "user.name", "Fixture Author")
            git(path, "config", "user.email", "fixture@example.com")

        def commit(path, message):
            git(path, "add", "--all")
            git(path, "-c", "commit.gpgsign=false", "commit", "-m", message)

        with tempfile.TemporaryDirectory(prefix="sync-author-test-") as temp:
            root = Path(temp)
            remote = root / "team.git"
            run(GIT, "init", "--bare", "--initial-branch=main", str(remote))
            seed = root / "seed"
            run(GIT, "init", "--initial-branch=main", str(seed))
            configure(seed)
            (seed / "README.md").write_text("team root")
            commit(seed, "initial team")
            git(seed, "remote", "add", "origin", str(remote))
            git(seed, "push", "origin", "main")

            source = root / "source"
            run(GIT, "init", "--initial-branch=main", str(source))
            configure(source)
            (source / "app.txt").write_text("source snapshot")
            commit(source, "source content")

            racer = root / "racer"
            run(GIT, "clone", str(remote), str(racer))
            configure(racer)
            (racer / "concurrent.txt").write_text("preserve competing update")
            commit(racer, "concurrent update")

            first_identity = root / "first-identity.json"
            draw_count = root / "draw-count"
            wrappers = root / "bin"
            wrappers.mkdir()
            python_wrapper = wrappers / "python3"
            python_wrapper.write_text(
                f"#!{sys.executable}\n"
                "import os, runpy, secrets, sys\n"
                "from pathlib import Path\n"
                "if len(sys.argv) > 1 and sys.argv[1].endswith('/sync_author.py'):\n"
                f"    counter = Path({str(draw_count)!r})\n"
                "    counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')\n"
                "    secrets.choice = lambda population: population[-1]\n"
                "    runpy.run_path(sys.argv[1], run_name='__main__')\n"
                "else:\n"
                f"    os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
            )
            python_wrapper.chmod(0o700)
            wrapper = wrappers / "git"
            wrapper.write_text(
                f"#!{sys.executable}\n"
                "import json, os, subprocess, sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                f"marker = Path({str(first_identity)!r})\n"
                "if len(args) > 2 and args[0] == '-C' and args[2] == 'push' and not marker.exists():\n"
                f"    identity = subprocess.check_output([{GIT!r}, '-C', args[1], 'show', '-s', '--format=%an%n%ae%n%cn%n%ce', 'HEAD'], text=True).splitlines()\n"
                "    marker.write_text(json.dumps(identity))\n"
                f"    subprocess.run([{GIT!r}, '-C', {str(racer)!r}, 'push', 'origin', 'main'], check=True)\n"
                f"os.execv({GIT!r}, [{GIT!r}, *args])\n"
            )
            wrapper.chmod(0o700)
            environment = dict(
                os.environ,
                PATH=str(wrappers) + os.pathsep + os.environ["PATH"],
                HACKATHON_USERNAME="fixture",
                HACKATHON_TOKEN="fixture-not-a-real-credential",
                HACKATHON_REPOSITORY_URL=str(remote),
                SYNC_SUBDIRECTORY="frontend",
                GITHUB_REPOSITORY="fixture/source",
            )
            output = run("bash", str(SCRIPT_DIRECTORY / "sync-team-17.sh"), cwd=source, env=environment)
            self.assertIn("retry 2/5", output)
            first = json.loads(first_identity.read_text())
            final = git(remote, "show", "-s", "--format=%an%n%ae%n%cn%n%ce", "main").splitlines()
            self.assertEqual(final, first)
            self.assertEqual(tuple(final[:2]), sync_author.TEAM_MEMBERS[-1])
            self.assertEqual(draw_count.read_text(), "1")
            self.assertEqual(final[:2], final[2:])
            self.assertEqual(git(remote, "show", "main:README.md"), "team root")
            self.assertEqual(git(remote, "show", "main:concurrent.txt"), "preserve competing update")
            self.assertEqual(git(remote, "show", "main:frontend/app.txt"), "source snapshot")


if __name__ == "__main__":
    unittest.main()
