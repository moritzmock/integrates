# Scheduled sync to team-17

The `Sync backend to team-17` GitHub Action publishes the latest committed
snapshot of this repository's default branch into `backend/` on the `main`
branch of https://repos.hackathon.bz.it/2026-sfscon/team-17.git.
The Hackathon2026 repository has a matching workflow that owns `frontend/`.

The frontend runs every 15 minutes at minutes 03, 18, 33, and 48. The backend
runs at minutes 08, 23, 38, and 53. GitHub may delay scheduled runs. Both
workflows can also be started from **Actions → Run workflow**.

## Credentials and activation

Each GitHub repository needs these **Settings → Secrets and variables →
Actions → Repository secrets**:

- `HACKATHON_USERNAME`: the hackathon server account's username.
- `HACKATHON_TOKEN`: an HTTPS password or access token that can read and push
  to team-17. Prefer an access token restricted to the target repository.

Commit and push the workflow and script to the GitHub default branch to
activate the schedule. Your Mac can be switched off; the runner connects to
the remote HTTPS repository directly.

## Sync behavior

- Only committed source files are copied; `.git` and untracked files are excluded.
- Source deletions are applied inside this project's folder.
- The other project and the root README/license are preserved.
- Unchanged snapshots do not create commits.
- Pushes use normal fast-forward updates. A rejected push is retried on the
  latest target branch, up to five attempts, to handle simultaneous syncs.
- Destination edits inside `backend/` are replaced by the next source sync.
  Make project changes in GitHub and commit/push them there first.
