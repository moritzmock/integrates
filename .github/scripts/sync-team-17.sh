#!/usr/bin/env bash
set -euo pipefail

: "${HACKATHON_USERNAME:?Set the HACKATHON_USERNAME Actions secret}"
: "${HACKATHON_TOKEN:?Set the HACKATHON_TOKEN Actions secret (HTTPS password or access token)}"
: "${SYNC_SUBDIRECTORY:?Choose frontend or backend}"

# Restrict deletion and staging to one of the two managed folders.
case "$SYNC_SUBDIRECTORY" in
  frontend|backend) ;;
  *) printf 'Invalid sync folder: %s\n' "$SYNC_SUBDIRECTORY" >&2; exit 1 ;;
esac

source_root=$(git rev-parse --show-toplevel)
source_commit=$(git -C "$source_root" rev-parse HEAD)
source_repository=${GITHUB_REPOSITORY:-$(basename "$source_root")}
target_url=${HACKATHON_REPOSITORY_URL:-https://repos.hackathon.bz.it/2026-sfscon/team-17.git}
target_branch=main
sync_work_dir=$(mktemp -d)
trap 'rm -rf -- "$sync_work_dir"' EXIT

# Credentials stay in the environment, never in a remote URL or a Git config.
cat > "$sync_work_dir/askpass.sh" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' "$HACKATHON_USERNAME" ;;
  *Password*) printf '%s\n' "$HACKATHON_TOKEN" ;;
  *) exit 1 ;;
esac
ASKPASS
chmod 700 "$sync_work_dir/askpass.sh"
export GIT_ASKPASS="$sync_work_dir/askpass.sh"
export GIT_TERMINAL_PROMPT=0
# Disable inherited credential helpers, including helpers that save passwords.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=credential.helper GIT_CONFIG_VALUE_0=''

# Archive only the committed snapshot, excluding .git and untracked local files.
git -C "$source_root" archive --format=tar --output="$sync_work_dir/source.tar" "$source_commit"
git clone --quiet --depth=1 --single-branch --branch "$target_branch" "$target_url" "$sync_work_dir/target"
target_root="$sync_work_dir/target"
script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Draw once so a rejected push keeps the same identity on every retry.
sync_identity=$(python3 "$script_directory/sync_author.py")
IFS=$'\t' read -r sync_author_name sync_author_email <<< "$sync_identity"
git -C "$target_root" config user.name "$sync_author_name"
git -C "$target_root" config user.email "$sync_author_email"
printf 'Sync author: %s <%s>\n' "$sync_author_name" "$sync_author_email"

for attempt in 1 2 3 4 5; do
  # Both source repositories push here. Rebuild on the newest remote commit
  # after a race, so the other project and all root files remain intact.
  git -C "$target_root" fetch --quiet origin "$target_branch"
  git -C "$target_root" reset --hard --quiet FETCH_HEAD
  rm -rf -- "$target_root/$SYNC_SUBDIRECTORY"
  mkdir -p "$target_root/$SYNC_SUBDIRECTORY"
  tar -xf "$sync_work_dir/source.tar" -C "$target_root/$SYNC_SUBDIRECTORY"
  git -C "$target_root" add --force --all -- "$SYNC_SUBDIRECTORY"

  if git -C "$target_root" diff --cached --quiet; then
    printf '%s/ is already up to date.\n' "$SYNC_SUBDIRECTORY"
    exit 0
  fi

  git -C "$target_root" -c commit.gpgsign=false commit --quiet \
    -m "Sync $source_repository to $SYNC_SUBDIRECTORY ($source_commit)"
  if git -C "$target_root" push origin "HEAD:refs/heads/$target_branch"; then
    printf 'Synced %s at %s into %s/.\n' "$source_repository" "$source_commit" "$SYNC_SUBDIRECTORY"
    exit 0
  fi

  if [ "$attempt" -lt 5 ]; then
    printf 'Push failed; refreshing the target before retry %s/5.\n' "$((attempt + 1))"
    sleep "$attempt"
  fi
done

printf 'Could not push after five attempts. Check HTTPS credentials and branch permissions.\n' >&2
exit 1
