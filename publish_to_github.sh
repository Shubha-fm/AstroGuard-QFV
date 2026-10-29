#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
REMOTE="https://github.com/Shubha-fm/AstroGuard-QFV.git"
if ! command -v git >/dev/null 2>&1; then echo "Install git first" >&2; exit 1; fi
# Run in an isolated temporary clone. Never remove the local source project.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/astroguard-github.XXXXXX")"
trap 'echo "Temporary work copy retained at: '"'$WORK'"'"' EXIT
printf 'Repository: %s
' "$REMOTE"
git clone --branch main "$REMOTE" "$WORK/repo"
cd "$WORK/repo"
OLD_SHA="$(git rev-parse HEAD)"
BACKUP="backup-before-18k-20261029"
if git ls-remote --exit-code --heads origin "$BACKUP" >/dev/null 2>&1; then
  echo "Backup branch already exists: $BACKUP"
else
  git branch "$BACKUP" "$OLD_SHA"
  git push origin "$BACKUP" || { echo "Backup push failed. Main was not modified." >&2; exit 1; }
fi
# Only overwrite a clean repository cloned freshly for this operation.
rsync -a --delete --exclude='.git/' --exclude='publish_to_github.sh'   --exclude='*.before_filename_update' "$HERE/" "$WORK/repo/"
# Keep the publishing script in version control as a reproducible utility.
cp "$HERE/publish_to_github.sh" "$WORK/repo/publish_to_github.sh"
chmod +x "$WORK/repo/publish_to_github.sh" "$WORK/repo/mac_run.sh"
git add -A
printf '
Changes to commit:
'
git status --short
# Protect against large files accidentally entering Git
python3 - <<'PYCHECK'
from pathlib import Path
bad=[]
for path in Path('.').rglob('*'):
    if path.is_file() and '.git' not in path.parts and path.stat().st_size>90_000_000:
        bad.append(str(path))
if bad: raise SystemExit('Oversized files; refusing commit: '+', '.join(bad))
PYCHECK
if git diff --cached --quiet; then echo 'No changes detected'; exit 0; fi
git commit -m 'Replace legacy prototype with reproducible 18K AstroGuard-QFV experiment'
# Non-force push. If another contributor changed main, the push will safely fail.
git push origin main
printf '
SUCCESS: https://github.com/Shubha-fm/AstroGuard-QFV
'
printf 'Backup branch: %s
' "$BACKUP"
