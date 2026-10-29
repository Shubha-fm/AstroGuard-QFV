# Publish the clean repository on GitHub

The accompanying `publish_to_github.sh` script replaces the *working tree* on `Shubha-fm/AstroGuard-QFV` with these curated files, and backs up the previous main commit to a dated branch first. It does not erase Git history or any historical commits.

Run from inside this extracted directory on your Mac:

```bash
bash publish_to_github.sh
```

Requirements: git installed and an authenticated GitHub account with **Contents: write** permission for the repository. On macOS, `gh auth login` can configure access when GitHub CLI is installed. Inspect `git status` in the prompted working copy and repository URL before uploading. The script creates/pushes a backup branch before replacing old files; it stops if backup cannot be pushed. It excludes raw data and model weights.

Current ChatGPT GitHub connector returned HTTP 403 (`Resource not accessible by integration`) on write actions. Therefore direct publishing via that connection is unavailable until permissions are changed. This local script uses your own Git credentials instead.
