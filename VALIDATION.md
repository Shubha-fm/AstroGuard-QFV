# Package validation

- Checked `mac_run.sh` shell syntax with `bash -n`.
- Ran `pytest tests/test_18k.py tests/test_mac_fresh.py`: **4 passed**.
- Tests cover fresh candidate-page initialization and resumption, balanced frozen cohorts, 65/15/20 splits, and incremental queue expansion after larger page queries.
- These tests use fake broker responses and synthetic observations strictly for software validation, **not** astronomical research findings.
- Not executed against live ALeRCE or on macOS hardware inside this environment.
- PyTorch/PennyLane installation and runtime on the user's specific Mac must still be checked locally.
- No previously downloaded light curves, model predictions or measured astrophysical results are included.
