# Fresh macOS experiment: AstroGuard-QFV 18K

**This starts from zero. No files from the older Windows experiments are needed.**
No raw ZTF observations, previously trained models, or predicted results are included.

Read `README_MAC_FRESH.md` for the detailed instructions.

```bash
cd ~/Downloads/AstroGuard_18K_macOS_Fresh_Start
bash mac_run.sh setup
bash mac_run.sh query 100 40
```

Repeat the query until `PAGES_READY`. Then repeat `bash mac_run.sh collect 1000`
until `READY_FOR_PREPARE`. Completed downloads and queries are checkpointed.
`bash mac_run.sh help` lists all available commands.
