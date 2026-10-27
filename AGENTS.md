# Maintaining the public YSZ RL Lab

- This is the public publication workspace. Preserve the independent research workspace and its evidence.
- User scope: only material and learning actually used in this YSZ project. Do not invent courses, implemented methods, personal mastery, experiment success, or retrospective Git commits.
- Daily publication is authorized at 20:00 America/Los_Angeles. Read local files only, export allowlisted material, refresh results, verify, commit and push meaningful changes. Never start research compute or query Slurm as part of publication.
- Source path is in the git-ignored `.local-source.json`. It must never be staged or printed in public docs.
- Historical files under `history/` and `PROJECT_RECORD_RSI_YSZ.md` are sanitized source snapshots. Keep their history, and put current interpretation in curated docs.
- Model files, credentials, cluster/user identifiers, third-party paper PDFs, VASP potentials, raw force logs and private message stores are outside the publication scope.
- Run `python tools/publish_snapshot.py`, `python tools/results.py --write`, `python tools/check_publication.py` and `python -m unittest discover -s tests -v` before committing an update. Review the diff and the actual result status. Keep learning claims grounded in user statements.
- No new file enters the snapshot allowlist without review. Do not bypass a failed publication check.
- On failure or a block needing user input, remove the corresponding daily automation as requested by the user; preserve the error and local changes for recovery.
- No empty status messages or empty commits when the monitored state has not changed.
