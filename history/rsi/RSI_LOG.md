# RSI on YSZ: build log

## 2026-10-28: judge, sandbox, improver runner, orchestrator (built, not yet piloted)
- **Judge** `rsi/judge/rsi_judge.py`: runs a search program S through formal12's menu-baseline path, with only the menu choice swapped to S. Copied to the cluster as `formal12/rsi_judge/`.
  - Check job 997 (synthetic physics): S0 matches the frozen greedy arm byte for byte (64 decisions, physics, starts, best energies).
  - An invalid program is reported as INVALID, both when it is caught at load time and when it fails mid-run. No traceback leaks.
- **Sandbox** `rsi/judge/rsi_sandbox.py` (4fb9f398): static AST rules, restricted builtins and imports, a runtime audit hook and a 5 s timeout.
  - Locally on Py3.12/np1.26 and Py3.10/np2.2.6: 32/32 sandbox tests and 4/4 audit tests pass.
  - The cluster re-check is in job 998.
  - The cluster's first false positive (numpy's lazy C-level import going through the candidate's `__import__`) is fixed.
- **Improver runner**: headless Claude Code, `claude-sonnet-5-5`, with `--safe-mode --restricted --strict-mcp-config --disable-slash-commands --no-session-persistence`, `--permission-mode acceptEdits --permission-prompts none`, a fixed system prompt, and Bash allowed only for the static checker.
  - Isolation tested twice (dontAsk and acceptEdits modes). Nothing outside the workspace can be read, listed, grepped or written. Compound commands, `$()` and backticks are refused. No MCP, no skills, no auto-memory.
  - The checker `rsi/improver/check_program.py` is static only and never executes M's code.
- **Orchestrator** `rsi/improver/orchestrator.py`, the fixed outer loop:
  - Commands: `init`, `phase A|B` (each a fresh isolated session), `evaluate` (fake or cluster), `collect` (one-shot sacct + fetch; it never polls and never computes on the login node) and `select` (only an evaluated, valid program can become S_{t+1}).
  - Provenance: improver/ hash at the start and end of every phase, cost, model and prompt hash.
  - Fixed arm: improver/ is restored to M0 after every phase. Recursive arm: edits are kept only from phase B.
- **Pipeline test with the fake judge** (`rsi/lineages/test_fixed`, `test_rec`; pseudo-scores, not physics):
  - Both arms ran gen 0, and the recursive arm also ran gen 1 phase A. Cost so far $0.27 for the improver, plus about $0.15 for the isolation tests.
  - Findings, both fixed:
    1. The recursive M chose not to revise improver/; it only wrote a suggestion in notes. Its phase-B prompt now *requires* a self-review: revise improver/, or record in notes why no revision is warranted.
    2. M tried to write files through shell heredocs (refused), and one M then concluded that the checker was unavailable too. The README now states the tool rules explicitly.
- **Before the pilot:**
  - Job 998: sandbox re-check on the cluster, plus the real cost of S0 on 8 starts with real physics. Set the screen8/full24 Slurm limits from it.
  - Freeze the judge (`rsi_judge/JUDGE.sha256`).
  - Fix the S0 development scores (val8 from 998, val24 = 2.561 from job 969).
  - Primary endpoint and the sealed-test seed commitment (sha256 now; seed kept outside every workspace).
  - Remove or rename the test lineages so the pilot starts clean.

## 2026-10-28: drive loop, pilot preparation
- User chose option B: the driver (`orchestrator.py drive`) may query `sacct` once per lineage every ≥900 s and fetch results with scp. The code enforces `--poll >= 900`. Nothing else runs on the login node.
- `drive` interleaves generations over the listed lineages: all phase A → submit all evaluations (singleton) → wait → all phase B → select.
  - Caps: improver total ≤ $50; GPU used + queued hard caps + worst case of the next submission ≤ 8 card-h.
  - It is resumable from provenance.
- Fake-judge test of `drive` (`_tests/td_*`, $0.21): one generation over a fixed and a recursive lineage.
  - With the required self-review, the recursive M **rewrote its procedure (M0 → M1)**: paired per-start comparison against the incumbent; a results ledger in notes; a budget rule to confirm screened winners and to re-measure the incumbent as a noise gauge; prefer information the incumbent ignores.
  - That is an L3 change (how it improves), not an L1 change.
  - M1 asks for a full24 confirmation (3 units), which the pilot budget of 2 cannot pay for. The orchestrator marks it "over budget, not evaluated" and M sees that in the archive. The pilot keeps budget 2, so that GPU stays within the 8 card-h cap.
- Test lineages moved to `rsi/lineages/_tests/`. Improver spend so far ≈ $0.63 including isolation tests.
- **Sealed test seed commitment:** sha256(seed) = `056bd3460c1a26818ff28e35d1ed9d37c3f9ec0d2a7967f6fe9a533b8eadc6dc`. The seed is in `SEALED_SEED_FILE` (mode 600), outside every workspace and not on the cluster. It is used only after every lineage's final program and improver are frozen.
- **Pilot plan:**
  - Lineages: p1_fixed, p1_recursive, p2_fixed, p2_recursive. Budget 2 units/generation, 2 generations, model claude-sonnet-5-5, S0 as the common start.
  - Start after job 998 confirms the cluster sandbox and gives S0's val8 score and real cost.
  - It is an engineering pilot: cost per generation, failure modes, whether recursive edits happen and take effect. It is not evidence for or against RSI.
  - Primary endpoint for the formal run: decide before it starts.

## 2026-10-28: job 998 and pilot launch
- **998** (rsi_check_2):
  - S0 run through the judge with real MACE physics on the GPU reproduces job 969's first 8 groups: per-start drops differ by ≤ 7e-12 eV, V(screen8) = 2.4925.
  - On the cluster's Python 3.10 / numpy 2.2.6, the fixed sandbox passes all sandbox tests and 4/4 audit tests. The program that 997 wrongly rejected now runs.
  - Real cost of one screen8 evaluation: 12 min 24 s (0.21 card-h).
- **Judge frozen:** `formal12/rsi_judge/JUDGE.sha256`, with rsi_judge.py 185bdd0b and rsi_sandbox.py 4fb9f398. Every evaluation job checks it, together with SOURCES.sha256.
- **Time limits from 998:** screen8 25 min with a 0.42 card-h hard cap; full24 75 min with a 1.25 card-h hard cap. Pilot worst case: 4 × 2 × 0.84 = 6.7 ≤ 8 card-h.
- **S0 development baseline:** `rsi/judge/s0_baseline_val8_val24.json`, copied into each workspace as `archive/s_current_gen0.json`. It holds per-start drops on screen8 (from 998) and full24 (from 969).
- **Pilot launched:** `drive --lineages p1_fixed,p1_recursive,p2_fixed,p2_recursive --gens 2 --poll 900`, with the log in `rsi/lineages/pilot_drive.log`.
  - Evaluation jobs use singleton on the shared job name, so they queue behind grid jobs 993–996. One GPU at a time.
