# Codex Agent Instructions — Ryvion Runtimes

## Project Context

Trusted managed OCI runners for workload-specific jobs. The active llama.cpp
inference path lives in `ryvion-node`, not in this repo.

Current active runner:

- `market-sim-runner/` — deterministic Chronomarket paper-trading replay for
  autonomous market agents. It runs staged market/news manifests with no network
  access and emits leaderboard-safe metrics plus receipt metadata.

## Container Contract

Future containers must:

1. Read `/work/job.json`.
2. Write `/work/receipt.json` with output hash and metadata.
3. Write `/work/metrics.json` with `output_name`.
4. Write output artifacts under `/work/output`.

## Network Policy

- Runners should assume `--network=none`.
- The node is responsible for staging inputs into `/work` before execution.

## Key Rules

- Do not restore render, media, or transcode images to the active build matrix.
- Do not import from `ryvion-archive`.
- Keep runners workload-specific and small.
- Do not let market agents browse live web data inside replay containers; news
  and market data must be staged through time-gated manifests.
- Commits: Keep messages SHORT, no Co-Authored-By.
