# Codex Agent Instructions — Ryvion Runtimes

## Project Context

Container images and runtime adapters for Ryvion render-farm workloads. Built by GitHub Actions, pushed to `ghcr.io/ryvion/*`.

## Container Contract

Active containers:
1. Read `/work/job.json`
2. Write `/work/receipt.json` (output hash + metadata)
3. Write `/work/metrics.json` (must include `output_name`)
4. Write output artifacts under `/work/`

## Network Policy

- Runners should assume `--network=none`.
- The node is responsible for staging inputs into `/work` before execution.

## Key Rules

- Keep runners workload-specific and small.
- Do not add AI inference, agent hosting, model training, or speculative verifier code here.
- Archive inactive runners instead of leaving them in the active build matrix.
- `metrics.json` must have `output_name` field
- Commits: Keep messages SHORT, no Co-Authored-By
