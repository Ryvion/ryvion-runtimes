# Codex Agent Instructions — Ryvion Runtimes

## Project Context

This repo is dormant for the Sovereign Relay pilot. The active llama.cpp
inference path lives in `ryvion-node`, not in this repo.

Current active runners: none.

## Container Contract

Future containers, if explicitly introduced for vetted document or embedding
workloads, must:

1. Read `/work/job.json`.
2. Write `/work/receipt.json` with output hash and metadata.
3. Write `/work/metrics.json` with `output_name`.
4. Write output artifacts under `/work/output`.

## Network Policy

- Runners should assume `--network=none`.
- The node is responsible for staging inputs into `/work` before execution.

## Key Rules

- Do not restore render, media, transcode, Chronomarket, or market simulation
  images to the active build matrix.
- Do not import from `ryvion-archive`.
- Keep runners workload-specific and small.
- Do not add arbitrary external manifest execution or browser-agent behavior.
- Commits: Keep messages SHORT, no Co-Authored-By.
