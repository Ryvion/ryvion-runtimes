# Runners

This repo currently has no active container runners. It is dormant for the
Sovereign Relay pilot.

The native llama.cpp execution path is implemented in `ryvion-node`.

## CI

`.github/workflows/build.yml` verifies the repository has not accidentally
reintroduced obsolete render/media/market runners. Add image build jobs only
with an explicit product contract for a vetted document or embedding workload.

## Contract For Future Runners

1. Read `/work/job.json`.
2. Use inputs staged under `/work`.
3. Write output artifacts under `/work/output`.
4. Write `/work/receipt.json`.
5. Write `/work/metrics.json` with `output_name`.

## Boundaries

- Do not restore Blender/render/transcode runners.
- Do not add Chronomarket, market simulation, speculative verifier,
  live-trading, browser-agent, or mesh runners.
- Shared orchestration belongs in `ryvion-hub` or `ryvion-node`, not in runner images.
