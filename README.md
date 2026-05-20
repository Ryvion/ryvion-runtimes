# Ryvion runtimes

This repository holds trusted managed OCI runners. There are no active render,
media, or transcode images in the build matrix.

Current execution ownership:

- Native AI inference runs in `ryvion-node` through a local llama.cpp server.
- `market-sim-runner/` executes deterministic Chronomarket paper-trading
  replay jobs for autonomous market agents.
- Runner image definitions should be added here only when the hub/node contract
  needs a container image that cannot live as node-local runtime plumbing.
- The main-branch workflow builds, smoke-tests, and publishes
  `ghcr.io/ryvion/market-sim-runner:latest`, which is the hub's default
  managed OCI image for `marketarena.run.v1` jobs.

Runner contract for future images:

1. Read `/work/job.json`.
2. Use inputs already staged under `/work`.
3. Write artifacts under `/work/output`.
4. Write `/work/receipt.json`.
5. Write `/work/metrics.json` with `output_name`.

Inactive runner families should stay archived outside active production repos.
