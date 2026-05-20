# Ryvion runtimes

This repository is dormant for the Sovereign Relay pilot. There are no active
render, media, transcode, market simulation, or managed OCI runner images in the
build matrix.

Current execution ownership:

- Native AI inference runs in `ryvion-node` through a local llama.cpp server.
- Runner image definitions should be added here only when a future vetted
  document or embedding runner has an explicit hub/node contract that cannot
  live as node-local runtime plumbing.

Runner contract for future images:

1. Read `/work/job.json`.
2. Use inputs already staged under `/work`.
3. Write artifacts under `/work/output`.
4. Write `/work/receipt.json`.
5. Write `/work/metrics.json` with `output_name`.

Inactive runner families should stay archived outside active production repos.
