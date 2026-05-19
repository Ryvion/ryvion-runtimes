# Ryvion runtimes

This repository is currently a contract placeholder for trusted managed OCI
runners. There are no active render, media, or transcode images in the build
matrix.

Current execution ownership:

- Native AI inference runs in `ryvion-node` through a local llama.cpp server.
- Managed OCI remains available for future custom/code workloads.
- Runner image definitions should be added here only when the hub/node contract
  needs a container image that cannot live as node-local runtime plumbing.

Runner contract for future images:

1. Read `/work/job.json`.
2. Use inputs already staged under `/work`.
3. Write artifacts under `/work/output`.
4. Write `/work/receipt.json`.
5. Write `/work/metrics.json` with `output_name`.

Inactive runner families should stay archived outside active production repos.
