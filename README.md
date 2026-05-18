# Ryvion runtimes

This repository owns the workload-specific container images that `ryvion-hub` can assign to `ryvion-node`.

Current scope:

- `blender-runner`
- `transcode-runner`

Runner ownership rule:

- Runtime images live in this repository.
- `ryvion-node` is responsible for runtime execution, not for owning runner image definitions.
- `blender-runner` is the active scene-rendering container.
- `transcode-runner` is the active media post-processing container.
- Old AI, verifier, agent, and spatial experiment runners live in `ryvion-archive`; production repos must not import from the archive.

Runner contract:

1. `ryvion-hub` writes a structured `job.json` into `/work/job.json`.
2. `ryvion-node` mounts `/work` and runs the assigned container image.
3. The runner must:
   - read `/work/job.json`
   - use inputs already staged under `/work`
   - write artifacts under `/work/output`
   - write `/work/receipt.json`
   - write `/work/metrics.json`

Active images:

- `ghcr.io/ryvion/blender-runner:latest`
- `ghcr.io/ryvion/transcode-runner:latest`

Compatibility:

- Each active image receives `latest` and commit SHA tags.
- Inactive images are not built or published from this repository.
