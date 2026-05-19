# Runners

Container images for Ryvion render/media workloads. Built by GitHub Actions and pushed to `ghcr.io/ryvion/*`.

## Build & CI

`.github/workflows/build.yml` validates and builds only the active runners:

- `blender-runner`
- `transcode-runner`

Pushes publish `:latest` and `:${SHA}` tags after image build, vulnerability scan, and signing.

Local validation:

- `python -m py_compile blender-runner/run.py`
- `cd transcode-runner && go test ./... && go vet ./... && go build ./...`

## Container Contract

All active containers:

1. Read `/work/job.json` written by `ryvion-node`.
2. Use inputs already staged under `/work`.
3. Write output artifacts under `/work/`.
4. Write `/work/receipt.json`.
5. Write `/work/metrics.json` with an `output_name` field so `ryvion-node` can find the artifact.

## Network Policy

- Active runners should assume `--network=none`.
- `ryvion-node` is responsible for staging input files before container start.
- Do not add runner-side downloads unless the control-plane contract explicitly changes.

## Active Runners

### blender-runner

Runs Blender scene renders from staged scene/assets. It writes render output files plus receipt and metrics documents.

### transcode-runner

Runs media transcode/post-processing work. It is built for `linux/amd64` and `linux/arm64`.

## Archived Directions

AI inference, embedding, image generation, whisper, fine-tuning, vLLM, verifier, agent-hosting, and spatial/reality experiment runners are not active in this repository. Keep those directions in `ryvion-archive`; do not import from the archive or re-add them to the active build matrix.

## Common Gotchas

- `metrics.json` must include `output_name`; otherwise `ryvion-node` will not locate the artifact.
- Runner code must keep signal handling graceful enough for abort/failure semantics.
- Keep runners workload-specific and small. Shared orchestration belongs in `ryvion-hub` or `ryvion-node`, not in runner images.
