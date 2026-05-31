# Runners

Active runners:

- **`em-fdtd-runner/`** — EM (electromagnetic FDTD) parametric-sweep workload.
  One job == one parameter point of a Study. Native-first: operators run it via
  the node-agent native bundle (no Docker; see `em-fdtd-runner/NATIVE_BUNDLE.md`);
  the `Dockerfile` here is the OCI/Linux fallback. Engines: gprMax/openEMS lead
  the native path (antenna + 6G-metasurface templates); Meep is the OCI photonics
  lane. Job contract is `em.job.v1` (`em-fdtd-runner/schema.json`).
  **Real-physics provisioning:** `em-fdtd-runner/GPU_NODE_PROVISIONING.md` (install
  gprMax → verify → publish bundle). One-command truth check on a GPU box:
  `python tools/smoke_real_physics.py` (REAL = exit 0; `+analytic` = exit 1).
  gprMax's CUDA engine is a Cython source build, NOT a pip wheel.

The native llama.cpp inference path lives in `ryvion-node`.

## CI

`.github/workflows/build.yml` allows the `em-fdtd-runner` image, validates its
`schema.json` + `py_compile`, gates against AGPL deps, and builds/pushes the OCI
fallback to `ghcr.io/ryvion/em-fdtd-runner:latest`. It still blocks reintroducing
obsolete render/media/market runners.

## Runner Contract

1. Read `/work/job.json`.
2. Use inputs staged under `/work`.
3. Write output artifacts under `/work/output` (EM: `result.npz` + `result.json`).
4. Write `/work/receipt.json` (with `output_hash`).
5. Write `/work/metrics.json` with `output_name`.

On any failure the runner still writes an error receipt with a hash (never hang
the hub). Runs `--network=none` (OCI) / no-network (native) once inputs staged.

## Boundaries

- Do not restore Blender/render/transcode runners.
- Do not add Chronomarket, market simulation, speculative verifier,
  live-trading, browser-agent, or mesh runners.
- Engines run as a hosted backend (never distributed to buyers): GPL/GPLv3 is
  clean here, but no AGPL deps.
- Shared orchestration belongs in `ryvion-hub` or `ryvion-node`, not in runner images.
