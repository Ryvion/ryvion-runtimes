# Ryvion runtimes

Active runner images:

- **`em-fdtd-runner/`** — EM (electromagnetic FDTD) simulation. A buyer Study
  fans out into N independent parameter-point jobs; each runs ONE FDTD sim on
  one operator GPU and returns a tiny result (S-params / transmission / phase /
  radiation pattern — KB, never field volumes). Native-first execution (node
  bundle, no Docker — see `em-fdtd-runner/NATIVE_BUNDLE.md`); the `Dockerfile` is
  the OCI/Linux fallback. Job contract: `em.job.v1` (`em-fdtd-runner/schema.json`).
  Templates: `grating_coupler`, `waveguide_coupler`, `antenna_unit_cell`,
  `metasurface_unit_cell`.

Native AI inference runs in `ryvion-node` through a local llama.cpp server.

## Runner contract

1. Read `/work/job.json`.
2. Use inputs already staged under `/work`.
3. Write artifacts under `/work/output`.
4. Write `/work/receipt.json` (with `output_hash`).
5. Write `/work/metrics.json` with `output_name`.

The runner always writes an error receipt on failure so the hub gets a clean
fail+refund. Runs offline (`--network=none` / native no-network) once inputs are
staged.

## Engines & licensing

gprMax / openEMS lead the native bundle path (antenna + 6G metasurface); Meep is
the OCI/Linux photonics lane. All are GPL/GPLv3 and run as a hosted backend that
is never distributed to buyers, so there is no copyleft distribution trigger. CI
gates against AGPL dependencies.

Obsolete render/media/transcode/market runner families stay archived outside
this repo.
