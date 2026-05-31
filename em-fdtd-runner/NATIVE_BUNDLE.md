# EM native bundle layout (no Docker required)

Per architecture §10, EM runs **natively** on operator machines by default;
OCI (the `Dockerfile` here) is the Linux/power-node fallback. This documents
the bundle the node-agent downloads and the entrypoint it invokes — the same
download / verify / auto-update path the inference runtime already uses for
`llama-server` + GGUFs under `~/.ryvion/`.

## On-disk layout

```
~/.ryvion/runtimes/em/<engine>-<version>/
├── manifest.json          # bundle descriptor (see below), Ed25519-signed
├── runner/                # this directory's .py files, frozen at <version>
│   ├── run.py             # ENTRYPOINT the node invokes
│   ├── budget.py geometry.py results.py
│   ├── engine.py engine_<engine>.py engine_analytic.py
│   └── templates/...
├── python/                # portable embedded CPython (per OS×arch)
│   └── bin/python3        # or python.exe on Windows
├── engine/                # the native FDTD engine + its GPU runtime
│   ├── gprmax/            # gprMax wheels + CUDA kernels   (native lead)
│   ├── openems/           # openEMS prebuilt binaries       (native lead, Win)
│   └── cuda/ | rocm/      # bundled GPU runtime libs, vendor-selected at run
└── bundle.sha256          # hash of all files, matched against manifest
```

`<engine>` ∈ `gprmax | openems | meep`. `<version>` is `engine_version` from the
job contract (the bundle KEY, e.g. `gprmax-v1`). **gprMax's CUDA engine is a
Cython source build + pycuda, NOT a pip wheel** — build it on a target OS+GPU per
`GPU_NODE_PROVISIONING.md`. openEMS ships prebuilt binaries; Meep stays OCI/Linux.

## manifest.json (signed)

```json
{
  "schema": "em.bundle.v1",
  "engine": "gprmax",
  "engine_version": "3.1.7+cuda12",
  "os": "linux", "arch": "x86_64", "gpu": ["cuda", "rocm"],
  "entrypoint": ["python/bin/python3", "runner/run.py"],
  "templates": ["grating_coupler","waveguide_coupler","antenna_unit_cell","metasurface_unit_cell"],
  "sha256": "…", "size_bytes": 0, "signature": "ed25519:…"
}
```

## Node-side execution (node-agent responsibility)

1. Scheduler assigns an `em_simulation` job whose execution kind is
   `ExecutorKindNativeEM` (vs `ExecutorKindManagedOCI`).
2. Node ensures `~/.ryvion/runtimes/em/<engine>-<version>/` exists; if not,
   download + verify signature + `bundle.sha256`, atomically swap (auto-update).
3. Stage inputs into a working dir jail: write `job.json` into `<work>/`.
4. Invoke `manifest.entrypoint` with `RYVION_WORK_DIR=<work>` and **no network
   access** (native equivalent of `--network=none`). Apply mem/cpu/time caps and
   kill on `TimeoutSeconds` — the sandbox OCI gave for free is re-created here.
5. Read `<work>/receipt.json` + `<work>/metrics.json`, upload
   `<work>/output/<output_name>` exactly as the OCI path does.

The job/result CONTRACT is identical to the OCI path, so the hub, StudyService,
QA, aggregator, billing, and buyer UX are unchanged. Native vs OCI is a
`ryvion-node` + packaging concern only.
