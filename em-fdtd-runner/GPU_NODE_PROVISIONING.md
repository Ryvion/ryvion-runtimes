# Provisioning a GPU node for REAL EM physics + publishing the native bundle

This is the founder hardware step. It takes the EM workload from
"analytic placeholder" to a **verified real gprMax FDTD solve**, then packages
that into the signed native bundle operators download.

Do the stages in order. **Stage 1 is the decisive, cheap one** (no hub, no
node-agent, no bundle) — get a REAL solve on one box first; everything after is
just delivery plumbing.

> **Honest scope for v1.** gprMax's CUDA engine is a Cython source build + pycuda
> + the NVIDIA CUDA Toolkit (with `nvcc` on `PATH` at *runtime* — pycuda
> JIT-compiles the kernels). NVIDIA is the supported path; AMD/ROCm is
> best-effort. The v1 node relies on the **host** CUDA toolkit being installed;
> a fully self-contained "ships its own nvcc" bundle is a later optimization.

---

## Prerequisites (the GPU box)

- An **NVIDIA** GPU (consumer is fine — RTX 3060/4070/4090, ≥8 GB VRAM) with a
  recent driver (`nvidia-smi` works).
- **CUDA Toolkit** installed and on `PATH` (`nvcc --version` works). Match the
  toolkit to a compiler gcc/clang the toolkit supports.
- **conda/miniconda** (gprMax ships a `conda_env.yml`).
- Linux x86_64 is the smoothest first target. (Windows works too; do Linux first.)

---

## Stage 1 — Install gprMax and prove REAL physics (no hub/node)

```bash
# 1. get gprMax + its conda env
git clone https://github.com/gprMax/gprMax.git
cd gprMax
conda env create -f conda_env.yml
conda activate gprMax

# 2. build the Cython solver (NOT a plain pip install)
python setup.py cleanall || true
python setup.py build
python setup.py install

# 3. GPU + output deps
pip install pycuda h5py        # pycuda = CUDA engine; h5py = read the .out HDF5

# 4. sanity
python -c "import gprMax, pycuda, h5py, numpy; print('gprMax import OK')"
nvcc --version                  # MUST work at runtime; pycuda JIT-needs it
nvidia-smi
```

Now point the runner's interpreter at THIS conda env and run the one-command
truth check from the `em-fdtd-runner/` directory:

```bash
cd /path/to/ryvion-runtimes/em-fdtd-runner

# antenna patch (RF) — auto-detects the GPU:
python tools/smoke_real_physics.py --template antenna_unit_cell --keep

# 6G metasurface unit cell:
python tools/smoke_real_physics.py --template metasurface_unit_cell --keep
```

You want to see:

```
  engine_version  : smoke              <- NO "+analytic" suffix
  gprMax .out file: model.out          <- a real HDF5 output was produced
  RESULT: ✓ REAL gprMax FDTD SOLVE CONFIRMED on this machine.   (exit 0)
```

If you instead see `…+analytic`, `gprMax .out file: (none)`, and exit 1, gprMax
isn't importable+runnable in *this* python yet — fix the install (see
Troubleshooting) and re-run. **Until this prints REAL, the workload is still a
placeholder.** This is the single most important checkpoint.

> Force CPU (to compare / debug) with `--gpu none`; force GPU with `--gpu cuda`.
> The engine reads `RYVION_EM_GPU` and passes `-gpu 0` to gprMax when a CUDA
> device is visible (`CUDA_VISIBLE_DEVICES`).

---

## Stage 2 — Build + sign the native bundle (on the GPU box)

The bundle is what operators download (no Docker). It must be built **on a
target OS+GPU** so the right portable CPython, gprMax build, and CUDA libs are
present. Layout: `<engine>-<version>/{manifest.json,runner/,python/,engine/,bundle.sha256}`.

```bash
# one-time: generate the platform Ed25519 bundle-signing key (32-byte seed, hex)
python - <<'PY'
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
sk = Ed25519PrivateKey.generate()
seed = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                        serialization.NoEncryption())
pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
open("em_bundle_ed25519.key","w").write(seed.hex())     # KEEP SECRET
open("em_bundle_ed25519.pub","w").write(pk.hex())        # give the node (hex)
print("pubkey (hex):", pk.hex())
PY
```

Stage a portable CPython (e.g. from `python-build-standalone`) into
`/opt/python-portable`, and the gprMax env + CUDA libs into `/opt/gprmax-bundle`
(`<env>/lib/python*/site-packages/gprMax`, pycuda, h5py, numpy, plus the CUDA
runtime `.so`s under `engine/cuda/`). Then:

```bash
cd /path/to/ryvion-runtimes/em-fdtd-runner

python tools/build_bundle.py \
  --engine gprmax --engine-version v1 \
  --os linux --arch x86_64 --gpu cuda \
  --out ./dist \
  --engine-root /opt/gprmax-bundle \
  --python-root /opt/python-portable \
  --signing-key ./em_bundle_ed25519.key

# verify it (re-hash every file, recompute aggregate, check the signature):
python tools/verify_bundle.py ./dist/gprmax-v1 \
  --public-key ./em_bundle_ed25519.pub --require-signature
# -> "bundle OK: ./dist/gprmax-v1"
```

> **`--engine-version v1`** is just the bundle KEY (the dir name + URL segment).
> Keep it consistent with the hub's `EM_ENGINE_VERSION` (Stage 3). The real
> gprMax release is recorded inside the result/receipt at run time.

> **Dev/CI dry run** (no GPU, runner-only skeleton, still a valid manifest):
> `python tools/build_bundle.py --engine gprmax --engine-version v1 --os linux
> --arch x86_64 --gpu none --out ./dist --skip-engine --skip-python
> --allow-unsigned` — this bundle *will* degrade to analytic at run time (no
> engine staged); it only proves the packaging.

---

## Stage 3 — Publish the bundle + point the node + hub at it

The node downloads `<base>/<engine>/<version>/<goos>-<goarch>.tar.gz` and
extracts it into `~/.ryvion/runtimes/em/<engine>-<version>/`. The archive must
contain the bundle **contents at its root** (so `runner/run.py` and
`python/bin/python3` sit at the extracted root):

```bash
# name it by GOOS-GOARCH: linux/amd64 -> linux-amd64.tar.gz (NOT x86_64)
tar -C ./dist/gprmax-v1 -czf linux-amd64.tar.gz .
# upload linux-amd64.tar.gz to  <CDN>/gprmax/v1/linux-amd64.tar.gz
```

On the **operator node** (env):

```bash
export RYV_EM_BUNDLE_BASE_URL="https://<your-cdn>/em"   # -> /em/gprmax/v1/linux-amd64.tar.gz
export RYV_EM_BUNDLE_PUBKEY="$(cat em_bundle_ed25519.pub)"   # hex ed25519 pubkey
# host CUDA toolkit must be on PATH for v1 (pycuda needs nvcc at runtime)
```

On the **hub** (env), pin the version the spec advertises so it matches the dir:

```bash
export EM_ENGINE_VERSION="v1"     # -> bundle key gprmax-v1
```

> **Honest gap (flag for later, not a blocker for your own node).** Today the hub
> does NOT yet embed a *signed download descriptor* in the job spec, so the node
> DERIVES the bundle URL and has no per-spec signature to verify. For your own
> single node you control end-to-end, set `RYV_EM_ALLOW_UNSIGNED_BUNDLE=1` on the
> node (it still extracts your bundle; the bundle's OWN manifest is signed and
> verifiable via `verify_bundle.py`). For multi-operator production, the
> follow-up is: have the hub sign+emit a `runtime` block
> (`{bundle_url,bundle_sha256,entrypoint,signature}`) so nodes verify the
> descriptor with `RYV_EM_BUNDLE_PUBKEY` and no allow-unsigned flag.

**Skip-download shortcut (fastest end-to-end test):** pre-place the bundle in the
cache and the node uses it without any CDN:

```bash
mkdir -p ~/.ryvion/runtimes/em/
cp -r ./dist/gprmax-v1 ~/.ryvion/runtimes/em/gprmax-v1
: > ~/.ryvion/runtimes/em/gprmax-v1/.em-bundle-ready     # empty ready marker
export RYV_EM_ALLOW_UNSIGNED_BUNDLE=1                     # cache-hit still checks signature first
```

---

## Stage 4 — Full end-to-end (hub → node → real solve)

1. Start the node with `--gpus 0` (or `auto`) so EM is assigned the GPU; keep the
   env from Stage 3.
2. Launch a 1-point Study from the hub (or POST one `em_simulation` job) for
   `antenna_unit_cell` / `metasurface_unit_cell`.
3. The node routes it to `native_em`, resolves `gprmax-v1`, runs
   `runner/run.py` with the **bundle's** python, solves on the GPU, and uploads
   `result.npz`.
4. Confirm REAL: the receipt's `engine_version` has **no** `+analytic` suffix
   (the buyer's dataset row is real physics). If it shows `+analytic`, the node
   used the host/system python without gprMax — check the embedded interpreter
   exists at `~/.ryvion/runtimes/em/gprmax-v1/python/bin/python3`, or set
   `RYV_EM_PYTHON` to your gprMax conda python for a quick win.

---

## Troubleshooting (why you got `+analytic`)

| Symptom | Cause | Fix |
|---|---|---|
| `+analytic`, `.out file: (none)` in smoke | gprMax/pycuda/h5py not importable in this python | re-run Stage 1 step 4; activate the right env |
| gprMax runs on CPU, slow | no CUDA device visible | `nvidia-smi`; set `CUDA_VISIBLE_DEVICES=0`; `--gpu cuda` |
| `nvcc: not found` at run time | CUDA toolkit not on `PATH` | add `$CUDA_HOME/bin` to `PATH` (pycuda JIT needs nvcc) |
| node job is `+analytic` but Stage 1 was REAL | node used system python, not the bundle's | ensure `python/bin/python3` is in the bundle, or set `RYV_EM_PYTHON` |
| `EM bundle signing key not configured` | `RYV_EM_BUNDLE_PUBKEY` unset and no allow flag | set the pubkey, or `RYV_EM_ALLOW_UNSIGNED_BUNDLE=1` for your own node |
| `entrypoint not found after extract` | archive tarred the dir, not its contents | `tar -C dist/gprmax-v1 -czf … .` (note the trailing `.`) |
| OOM / exit early on GPU | cell budget too big for VRAM | lower grid `resolution` / cap `max_cells`; raise `est_vram_mb` |

Node EM env knobs (all read by `ryvion-node/internal/runner/native_em*.go`):
`RYV_EM_BUNDLE_BASE_URL`, `RYV_EM_BUNDLE_PUBKEY`, `RYV_EM_ALLOW_UNSIGNED_BUNDLE`,
`RYVION_EM_RUNTIME_ROOT`, `RYV_EM_PYTHON`, `RYV_EM_CPU_CORES`, `RYV_EM_RAM_CAP_MB`.
Engine GPU knob (read by `engine_gprmax.py`): `RYVION_EM_GPU` (`auto|cuda|none`).
