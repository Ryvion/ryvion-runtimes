# em-fdtd-runner — OCI FALLBACK image for Ryvion EM (FDTD) simulation jobs.
#
# Native-first: operators run EM via the node-agent native bundle (no Docker).
# This image is the FALLBACK for Linux/power nodes (architecture §10) and the
# dev loop. Photonics (Meep) is the OCI/Linux lane; gprMax/openEMS lead native.
#
# Contract: read /work/job.json -> write /work/output/result.npz (+result.json),
# /work/receipt.json, /work/metrics.json. Runs --network=none.
#
# License note (doc §3.1): Meep/gprMax/openEMS are GPL/GPLv3, run as a hosted
# backend never distributed to buyers -> no copyleft distribution trigger. CI
# must verify no AGPL transitive dependency is pulled in.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RYVION_WORK_DIR=/work

WORKDIR /opt/em-fdtd-runner

# Minimal deps for the result contract. Heavy FDTD engines (meep/gprmax/openems)
# are layered on top in the per-engine build variants; numpy is enough to emit a
# valid result.npz and run the analytic fallback used by CI / dev.
RUN pip install --no-cache-dir numpy==1.26.4

# TODO(em): engine layers. The CUDA/ROCm engine builds are added as separate
# build stages / tags (em-fdtd-runner:meep, :gprmax) so the consumer node can
# select by GPU vendor at runtime. Kept out of the base image to keep it small.

COPY . /opt/em-fdtd-runner

# Sanity: schema + sources import cleanly at build time.
RUN python -c "import json,glob; [json.load(open(p)) for p in glob.glob('*.json')]" \
 && python -m py_compile run.py budget.py geometry.py results.py engine.py \
      engine_analytic.py engine_gprmax.py engine_openems.py engine_meep.py \
      gprmax_io.py openems_io.py meep_io.py \
      templates/registry.py templates/grating_coupler.py \
      templates/waveguide_coupler.py templates/antenna_unit_cell.py \
      templates/metasurface_unit_cell.py

WORKDIR /work
ENTRYPOINT ["python", "/opt/em-fdtd-runner/run.py"]
