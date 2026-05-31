"""test_run_native_contract.py — runner <-> node-agent execution contract.

Regression test for the native-EM wiring: the node-agent launches run.py with
`--job <dir>/job.json --work <dir>` (and a RYV_WORK_DIR env), from a foreign
working directory, with RYVION_WORK_DIR UNSET. run.py must still find job.json,
run, and write receipt.json / metrics.json / output/result.json INTO that work
dir — not the OCI default /work. Before the fix run.py read only RYVION_WORK_DIR
and ignored argv, so every native job failed before the engine started.

No gprMax / GPU required: with no native engine importable the runner takes the
deterministic analytic fallback, which is exactly what proves the *contract*
(staging -> run -> receipt -> artifact) independently of the physics. Run:

    python -m unittest tests.test_run_native_contract -v
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.dirname(_HERE)
_RUN_PY = os.path.join(_RUNNER, "run.py")

EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

_JOB = {
    "schema_version": "em.job.v1",
    "task": "em_simulation",
    "engine": "gprmax",
    "engine_version": "v1",
    "study_id": "study_contract",
    "variant_id": "study_contract#0000",
    "point_index": 0,
    "seed": 7,
    "device_template": "antenna_unit_cell",
    "params": {"patch_w_mm": 12.0, "patch_l_mm": 16.0, "substrate_eps": 4.4,
               "substrate_h_mm": 1.6, "feed_pos": 0.3},
    "source": {"type": "gaussian", "center_freq_ghz": 5.0, "bandwidth": 4.0,
               "polarization": "Ez", "port": "feed"},
    "grid": {"resolution": 4, "pml_layers": 8, "dimensionality": 3},
    "frequency": {"points": 41, "range_ghz": [3.0, 8.0]},
    "outputs": {"want": ["s_params", "fom", "radiation_pattern"]},
    "budget": {"max_cells": 30000000, "est_vram_mb": 4000, "est_runtime_s": 60},
}


class TestNativeRunContract(unittest.TestCase):

    def _run(self, work_dir, *, use_argv, use_env, foreign_cwd):
        job_path = os.path.join(work_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as fh:
            json.dump(_JOB, fh)

        # A clean env that emphatically does NOT carry RYVION_WORK_DIR (the var
        # the runner used to depend on) so we prove the new resolution works.
        env = {k: v for k, v in os.environ.items() if k != "RYVION_WORK_DIR"}
        if use_env:
            env["RYV_WORK_DIR"] = work_dir  # node-agent sets this alias.

        cmd = [sys.executable, _RUN_PY]
        if use_argv:
            cmd += ["--job", job_path, "--work", work_dir]  # node-agent argv form.

        proc = subprocess.run(
            cmd, cwd=foreign_cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        )
        return proc

    def _assert_valid_result(self, work_dir, proc):
        self.assertEqual(proc.returncode, 0,
                         f"runner failed: {proc.stdout.decode('utf-8', 'ignore')}")

        receipt_path = os.path.join(work_dir, "receipt.json")
        metrics_path = os.path.join(work_dir, "metrics.json")
        self.assertTrue(os.path.exists(receipt_path), "receipt.json not written into work dir")
        self.assertTrue(os.path.exists(metrics_path), "metrics.json not written into work dir")

        with open(receipt_path, encoding="utf-8") as fh:
            receipt = json.load(fh)
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)

        # A real (non-empty) artifact hash => the engine ran and wrote a result.
        self.assertNotEqual(receipt["output_hash"], EMPTY_SHA256, "empty/failed receipt")
        self.assertTrue(receipt["converged"])
        # No gprMax here -> analytic fallback, surfaced honestly on engine_version.
        self.assertTrue(str(receipt["engine_version"]).endswith("+analytic"),
                        f"expected +analytic marker, got {receipt['engine_version']!r}")

        # metrics.output_name must locate the artifact the node uploads.
        out_name = metrics["output_name"]
        self.assertTrue(out_name, "metrics.output_name empty -> node uploads wrong file")
        artifact = os.path.join(work_dir, "output", out_name)
        self.assertTrue(os.path.exists(artifact), f"artifact {artifact} missing")

        # The hash in the receipt must match the artifact actually written.
        h = hashlib.sha256()
        with open(artifact, "rb") as fh:
            h.update(fh.read())
        self.assertEqual(receipt["output_hash"], h.hexdigest())

    def test_node_argv_form(self):
        """Node passes --job/--work; RYVION_WORK_DIR unset; foreign cwd."""
        with tempfile.TemporaryDirectory() as work_dir, \
                tempfile.TemporaryDirectory() as cwd:
            proc = self._run(work_dir, use_argv=True, use_env=True, foreign_cwd=cwd)
            self._assert_valid_result(work_dir, proc)

    def test_node_env_only_form(self):
        """RYV_WORK_DIR env alone (no argv) also resolves the work dir."""
        with tempfile.TemporaryDirectory() as work_dir, \
                tempfile.TemporaryDirectory() as cwd:
            proc = self._run(work_dir, use_argv=False, use_env=True, foreign_cwd=cwd)
            self._assert_valid_result(work_dir, proc)


if __name__ == "__main__":
    unittest.main()
