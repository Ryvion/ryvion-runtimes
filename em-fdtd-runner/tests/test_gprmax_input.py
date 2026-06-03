"""test_gprmax_input.py — unit tests for the IR -> gprMax input GENERATION.

These exercise gprmax_io.build_input(geo, job) against the real device
templates. No gprMax / GPU required (pure string generation). Run:

    python -m unittest tests.test_gprmax_input -v

(from the em-fdtd-runner/ directory).
"""
from __future__ import annotations

import os
import sys
import unittest

# Make the runner modules importable when run from the repo root or here.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.dirname(_HERE)
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

import gprmax_io  # noqa: E402
from templates import registry  # noqa: E402


def _commands(text: str):
    """Map command-name -> list of full lines for that #command."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#"):
            continue
        name = line.split(":", 1)[0]
        out.setdefault(name, []).append(line)
    return out


ANTENNA_JOB = {
    "schema_version": "em.job.v1",
    "task": "em_simulation",
    "engine": "gprmax",
    "device_template": "antenna_unit_cell",
    "seed": 7,
    "params": {"patch_w_mm": 12.0, "patch_l_mm": 16.0, "substrate_eps": 4.4,
               "substrate_h_mm": 1.6, "feed_pos": 0.3},
    "source": {"type": "gaussian", "center_freq_ghz": 5.0, "bandwidth": 4.0,
               "polarization": "Ez", "port": "feed"},
    "grid": {"resolution": 4, "pml_layers": 8, "dimensionality": 3},
    "frequency": {"points": 41, "range_ghz": [3.0, 8.0]},
    "outputs": {"want": ["s_params", "fom", "radiation_pattern"]},
}

METASURFACE_JOB = {
    "schema_version": "em.job.v1",
    "task": "em_simulation",
    "engine": "gprmax",
    "device_template": "metasurface_unit_cell",
    "params": {"element_size_um": 4.0, "post_h_um": 30.0, "post_r_um": 8.0,
               "lattice_period_um": 40.0, "freq_ghz": 100.0},
    "materials": {"post_eps": 11.7},
    "grid": {"resolution": 20, "pml_layers": 8, "dimensionality": 3},
    "frequency": {"points": 21, "range_ghz": [80.0, 120.0]},
    "outputs": {"want": ["reflection", "transmission", "phase"]},
}

GRATING_JOB = {
    "schema_version": "em.job.v1",
    "task": "em_simulation",
    "engine": "gprmax",
    "device_template": "grating_coupler",
    "params": {"period_nm": 630, "fill_factor": 0.5, "etch_depth_nm": 70,
               "n_teeth": 8, "wg_thickness_nm": 220},
    "materials": {"core_index": 3.48, "clad_index": 1.44},
    "source": {"center_wavelength_nm": 1550, "bandwidth": 100, "polarization": "TE"},
    "grid": {"resolution": 30, "pml_layers": 8, "dimensionality": 2},
    "frequency": {"points": 51, "range_nm": [1500, 1600]},
    "outputs": {"want": ["transmission", "phase", "fom"]},
}


class TestBuildInput(unittest.TestCase):

    def _build(self, job):
        tmpl = registry.get(job["device_template"])
        geo = tmpl.build(job["params"], job)
        text = gprmax_io.build_input(geo, job)
        return geo, text, _commands(text)

    # ---- structural invariants common to every template ----

    def test_required_header_commands_present(self):
        for job in (ANTENNA_JOB, METASURFACE_JOB, GRATING_JOB):
            _geo, _text, cmds = self._build(job)
            for required in ("#title", "#domain", "#dx_dy_dz", "#pml_cells",
                             "#time_window", "#waveform"):
                self.assertIn(required, cmds,
                              f"{job['device_template']} missing {required}")
            self.assertEqual(len(cmds["#domain"]), 1)
            self.assertEqual(len(cmds["#dx_dy_dz"]), 1)
            self.assertEqual(len(cmds["#waveform"]), 1)

    def test_domain_has_three_positive_floats(self):
        for job in (ANTENNA_JOB, METASURFACE_JOB, GRATING_JOB):
            _geo, _text, cmds = self._build(job)
            parts = cmds["#domain"][0].split(":", 1)[1].split()
            self.assertEqual(len(parts), 3)
            for p in parts:
                self.assertGreater(float(p), 0.0)

    def test_dx_dy_dz_three_positive_floats(self):
        for job in (ANTENNA_JOB, METASURFACE_JOB, GRATING_JOB):
            _geo, _text, cmds = self._build(job)
            parts = cmds["#dx_dy_dz"][0].split(":", 1)[1].split()
            self.assertEqual(len(parts), 3)
            for p in parts:
                self.assertGreater(float(p), 0.0)

    def test_pml_cells_matches_grid(self):
        _geo, _text, cmds = self._build(ANTENNA_JOB)
        n = int(cmds["#pml_cells"][0].split(":", 1)[1].strip())
        self.assertEqual(n, 8)

    def test_time_window_positive(self):
        for job in (ANTENNA_JOB, METASURFACE_JOB, GRATING_JOB):
            _geo, _text, cmds = self._build(job)
            tw = float(cmds["#time_window"][0].split(":", 1)[1].strip())
            self.assertGreater(tw, 0.0)

    # ---- material translation ----

    def test_dielectric_materials_emitted_once_per_eps(self):
        # antenna substrate eps 4.4 -> exactly one #material; metal -> pec (none).
        _geo, _text, cmds = self._build(ANTENNA_JOB)
        mats = cmds.get("#material", [])
        self.assertEqual(len(mats), 1, f"expected 1 dielectric material, got {mats}")
        parts = mats[0].split(":", 1)[1].split()
        # eps_r sigma mu_r sigma* identifier
        self.assertEqual(len(parts), 5)
        self.assertAlmostEqual(float(parts[0]), 4.4, places=6)
        self.assertEqual(parts[1], "0")  # lossless
        self.assertEqual(parts[2], "1")  # mu_r

    def test_metal_boxes_use_pec(self):
        _geo, text, cmds = self._build(ANTENNA_JOB)
        # ground + patch are PEC -> their #box lines reference "pec".
        pec_boxes = [b for b in cmds.get("#box", []) if b.rstrip().endswith(" pec")]
        self.assertGreaterEqual(len(pec_boxes), 2)

    # ---- source translation ----

    def test_antenna_uses_transmission_line_feed(self):
        _geo, _text, cmds = self._build(ANTENNA_JOB)
        # antenna has an s_param monitor -> transmission_line (for S11), not dipole.
        self.assertIn("#transmission_line", cmds)
        self.assertNotIn("#hertzian_dipole", cmds)
        tl = cmds["#transmission_line"][0].split(":", 1)[1].split()
        # polarisation coords resistance waveform_id
        self.assertIn(tl[0], ("x", "y", "z"))
        self.assertEqual(tl[-1], "ryv_pulse")
        # resistance default 50 ohm in position 4 (after the 3 coords).
        self.assertAlmostEqual(float(tl[4]), 50.0, places=3)

    def test_metasurface_uses_dipole_source(self):
        _geo, _text, cmds = self._build(METASURFACE_JOB)
        self.assertIn("#hertzian_dipole", cmds)
        self.assertNotIn("#transmission_line", cmds)

    def test_waveform_type_and_freq(self):
        _geo, _text, cmds = self._build(ANTENNA_JOB)
        wf = cmds["#waveform"][0].split(":", 1)[1].split()
        # type amplitude centre_freq id
        self.assertEqual(wf[0], "gaussiandotnorm")
        self.assertEqual(wf[-1], "ryv_pulse")
        self.assertGreater(float(wf[2]), 0.0)  # centre frequency Hz

    # ---- geometry primitives ----

    def test_metasurface_post_emits_cylinder(self):
        _geo, _text, cmds = self._build(METASURFACE_JOB)
        self.assertIn("#cylinder", cmds, "dielectric post should be a #cylinder")
        cyl = cmds["#cylinder"][0].split(":", 1)[1].split()
        # x1 y1 z1 x2 y2 z2 radius material  -> 8 tokens
        self.assertEqual(len(cyl), 8)
        self.assertGreater(float(cyl[6]), 0.0)  # radius

    def test_receivers_emitted_for_flux_monitors(self):
        _geo, _text, cmds = self._build(METASURFACE_JOB)
        # metasurface has R and T flux monitors -> two #rx.
        self.assertIn("#rx", cmds)
        self.assertEqual(len(cmds["#rx"]), 2)

    def test_grating_collapses_to_single_z_slab(self):
        geo, _text, cmds = self._build(GRATING_JOB)
        # 2D -> domain z thickness == one cell; dx_dy_dz z == dx.
        dom = cmds["#domain"][0].split(":", 1)[1].split()
        ddd = cmds["#dx_dy_dz"][0].split(":", 1)[1].split()
        self.assertAlmostEqual(float(dom[2]), float(ddd[2]), places=12)

    def test_every_command_single_line(self):
        # gprMax requires one command per line; ensure no embedded newlines.
        _geo, text, _cmds = self._build(GRATING_JOB)
        for line in text.splitlines():
            if line.startswith("#"):
                self.assertEqual(line.count("#"), 1, f"multiple commands on line: {line!r}")

    def test_deterministic_output(self):
        # Same job -> byte-identical input file (QA reproducibility).
        _g1, t1, _ = self._build(ANTENNA_JOB)
        _g2, t2, _ = self._build(ANTENNA_JOB)
        self.assertEqual(t1, t2)


class TestInputInjectionSecurity(unittest.TestCase):
    """Regression guard for the gprMax `#python:` injection RCE (variant_id was
    interpolated raw into #title; gprMax executes #python: blocks). See
    SECURITY-AUDIT-2026-06-02."""

    # The exact proof-of-concept payload from the audit.
    POC = "x\n#python:\nimport os; os.system('id')\n#end_python:"

    def _build(self, job):
        tmpl = registry.get(job["device_template"])
        geo = tmpl.build(job["params"], job)
        return gprmax_io.build_input(geo, job)

    def test_variant_id_cannot_inject_scripting_block(self):
        job = dict(ANTENNA_JOB, variant_id=self.POC)
        text = self._build(job)
        lowered = text.lower()
        self.assertNotIn("#python", lowered)
        self.assertNotIn("#end_python", lowered)
        self.assertNotIn("#import", lowered)
        # Title stays a single line (no breakout).
        for line in text.splitlines():
            if line.startswith("#title"):
                self.assertEqual(line.count("#"), 1)

    def test_sanitizer_strips_newlines_and_hash(self):
        tok = gprmax_io._safe_inline_token(self.POC)
        self.assertNotIn("\n", tok)
        self.assertNotIn("#", tok)

    def test_guard_rejects_smuggled_directive(self):
        with self.assertRaises(ValueError):
            gprmax_io._assert_no_scripting(["#title: ok", "  #python:", "import os"])

    def test_legit_variant_id_unaffected(self):
        job = dict(ANTENNA_JOB, variant_id="sweep-0042.v3")
        text = self._build(job)
        self.assertIn("sweep-0042.v3", text)


if __name__ == "__main__":
    unittest.main()
