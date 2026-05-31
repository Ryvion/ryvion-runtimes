"""test_gprmax_output.py — unit tests for the gprMax .out HDF5 PARSER.

Builds synthetic gprMax .out HDF5 FIXTURE files with h5py (no GPU / no gprMax),
matching the real layout per https://docs.gprmax.com/en/latest/output.html:

  root attrs: gprMax, Title, Iterations, nx_ny_nz, dx_dy_dz, dt, nsrc, nrx
  /tls/tl1   : Vinc, Vtotal, Iinc, Itotal   (antenna / S11 path)
  /rxs/rx1.. : attrs Name, Position ; datasets Ex,Ey,Ez,Hx,Hy,Hz (receiver path)

Then asserts gprmax_io.parse_output(...) maps them onto the em.result.v1
contract correctly. Run:

    python -m unittest tests.test_gprmax_output -v

Skips automatically if numpy/h5py are unavailable (so it never breaks a minimal
CI), but they ARE available in this dev environment so the tests execute.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.dirname(_HERE)
if _RUNNER not in sys.path:
    sys.path.insert(0, _RUNNER)

try:
    import h5py  # type: ignore
    import numpy as np  # type: ignore
    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False

import gprmax_io  # noqa: E402
from templates import registry  # noqa: E402


def _antenna_job():
    return {
        "schema_version": "em.job.v1", "task": "em_simulation", "engine": "gprmax",
        "device_template": "antenna_unit_cell",
        "params": {"patch_w_mm": 12.0, "patch_l_mm": 16.0, "substrate_eps": 4.4,
                   "substrate_h_mm": 1.6},
        "source": {"center_freq_ghz": 5.0, "bandwidth": 4.0, "polarization": "Ez"},
        "grid": {"resolution": 4, "pml_layers": 8, "dimensionality": 3},
        "frequency": {"points": 41, "range_ghz": [3.0, 8.0]},
        "outputs": {"want": ["s_params", "fom", "radiation_pattern"]},
    }


def _metasurface_job():
    return {
        "schema_version": "em.job.v1", "task": "em_simulation", "engine": "gprmax",
        "device_template": "metasurface_unit_cell",
        "params": {"post_h_um": 30.0, "post_r_um": 8.0, "lattice_period_um": 40.0,
                   "freq_ghz": 100.0},
        "grid": {"resolution": 20, "pml_layers": 8, "dimensionality": 3},
        "frequency": {"points": 21, "range_ghz": [80.0, 120.0]},
        "outputs": {"want": ["reflection", "transmission", "phase"]},
    }


def _write_root_attrs(f, iterations, dt, ncells):
    f.attrs["gprMax"] = "3.1.7"
    f.attrs["Title"] = "ryvion-em-fixture"
    f.attrs["Iterations"] = iterations
    f.attrs["nx_ny_nz"] = np.array(ncells, dtype=np.int64)
    f.attrs["dx_dy_dz"] = np.array([1e-3, 1e-3, 1e-3], dtype=np.float64)
    f.attrs["dt"] = dt


def _gauss_pulse(n, dt, f0, t0_frac=0.3):
    t = np.arange(n) * dt
    t0 = n * dt * t0_frac
    tau = 1.0 / (2.0 * math.pi * f0 * 0.3)
    env = np.exp(-((t - t0) ** 2) / (2 * tau * tau))
    return env * np.sin(2 * math.pi * f0 * (t - t0))


def write_antenna_fixture(path, iterations=8192, f_res=5e9):
    """A transmission-line .out where the device resonates near f_res.

    Models a PASSIVE 1-port: the reflected voltage is the incident pulse passed
    through a reflection coefficient rho(f) whose magnitude has a deep notch at
    f_res (|rho| < 1 everywhere, so |S11| <= 1 by construction — the physical
    constraint a real antenna obeys). Implemented in the frequency domain so the
    notch is exact, then transformed back to a real time series.
    """
    dt = 1.0 / (40e9)  # 40 GHz sampling -> resolves up to 20 GHz.
    n = iterations
    Vinc = _gauss_pulse(n, dt, 5.5e9)
    freqs = np.fft.fftfreq(n, d=dt)
    # |rho(f)| = 1 - 0.95*exp(-((f-f_res)/notch_bw)^2): a deep dip at f_res.
    notch_bw = 0.6e9
    rho_mag = 1.0 - 0.95 * np.exp(-((np.abs(freqs) - f_res) ** 2) / (2 * notch_bw ** 2))
    # small linear phase (a fixed group delay) keeps it causal-ish.
    rho = rho_mag * np.exp(-1j * 2 * math.pi * freqs * (5 * dt))
    Vref = np.real(np.fft.ifft(np.fft.fft(Vinc) * rho))
    Vtotal = Vinc + Vref
    Iinc = Vinc / 50.0
    Itotal = Vtotal / 50.0
    with h5py.File(path, "w") as f:
        _write_root_attrs(f, n, dt, [60, 60, 24])
        f.attrs["nsrc"] = 1
        f.attrs["nrx"] = 0
        tls = f.create_group("tls")
        tl = tls.create_group("tl1")
        tl.attrs["Position"] = np.array([0.03, 0.03, 0.0])
        tl.attrs["Resistance"] = 50.0
        tl.create_dataset("Vinc", data=Vinc)
        tl.create_dataset("Vtotal", data=Vtotal)
        tl.create_dataset("Iinc", data=Iinc)
        tl.create_dataset("Itotal", data=Itotal)
    return f_res


def write_metasurface_fixture(path, iterations=4096, comp="Ex"):
    """A receiver .out with T and R monitors recording field time histories.

    The transmitted/reflected energy is written into the field component the
    source polarization drives (`comp`) so the parser (which reads that same
    component) sees it — exactly how a real gprMax rx records the excited field.
    """
    dt = 1.0 / (400e9)  # 400 GHz sampling for a 100 GHz device.
    n = iterations
    # transmitted wave: most energy passes (post is a phase shifter).
    Et = 0.9 * _gauss_pulse(n, dt, 100e9, t0_frac=0.4)
    Er = 0.3 * _gauss_pulse(n, dt, 100e9, t0_frac=0.25)
    with h5py.File(path, "w") as f:
        _write_root_attrs(f, n, dt, [40, 40, 60])
        f.attrs["nsrc"] = 1
        f.attrs["nrx"] = 2
        rxs = f.create_group("rxs")
        rx1 = rxs.create_group("rx1")
        rx1.attrs["Name"] = np.bytes_(b"R")
        rx1.attrs["Position"] = np.array([2e-5, 2e-5, 5e-5])
        for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            rx1.create_dataset(c, data=(Er if c == comp else np.zeros(n)))
        rx2 = rxs.create_group("rx2")
        rx2.attrs["Name"] = np.bytes_(b"T")
        rx2.attrs["Position"] = np.array([2e-5, 2e-5, 1e-5])
        for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            rx2.create_dataset(c, data=(Et if c == comp else np.zeros(n)))


@unittest.skipUnless(_HAVE, "numpy + h5py required for the parser fixtures")
class TestParseAntenna(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "model.out")
        self.f_res = write_antenna_fixture(self.out)
        self.job = _antenna_job()
        self.geo = registry.get("antenna_unit_cell").build(self.job["params"], self.job)

    def tearDown(self):
        self.tmp.cleanup()

    def test_contract_shape(self):
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        npts = self.job["frequency"]["points"]
        self.assertEqual(len(r["freqs_hz"]), npts)
        self.assertEqual(len(r["transmission"]), npts)
        self.assertEqual(len(r["phase_rad"]), npts)
        self.assertEqual(len(r["s_params"]["s11_re"]), npts)
        self.assertEqual(len(r["s_params"]["s11_im"]), npts)
        self.assertTrue(r["converged"])
        self.assertGreater(r["mesh_cells"], 0)

    def test_s11_magnitude_bounded(self):
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        for re, im in zip(r["s_params"]["s11_re"], r["s_params"]["s11_im"]):
            self.assertLessEqual(math.hypot(re, im), 1.0 + 1e-6)

    def test_resonance_dip_near_f_res(self):
        # The fitted return loss minimum (most negative dB) should land near the
        # fixture's injected resonance, proving the FFT + S11 mapping is real.
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        fom = registry.get("antenna_unit_cell").extract_fom(r, self.job["params"])
        self.assertIsNotNone(fom["resonance_hz"])
        self.assertIsNotNone(fom["return_loss_db"])
        # within +/-1 GHz of the 5 GHz injected dip.
        self.assertLess(abs(fom["resonance_hz"] - self.f_res), 1.0e9)
        # a genuine dip is well below 0 dB.
        self.assertLess(fom["return_loss_db"], -0.5)

    def test_radiation_pattern_present_when_requested(self):
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        self.assertIn("radiation_pattern", r)
        self.assertEqual(len(r["radiation_pattern"]["theta"]),
                         len(r["radiation_pattern"]["gain_db"]))


@unittest.skipUnless(_HAVE, "numpy + h5py required for the parser fixtures")
class TestParseMetasurface(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "model.out")
        self.job = _metasurface_job()
        self.geo = registry.get("metasurface_unit_cell").build(self.job["params"], self.job)
        # write the fixture into the component the source polarization drives.
        write_metasurface_fixture(self.out, comp=gprmax_io._field_component(self.geo))

    def tearDown(self):
        self.tmp.cleanup()

    def test_contract_shape(self):
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        npts = self.job["frequency"]["points"]
        self.assertEqual(len(r["freqs_hz"]), npts)
        self.assertEqual(len(r["transmission"]), npts)
        self.assertEqual(len(r["reflection"]), npts)
        self.assertEqual(len(r["phase_rad"]), npts)

    def test_transmission_dominant(self):
        # Fixture has 0.9 transmitted vs 0.3 reflected -> transmission > reflection.
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        mid = len(r["transmission"]) // 2
        self.assertGreater(r["transmission"][mid], r["reflection"][mid])

    def test_magnitudes_bounded(self):
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        for t, rf in zip(r["transmission"], r["reflection"]):
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 1.0)
            self.assertGreaterEqual(rf, 0.0)
            self.assertLessEqual(rf, 1.0)

    def test_fom_phase_extracted(self):
        r = gprmax_io.parse_output(self.out, self.job, self.geo)
        fom = registry.get("metasurface_unit_cell").extract_fom(r, self.job["params"])
        self.assertIsNotNone(fom["transmission_phase_rad"])
        self.assertIsNotNone(fom["transmission_mag"])


@unittest.skipUnless(_HAVE, "numpy + h5py required")
class TestFftHelper(unittest.TestCase):

    def test_fft_at_picks_correct_bin(self):
        # A pure tone at f0 should peak at the f0 sample.
        dt = 1e-12
        n = 8192
        f0 = 1e11
        t = np.arange(n) * dt
        sig = np.sin(2 * math.pi * f0 * t)
        targets = [0.5e11, 1.0e11, 1.5e11]
        spec = gprmax_io._fft_at(sig, dt, targets, np)
        mags = np.abs(spec)
        self.assertEqual(int(np.argmax(mags)), 1)  # the 1e11 target dominates.


if __name__ == "__main__":
    unittest.main()
