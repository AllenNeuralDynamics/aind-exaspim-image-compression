"""Tests for the intensity-transform module."""

import unittest

import numpy as np

from aind_exaspim_image_compression.machine_learning.transforms import (
    AnscombeTransform,
    AsinhTransform,
    IntensityTransform,
    build_transform,
    calibrate_transform,
    estimate_offset,
)


class AsinhTransformTest(unittest.TestCase):
    """Tests for AsinhTransform."""

    def test_round_trip(self):
        """Inverse recovers counts across the range, with no plateau."""
        t = AsinhTransform(offset=35, scale=32)
        vals = np.array([0, 100, 1000, 10000, 60000, 65535], dtype=np.float32)
        rec = t.inverse(t.forward(vals)).astype(np.float64)
        np.testing.assert_allclose(rec, vals, rtol=1e-2, atol=3)

    def test_no_bright_plateau(self):
        """Distinct bright counts map to distinct recovered counts."""
        t = AsinhTransform(offset=35, scale=32)
        vals = np.array([2500, 10000, 60000], dtype=np.float32)
        rec = t.inverse(t.forward(vals)).astype(np.float64)
        self.assertTrue(np.all(np.diff(rec) > 1000))

    def test_monotonic(self):
        """Forward is strictly increasing (no plateau)."""
        t = AsinhTransform(offset=35, scale=32)
        xs = np.linspace(0, 65535, 500).astype(np.float32)
        ys = t.forward(xs)
        self.assertTrue(np.all(np.diff(ys) > 0))

    def test_bounded(self):
        """Max maps to 1, background near 0, sub-background negative."""
        t = AsinhTransform(offset=35, scale=32)
        self.assertAlmostEqual(
            float(t.forward(np.array(65535.0))), 1.0, places=4
        )
        self.assertLess(abs(float(t.forward(np.array(35.0)))), 0.05)
        self.assertLess(float(t.forward(np.array(0.0))), 0.0)


class AnscombeTransformTest(unittest.TestCase):
    """Tests for AnscombeTransform."""

    def test_round_trip_algebraic(self):
        """Algebraic inverse (3/8) round-trips the forward transform."""
        t = AnscombeTransform(
            gain=8, read_noise=5, offset=100, unbiased_inverse=False
        )
        vals = np.array([100, 500, 2000, 20000, 65535], dtype=np.float32)
        rec = t.inverse(t.forward(vals)).astype(np.float64)
        np.testing.assert_allclose(rec, vals, rtol=5e-3, atol=3)

    def test_unbiased_inverse_has_expected_bias(self):
        """Unbiased inverse exceeds algebraic by ~gain/4 (no exact id)."""
        t_alg = AnscombeTransform(gain=8, unbiased_inverse=False)
        t_unb = AnscombeTransform(gain=8, unbiased_inverse=True)
        vals = np.array([1000, 20000, 60000], dtype=np.float32)
        y = t_alg.forward(vals)
        rec_alg = t_alg.inverse(y).astype(np.float64)
        rec_unb = t_unb.inverse(y).astype(np.float64)
        np.testing.assert_allclose(rec_unb - rec_alg, 2.0, atol=1.0)

    def test_monotonic(self):
        """Forward is strictly increasing (no plateau)."""
        t = AnscombeTransform(gain=8, read_noise=5, offset=0)
        xs = np.linspace(0, 65535, 500).astype(np.float32)
        ys = t.forward(xs)
        self.assertTrue(np.all(np.diff(ys) > 0))

    def test_bounded(self):
        """Max maps to 1."""
        t = AnscombeTransform(gain=8, read_noise=5, offset=100)
        self.assertAlmostEqual(
            float(t.forward(np.array(65535.0))), 1.0, places=4
        )

    def test_reduces_to_standard_anscombe(self):
        """Default params match the standard 2*sqrt(x + 3/8)."""
        t = AnscombeTransform(gain=1, read_noise=0, offset=0)
        x = np.array([0, 1, 10, 100, 1000], dtype=np.float32)
        expected = 2.0 * np.sqrt(x + 3.0 / 8.0)
        np.testing.assert_allclose(t._gat(x), expected, rtol=1e-5)


class HelperTest(unittest.TestCase):
    """Tests for module-level helpers."""

    def test_estimate_offset(self):
        """Offset estimate matches the requested percentile."""
        sample = np.arange(0, 101, dtype=np.float32)
        self.assertAlmostEqual(estimate_offset(sample, percentile=0), 0.0)
        self.assertAlmostEqual(estimate_offset(sample, percentile=50), 50.0)
        self.assertAlmostEqual(estimate_offset(sample, percentile=100), 100.0)

    def test_build_transform(self):
        """Factory builds each kind and rejects unknown kinds."""
        t = build_transform({"kind": "asinh"})
        self.assertIsInstance(t, AsinhTransform)

        t = build_transform({"kind": "anscombe", "params": {"gain": 8}})
        self.assertIsInstance(t, AnscombeTransform)
        self.assertEqual(t.gain, 8.0)

        with self.assertRaises(ValueError):
            build_transform({"kind": "nope"})

    def test_calibrate_transform_sets_offset(self):
        """Calibration freezes the offset without mutating the input."""
        sample = np.arange(0, 1000, dtype=np.float32)
        cfg = {
            "kind": "asinh",
            "calibrate": {"offset": True, "offset_percentile": 10.0},
        }
        out = calibrate_transform(cfg, sample)
        self.assertAlmostEqual(
            out["params"]["offset"],
            float(np.percentile(sample, 10.0)),
            places=4,
        )
        self.assertNotIn("params", cfg)

    def test_calibrate_transform_noop(self):
        """Without a calibrate block, params pass through unchanged."""
        cfg = {"kind": "anscombe", "params": {"gain": 2}}
        out = calibrate_transform(cfg, np.zeros(10, dtype=np.float32))
        self.assertEqual(out["params"], {"gain": 2})

    def test_base_class_not_implemented(self):
        """The abstract base raises for both directions."""
        t = IntensityTransform()
        with self.assertRaises(NotImplementedError):
            t.forward(np.zeros(1))
        with self.assertRaises(NotImplementedError):
            t.inverse(np.zeros(1))


if __name__ == "__main__":
    unittest.main()
