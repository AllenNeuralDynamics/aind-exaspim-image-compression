"""Tests for the validation-metrics module."""

import unittest

import numpy as np

from scipy import ndimage

from aind_exaspim_image_compression.machine_learning.metrics import (
    DEFAULT_CHECKPOINT_WEIGHTS,
    checkpoint_score,
    evaluate_example,
    false_bright_rate,
    foreground_background_mae,
    highfreq_energy_fraction,
    local_autocorr,
    make_foreground_mask,
    mip_max_error,
    patch_has_incoherent_segment,
)


def _smooth_blob(shape=(48, 48, 48), lo=(8, 8, 8), hi=(40, 40, 40),
                 amp=800.0, sigma=2.0):
    """A bright, spatially smooth (PSF-like) blob -- stands in for a neurite."""
    v = np.zeros(shape, dtype=np.float32)
    v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = amp
    return ndimage.gaussian_filter(v, sigma)


def _salt_pepper(shape=(48, 48, 48), lo=(8, 8, 8), hi=(40, 40, 40),
                 amp=900.0, rate=0.4, seed=0):
    """A bright, spatially incoherent salt-and-pepper block -- the artifact."""
    rng = np.random.default_rng(seed)
    v = np.zeros(shape, dtype=np.float32)
    region = np.zeros(shape, dtype=bool)
    region[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    v[(rng.random(shape) < rate) & region] = amp
    return v, region


class CoherenceGateTest(unittest.TestCase):
    """Tests for spatial-coherence artifact detection (patch rejection)."""

    def test_metrics_separate_signal_from_noise(self):
        """Smooth signal has high lag-2 autocorr and low HF energy; noise the
        opposite."""
        blob = _smooth_blob()
        sp, region = _salt_pepper()
        blob_mask = blob > 50
        self.assertGreater(local_autocorr(blob, blob_mask, lag=2), 0.5)
        self.assertLess(highfreq_energy_fraction(blob, blob_mask), 0.35)
        self.assertLess(local_autocorr(sp, region, lag=2), 0.4)
        self.assertGreater(highfreq_energy_fraction(sp, region), 0.35)

    def test_flags_patch_with_incoherent_segment(self):
        """A patch whose label is salt-and-pepper is flagged for rejection."""
        sp, region = _salt_pepper()
        labels = np.zeros(region.shape, dtype=np.uint64)
        labels[region] = 22
        self.assertTrue(patch_has_incoherent_segment(labels, sp))

    def test_keeps_patch_with_coherent_segment(self):
        """A patch whose label is a smooth blob is not flagged."""
        blob = _smooth_blob()
        labels = np.zeros(blob.shape, dtype=np.uint64)
        labels[blob > 50] = 11
        self.assertFalse(patch_has_incoherent_segment(labels, blob))

    def test_empty_labels_not_flagged(self):
        """A patch with no labels is never flagged."""
        labels = np.zeros((48, 48, 48), dtype=np.uint64)
        raw = np.zeros((48, 48, 48), dtype=np.float32)
        self.assertFalse(patch_has_incoherent_segment(labels, raw))

    def test_small_incoherent_segments_ignored(self):
        """A sub-min_segment_voxels noise speck does not trigger rejection."""
        sp, region = _salt_pepper(lo=(20, 20, 20), hi=(23, 23, 23))  # 27 vox
        labels = np.zeros(region.shape, dtype=np.uint64)
        labels[region] = 7
        self.assertFalse(
            patch_has_incoherent_segment(labels, sp, min_segment_voxels=50))


class MaskTest(unittest.TestCase):
    """Tests for make_foreground_mask."""

    def test_flags_bright_block(self):
        """A bright block is flagged foreground; flat background is not."""
        raw = np.zeros((16, 16, 16), dtype=np.float32)
        raw[6:10, 6:10, 6:10] = 5000
        mask = make_foreground_mask(raw, k=6.0, dilate=1)
        self.assertTrue(mask[7, 7, 7])
        self.assertFalse(mask[0, 0, 0])
        self.assertGreaterEqual(mask.sum(), 64)  # >= block, dilation adds more


class MetricTest(unittest.TestCase):
    """Tests for the individual metric functions."""

    def test_foreground_background_mae(self):
        """MAE is split correctly by the mask."""
        pred = np.array([[10.0, 20.0]], dtype=np.float64)
        ref = np.array([[0.0, 0.0]], dtype=np.float64)
        fg = np.array([[True, False]])
        fg_mae, bg_mae = foreground_background_mae(pred, ref, fg)
        self.assertAlmostEqual(fg_mae, 10.0)
        self.assertAlmostEqual(bg_mae, 20.0)

    def test_mip_max_error(self):
        """MIP-max error is the absolute difference of maxima."""
        pred = np.array([1.0, 900.0])
        raw = np.array([0.0, 1000.0])
        self.assertAlmostEqual(mip_max_error(pred, raw), 100.0)

    def test_false_bright_rate(self):
        """Background voxels the model brightened are counted."""
        raw = np.zeros((10,), dtype=np.float64)
        raw[0] = 5000.0
        fg = np.zeros((10,), dtype=bool)
        fg[0] = True
        pred = np.zeros((10,), dtype=np.float64)
        pred[1] = 5000.0  # one background voxel hallucinated bright
        self.assertAlmostEqual(false_bright_rate(pred, raw, fg), 1.0 / 9.0)


class EvaluateTest(unittest.TestCase):
    """Tests for evaluate_example and checkpoint_score."""

    def test_evaluate_example_keys_and_preservation(self):
        """Perfect prediction preserves the bright tail (ratio ~ 1)."""
        raw = np.zeros((16, 16, 16), dtype=np.float32)
        raw[4:12, 4:12, 4:12] = 60000  # >0.1% of voxels, so p99.9 is bright
        fg = make_foreground_mask(raw)
        metrics = evaluate_example(raw, raw, raw, fg)
        for key in ("fg_mae", "bg_mae", "top_pct_error",
                    "top_pct_preservation", "mip_max_error",
                    "false_bright_rate"):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(metrics["fg_mae"], 0.0)
        self.assertAlmostEqual(metrics["mip_max_error"], 0.0)
        self.assertAlmostEqual(metrics["top_pct_preservation"], 1.0, places=5)

    def test_attenuation_lowers_preservation(self):
        """Halving the bright signal drops preservation below 1."""
        raw = np.zeros((16, 16, 16), dtype=np.float32)
        raw[4:12, 4:12, 4:12] = 60000
        fg = make_foreground_mask(raw)
        pred = raw * 0.5
        metrics = evaluate_example(pred, raw, raw, fg)
        self.assertLess(metrics["top_pct_preservation"], 1.0)
        self.assertGreater(metrics["mip_max_error"], 0.0)

    def test_checkpoint_score_default_and_cratio(self):
        """Default score ignores cratio; a positive weight rewards it."""
        metrics = {"fg_mae": 10.0, "bg_mae": 5.0, "top_pct_error": 20.0}
        base = checkpoint_score(metrics, cratio=3.0)
        expected = 1.0 * 10.0 + 0.2 * 5.0 + 0.5 * 20.0
        self.assertAlmostEqual(base, expected)
        self.assertEqual(DEFAULT_CHECKPOINT_WEIGHTS["cratio"], 0.0)

        weights = dict(DEFAULT_CHECKPOINT_WEIGHTS, cratio=1.0)
        with_cratio = checkpoint_score(metrics, cratio=3.0, weights=weights)
        self.assertAlmostEqual(with_cratio, expected - 3.0)


if __name__ == "__main__":
    unittest.main()
