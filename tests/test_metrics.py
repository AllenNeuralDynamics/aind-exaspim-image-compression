"""Tests for the validation-metrics module."""

import unittest

import numpy as np

from scipy import ndimage

from aind_exaspim_image_compression.machine_learning.metrics import (
    DEFAULT_CHECKPOINT_WEIGHTS,
    aggregate_stratified_metrics,
    checkpoint_score,
    evaluate_example,
    evaluate_stratified_example,
    false_bright_rate,
    foreground_background_mae,
    highfreq_energy_fraction,
    local_autocorr,
    make_foreground_mask,
    make_skeleton_mask,
    mip_max_error,
    patch_has_incoherent_segment,
    select_checkpoint,
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

    def test_skeleton_mask_marks_nodes_without_filling_edges(self):
        """Skeleton masks mark supplied nodes without interpolating gaps."""
        points = np.array([[1, 1, 1], [3, 3, 3]], dtype=np.int32)
        mask = make_skeleton_mask(
            points,
            start=(0, 0, 0),
            patch_shape=(5, 5, 5),
            dilate=0,
        )
        self.assertTrue(mask[1, 1, 1])
        self.assertTrue(mask[3, 3, 3])
        self.assertFalse(mask[2, 2, 2])


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


class StratifiedMetricTest(unittest.TestCase):
    """Tests provenance strata, saturation masks, and halo distance bands."""

    @staticmethod
    def _record(
        brain_id="bright",
        foreground=True,
        pred_scale=1.0,
        saturated=True,
    ):
        """Construct one synthetic bright or blank validation record."""
        shape = (41, 41, 41)
        raw = np.full(shape, 100.0, dtype=np.float32)
        grid = np.indices(shape).sum(axis=0)
        raw += np.where(grid % 2, 20.0, -20.0).astype(np.float32)
        fg = np.zeros(shape, dtype=bool)
        if foreground:
            fg[17:24, 17:24, 17:24] = True
            raw[fg] = 10000.0
        if saturated:
            raw[19:22, 19:22, 19:22] = 65535.0
        pred = ndimage.gaussian_filter(raw, sigma=1.0) * pred_scale
        target = ndimage.gaussian_filter(raw, sigma=1.0)
        return evaluate_stratified_example(
            pred,
            raw,
            target,
            fg,
            brain_id=brain_id,
            center=(100, 200, 300),
            offset=0,
            noise_params=np.array([1.8, 400], dtype=np.float32),
            saturation_margin=64,
        )

    def test_brightness_noise_saturation_and_provenance_strata(self):
        """Patch records retain source and fixed global stratum labels."""
        record = self._record()
        self.assertEqual(record["provenance"]["brain_id"], "bright")
        self.assertEqual(record["provenance"]["center"], [100, 200, 300])
        self.assertEqual(record["strata"]["foreground_presence"], "foreground")
        self.assertEqual(
            record["strata"]["foreground_intensity_bin"], "60k+"
        )
        self.assertEqual(record["strata"]["background_noise_bin"], "0-25")
        self.assertEqual(record["strata"]["saturation"], "saturated")
        self.assertEqual(record["patch_metrics"]["saturated_voxel_count"], 27)
        self.assertAlmostEqual(
            record["patch_metrics"]["calibrated_background_sigma"], 20.0
        )

    def test_halo_bands_cover_patch_and_report_cleanup_and_bias(self):
        """Distance bands report weighted noise, intensity, and edge metrics."""
        record = self._record()
        halo = record["halo"]
        self.assertEqual(set(halo), {"0-2", "2-5", "5-10", "10-20", "20+"})
        self.assertEqual(
            sum(values["voxel_count"] for values in halo.values()),
            41 ** 3,
        )
        self.assertGreater(halo["0-2"]["voxel_count"], 27)
        self.assertTrue(np.isfinite(halo["0-2"]["intensity_preservation"]))
        self.assertTrue(
            np.isfinite(halo["0-2"]["saturated_core_preservation"])
        )
        self.assertTrue(np.isfinite(halo["0-2"]["mean_intensity_bias"]))
        self.assertLess(
            halo["20+"]["output_hf_std"], halo["20+"]["input_hf_std"]
        )

    def test_unsaturated_patch_has_zero_count_halo_bands(self):
        """Unsaturated patches remain a stratum without fabricated halo data."""
        record = self._record(saturated=False)
        self.assertEqual(record["strata"]["saturation"], "unsaturated")
        self.assertTrue(
            all(values["voxel_count"] == 0 for values in record["halo"].values())
        )

    def test_aggregation_is_per_brain_and_excludes_blank_foreground_error(self):
        """Blank patches do not dilute foreground MAE with artificial zeros."""
        foreground = self._record(brain_id="a", pred_scale=0.9)
        blank = self._record(
            brain_id="b", foreground=False, saturated=False
        )
        self.assertGreater(foreground["legacy"]["fg_mae"], 0)
        self.assertEqual(blank["legacy"]["fg_mae"], 0)
        report = aggregate_stratified_metrics([foreground, blank])
        self.assertEqual(set(report["by_brain"]), {"a", "b"})
        self.assertEqual(set(report["halo_by_brain"]), {"a", "b"})
        self.assertAlmostEqual(
            report["overall"]["fg_mae"], foreground["legacy"]["fg_mae"]
        )
        self.assertEqual(report["overall"]["foreground_patch_count"], 1)
        self.assertEqual(
            report["by_foreground_presence"]["blank"]["foreground_patch_count"],
            0,
        )


class ConstrainedCheckpointSelectionTest(unittest.TestCase):
    """Tests constraint-first checkpoint validity and objective ranking."""

    @staticmethod
    def _report(bright_fg_mae=100.0, halo_bias=-20.0, bg_mae=5.0):
        """Return the structured subset needed by checkpoint selection."""
        return {
            "overall": {
                "fg_mae": bright_fg_mae,
                "bg_mae": bg_mae,
                "top_pct_error": 10.0,
            },
            "by_foreground_intensity": {
                "60k+": {"fg_mae": bright_fg_mae}
            },
            "halo_distance_bands": {
                "2-5": {"mean_intensity_bias": halo_bias}
            },
        }

    @staticmethod
    def _config(objective=None):
        """Require bright fidelity and absolute halo bias limits."""
        return {
            "mode": "constrained",
            "constraints": [
                {
                    "name": "bright_neurite_mae",
                    "path": [
                        "by_foreground_intensity",
                        "60k+",
                        "fg_mae",
                    ],
                    "max": 200.0,
                },
                {
                    "name": "halo_bias",
                    "path": [
                        "halo_distance_bands",
                        "2-5",
                        "mean_intensity_bias",
                    ],
                    "absolute": True,
                    "max": 50.0,
                },
            ],
            "objective": objective
            or {"path": "cratio", "direction": "maximize"},
        }

    def test_valid_checkpoint_is_ranked_by_compression_only(self):
        """Passing constraints produces lower scores for higher compression."""
        config = self._config()
        low = select_checkpoint(self._report(), 5.0, config)
        high = select_checkpoint(self._report(), 10.0, config)
        self.assertTrue(high["valid"])
        self.assertEqual(high["score"], -10.0)
        self.assertLess(high["score"], low["score"])
        self.assertTrue(all(row["passed"] for row in high["constraints"]))

    def test_legacy_mode_preserves_additive_checkpoint_score(self):
        """Existing experiments retain their exact legacy ranking path."""
        report = self._report()
        weights = dict(DEFAULT_CHECKPOINT_WEIGHTS, cratio=1.0)
        selection = select_checkpoint(
            report,
            cratio=3.0,
            config={"mode": "legacy"},
            legacy_weights=weights,
        )
        self.assertEqual(
            selection["score"],
            checkpoint_score(report["overall"], 3.0, weights),
        )

    def test_constraint_failure_makes_checkpoint_ineligible(self):
        """Compression cannot rescue excessive bright or halo error."""
        selection = select_checkpoint(
            self._report(bright_fg_mae=250, halo_bias=-75),
            cratio=1000,
            config=self._config(),
        )
        self.assertFalse(selection["valid"])
        self.assertTrue(np.isinf(selection["score"]))
        self.assertFalse(selection["constraints"][0]["passed"])
        self.assertFalse(selection["constraints"][1]["passed"])

    def test_report_metric_can_be_minimized_as_objective(self):
        """A valid checkpoint may rank by background cleanup instead."""
        config = self._config(
            {"path": ["overall", "bg_mae"], "direction": "minimize"}
        )
        selection = select_checkpoint(self._report(bg_mae=4.0), 10.0, config)
        self.assertTrue(selection["valid"])
        self.assertEqual(selection["score"], 4.0)

    def test_missing_required_metric_fails_but_optional_metric_passes(self):
        """Unavailable bright strata are explicit validity decisions."""
        report = self._report()
        del report["by_foreground_intensity"]["60k+"]
        required = select_checkpoint(report, 10.0, self._config())
        self.assertFalse(required["valid"])

        config = self._config()
        config["constraints"][0]["required"] = False
        optional = select_checkpoint(report, 10.0, config)
        self.assertTrue(optional["valid"])


if __name__ == "__main__":
    unittest.main()
