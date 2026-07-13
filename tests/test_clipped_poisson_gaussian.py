"""Seeded regressions for the reference clipped noise estimator."""

import unittest

import numpy as np

from aind_exaspim_image_compression.machine_learning.clipped_poisson_gaussian import (  # noqa: E501
    estimate_clipped_poisson_gaussian,
)


def make_scene(shape=(768, 768), lo=-0.08, hi=1.10, seed=1):
    """Create the seeded clipped 2-D reference-demo scene."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    scene = lo + (hi - lo) * (xx / (width - 1))
    scene += 0.15 * np.sin(2 * np.pi * yy / 180.0)
    rng = np.random.default_rng(seed)
    for _ in range(6):
        center_y, center_x = rng.uniform(0, height), rng.uniform(0, width)
        radius = rng.uniform(60, 140)
        amplitude = rng.uniform(-0.25, 0.25)
        scene += amplitude * np.exp(
            -((yy - center_y) ** 2 + (xx - center_x) ** 2)
            / (2 * radius * radius)
        )
    return scene


def make_volume(shape=(96, 192, 192), lo=-0.08, hi=1.10, seed=3):
    """Create the seeded smooth full-3-D reference-demo scene."""
    depth, height, width = shape
    zz, yy, xx = np.mgrid[0:depth, 0:height, 0:width].astype(float)
    volume = lo + (hi - lo) * (xx / (width - 1))
    volume += 0.12 * np.sin(2 * np.pi * zz / 40.0)
    rng = np.random.default_rng(seed)
    for _ in range(8):
        center_z, center_y, center_x = (rng.uniform(0, size) for size in shape)
        radius = rng.uniform(25, 60)
        amplitude = rng.uniform(-0.25, 0.25)
        volume += amplitude * np.exp(
            -(
                (zz - center_z) ** 2
                + (yy - center_y) ** 2
                + (xx - center_x) ** 2
            )
            / (2 * radius * radius)
        )
    return volume


def make_video(n_frames=48, shape=(384, 384), speed=3, seed=7):
    """Create the seeded translating reference-demo video."""
    big = make_scene(shape=(shape[0], shape[1] + speed * n_frames), seed=seed)
    return np.stack(
        [
            big[:, speed * frame : speed * frame + shape[1]]  # noqa: E203
            for frame in range(n_frames)
        ]
    )


def synthesize(scene, a, b, seed=0):
    """Add seeded heteroscedastic Gaussian noise and sensor clipping."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.maximum(a * scene + b, 0.0))
    noisy = scene + sigma * rng.standard_normal(scene.shape)
    return np.clip(noisy, 0.0, 1.0)


class SeededReferenceRegressionTest(unittest.TestCase):
    """Checks the scenarios shipped with both reference demo files."""

    def test_2d_poisson_gaussian_and_pure_gaussian(self):
        """Two-dimensional fits recover mixed and pure Gaussian noise."""
        scene = make_scene(seed=10)
        estimate = estimate_clipped_poisson_gaussian(
            synthesize(scene, 4.0e-3, 1.0e-4, seed=20)
        )
        self.assertAlmostEqual(estimate.a, 4.0e-3, delta=4.0e-4)
        self.assertAlmostEqual(estimate.b, 1.0e-4, delta=2.5e-5)

        scene = make_scene(seed=13)
        estimate = estimate_clipped_poisson_gaussian(
            synthesize(scene, 0.0, 6.4e-4, seed=23)
        )
        self.assertLess(estimate.a, 1.0e-6)
        self.assertAlmostEqual(estimate.b, 6.4e-4, delta=3.2e-5)

    def test_full_volume_and_small_intercept_modes(self):
        """Full and slicewise 3-D fits recover both reference volume cases."""
        volume = synthesize(make_volume(), 4.0e-3, 1.0e-4, seed=42)
        estimate = estimate_clipped_poisson_gaussian(volume, mode="full")
        self.assertAlmostEqual(estimate.a, 4.0e-3, delta=2.0e-4)
        self.assertAlmostEqual(estimate.b, 1.0e-4, delta=1.5e-5)

        volume = synthesize(make_volume(seed=4), 2.5e-3, 2.5e-5, seed=43)
        for mode in ("full", "slicewise"):
            with self.subTest(mode=mode):
                estimate = estimate_clipped_poisson_gaussian(volume, mode=mode)
                self.assertAlmostEqual(estimate.a, 2.5e-3, delta=1.25e-4)
                self.assertAlmostEqual(estimate.b, 2.5e-5, delta=5.0e-6)

    def test_slicewise_mode_handles_translating_video(self):
        """Slicewise filtering avoids temporal-motion bias in the full mode."""
        video = synthesize(make_video(), 4.0e-3, 1.0e-4, seed=44)
        slicewise = estimate_clipped_poisson_gaussian(video, mode="slicewise")
        full = estimate_clipped_poisson_gaussian(video, mode="full")
        self.assertAlmostEqual(slicewise.a, 4.0e-3, delta=2.0e-4)
        self.assertAlmostEqual(slicewise.b, 1.0e-4, delta=1.0e-5)
        self.assertLess(abs(slicewise.b - 1.0e-4), abs(full.b - 1.0e-4))


if __name__ == "__main__":
    unittest.main()
