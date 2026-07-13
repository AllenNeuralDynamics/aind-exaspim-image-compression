"""Estimate clipped Poisson--Gaussian noise from one image or volume.

This is a numerical reimplementation of the algorithm behind Foi's MATLAB
``function_ClipPoisGaus_stdEst2D.p`` and ``ml_fun_ClipPoisGaus.p`` files,
based on the method described in:

    A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian,
    "Practical Poissonian-Gaussian noise modeling and fitting for
    single-image raw-data", IEEE Trans. Image Processing, 17(10), 2008.

The normalized noise model is ``sigma**2(y) = a*y + b`` followed by clipping
to ``[0, 1]``. The estimator supports 2-D images, full 3-D filtering for
isotropic volumes, and slicewise 2-D filtering with pooled level sets for
videos or anisotropic stacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    convolve1d,
    uniform_filter,
)
from scipy.optimize import least_squares
from scipy.special import gammaln, ndtr

_DB2_LO = np.array(
    [
        0.48296291314453414,
        0.83651630373780772,
        0.22414386804201339,
        -0.12940952255126037,
    ]
)
_DB2_HI = np.array(
    [
        -0.12940952255126037,
        -0.22414386804201339,
        0.83651630373780772,
        -0.48296291314453414,
    ]
)


def _normal_pdf(x):
    """Return the standard-normal probability density elementwise."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def clipped_normal_moments(mu, sigma, lo=0.0, hi=1.0):
    """Return the exact mean and standard deviation of a clipped normal.

    The unclipped variable is distributed as ``N(mu, sigma**2)`` and the
    returned moments describe ``min(hi, max(lo, X))``. Inputs are broadcast
    using NumPy's ordinary array rules.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)

    alpha = (lo - mu) / sigma
    beta = (hi - mu) / sigma
    phi_a_cdf, phi_b_cdf = ndtr(alpha), ndtr(beta)
    phi_a, phi_b = _normal_pdf(alpha), _normal_pdf(beta)
    p_in = phi_b_cdf - phi_a_cdf

    mean = (
        lo * phi_a_cdf
        + hi * (1.0 - phi_b_cdf)
        + mu * p_in
        + sigma * (phi_a - phi_b)
    )

    second = (
        lo**2 * phi_a_cdf
        + hi**2 * (1.0 - phi_b_cdf)
        + (mu**2 + sigma**2) * p_in
        + 2.0 * mu * sigma * (phi_a - phi_b)
        + sigma**2 * (alpha * phi_a - beta * phi_b)
    )

    variance = np.maximum(second - mean**2, 0.0)
    return mean, np.sqrt(variance)


def _invert_clipped_mean(
    target_mean,
    a,
    b,
    lo=0.0,
    hi=1.0,
    y_lo=-1.0,
    y_hi=2.0,
    iters=60,
):
    """Solve for the unclipped mean that gives a target clipped mean."""
    target = np.asarray(target_mean, dtype=float)
    lo_y = np.full_like(target, y_lo)
    hi_y = np.full_like(target, y_hi)
    for _ in range(iters):
        mid = 0.5 * (lo_y + hi_y)
        sigma = np.sqrt(np.maximum(a * mid + b, 1e-12))
        mean, _ = clipped_normal_moments(mid, sigma, lo, hi)
        too_low = mean < target
        lo_y = np.where(too_low, mid, lo_y)
        hi_y = np.where(too_low, hi_y, mid)
    return 0.5 * (lo_y + hi_y)


def _wavelet_fields(z, axes=None):
    """Return local approximation and unit-norm db2 detail fields."""
    if axes is None:
        axes = tuple(range(z.ndim))
    lowpass = _DB2_LO / _DB2_LO.sum()
    highpass = _DB2_HI
    z_approximation = z
    z_detail = z
    for axis in axes:
        z_approximation = convolve1d(
            z_approximation, lowpass, axis=axis, mode="reflect"
        )
        z_detail = convolve1d(z_detail, highpass, axis=axis, mode="reflect")
    return z_approximation, z_detail


def _texture_operator(image, axes=None):
    """Return a smoothed gradient magnitude over the selected axes."""
    if axes is None:
        axes = tuple(range(image.ndim))
    derivative = np.array([-0.5, 0.0, 0.5])
    squared = np.zeros_like(image)
    for axis in axes:
        gradient = convolve1d(image, derivative, axis=axis, mode="reflect")
        squared += gradient * gradient
    magnitude = np.sqrt(squared)
    box = np.ones(5) / 5.0
    for axis in axes:
        magnitude = convolve1d(magnitude, box, axis=axis, mode="reflect")
    return magnitude


_TEXTURE_GAIN_CACHE = {}


def _texture_noise_gain(n_axes=2, seed=0):
    """Calibrate the texture operator on unit-variance white noise."""
    key = (n_axes, seed)
    if key not in _TEXTURE_GAIN_CACHE:
        rng = np.random.default_rng(seed)
        side = {1: 1 << 18, 2: 512, 3: 128}.get(
            n_axes, int(round(2e6 ** (1.0 / n_axes)))
        )
        white_noise = rng.standard_normal((side,) * n_axes)
        approximation, _ = _wavelet_fields(white_noise)
        _TEXTURE_GAIN_CACHE[key] = float(
            np.median(_texture_operator(approximation))
        )
    return _TEXTURE_GAIN_CACHE[key]


def _unbias_std_factor(n):
    """Return the normal-distribution sample-standard-deviation bias."""
    n = np.asarray(n, dtype=float)
    with np.errstate(all="ignore"):
        factor = np.sqrt(2.0 / (n - 1.0)) * np.exp(
            gammaln(n / 2.0) - gammaln((n - 1.0) / 2.0)
        )
    return np.where(n > 1, factor, 1.0)


@dataclass
class NoiseEstimate:
    """Clipped Poisson--Gaussian fit and retained level-set diagnostics.

    Parameters are expressed for data normalized to ``[0, 1]``. The
    unclipped conditional variance is ``a*y + b``.
    """

    a: float
    b: float
    level_means: np.ndarray = field(repr=False)
    level_stds: np.ndarray = field(repr=False)
    level_counts: np.ndarray = field(repr=False)
    model_stds: np.ndarray = field(repr=False)
    smooth_fraction: float = 0.0

    def sigma(self, y):
        """Return the unclipped noise standard deviation at intensity y."""
        return np.sqrt(np.maximum(self.a * np.asarray(y, float) + self.b, 0.0))

    def clipped_curve(self, y):
        """Return the clipped mean and standard deviation along the fit."""
        return clipped_normal_moments(y, self.sigma(y))


def estimate_clipped_poisson_gaussian(  # noqa: C901
    z,
    n_levels=60,
    min_samples_per_level=200,
    edge_tau=2.0,
    fit_loss="soft_l1",
    mode="full",
):
    """Estimate clipped Poisson--Gaussian parameters from one image.

    Parameters
    ----------
    z : numpy.ndarray
        Two- or three-dimensional raw sensor data normalized by its white
        level. Axis 0 is the slice or frame axis in slicewise mode.
    n_levels : int, optional
        Number of intensity level sets. Default is 60.
    min_samples_per_level : int, optional
        Minimum samples retained in each level set. Default is 200.
    edge_tau : float, optional
        Smooth-mask threshold in noise-response units. Default is 2.
    fit_loss : str, optional
        Robust loss passed to ``scipy.optimize.least_squares``. Default is
        ``"soft_l1"``.
    mode : {"full", "slicewise"}, optional
        Full N-D filtering or slicewise 2-D filtering with pooled level sets.

    Returns
    -------
    NoiseEstimate
        The normalized fit and retained level-set diagnostics.

    Raises
    ------
    ValueError
        If the input is not two- or three-dimensional or mode is invalid.
    RuntimeError
        If fewer than three usable level sets remain.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim not in (2, 3):
        raise ValueError("z must be a 2-D image or a 3-D volume/stack.")
    if mode not in ("full", "slicewise"):
        raise ValueError("mode must be 'full' or 'slicewise'.")
    if z.ndim == 3 and mode == "slicewise":
        axes = (1, 2)
    else:
        axes = tuple(range(z.ndim))

    z_approximation, z_detail = _wavelet_fields(z, axes)
    bin_box = 7
    box_size = tuple(bin_box if axis in axes else 1 for axis in range(z.ndim))
    z_bin = uniform_filter(z, size=box_size, mode="reflect")

    sigma0 = np.median(np.abs(z_detail - np.median(z_detail))) / 0.6745
    sigma0 = max(sigma0, 1e-8)
    texture = _texture_operator(z_approximation, axes)
    threshold = edge_tau * sigma0 * _texture_noise_gain(len(axes)) * 2.0
    smooth = texture < threshold
    structure_shape = tuple(5 if axis in axes else 1 for axis in range(z.ndim))
    smooth = binary_erosion(smooth, structure=np.ones(structure_shape))
    smooth_fraction = float(smooth.mean())
    if smooth.sum() < 10 * min_samples_per_level:
        cutoff = np.quantile(texture, 0.5)
        smooth = texture < cutoff
        smooth_fraction = float(smooth.mean())

    approximation = z_bin[smooth]
    detail = z_detail[smooth]

    edges = np.quantile(approximation, np.linspace(0.0, 1.0, n_levels + 1))
    edges = np.unique(edges)
    indices = np.clip(
        np.searchsorted(edges, approximation, side="right") - 1,
        0,
        len(edges) - 2,
    )

    means = []
    standard_deviations = []
    counts = []
    half_widths = []
    for index in range(len(edges) - 1):
        selected = indices == index
        count = int(selected.sum())
        if count < min_samples_per_level:
            continue
        selected_detail = detail[selected]
        standard_deviation = selected_detail.std(ddof=1) / _unbias_std_factor(
            count
        )
        if not np.isfinite(standard_deviation):
            continue
        means.append(float(approximation[selected].mean()))
        standard_deviations.append(float(standard_deviation))
        counts.append(count)
        half_widths.append(0.5 * float(edges[index + 1] - edges[index]))

    means = np.array(means)
    standard_deviations = np.array(standard_deviations)
    counts = np.array(counts, dtype=float)
    half_widths = np.array(half_widths)
    if len(means) < 3:
        raise RuntimeError(
            "Too few usable level sets; image may be too small or too flat."
        )

    margin = 3.0 * np.maximum(standard_deviations, 1e-6)
    inner = (means > margin) & (means < 1.0 - margin)
    if inner.sum() >= 3:
        weights = counts[inner]
        design = np.stack([means[inner], np.ones(inner.sum())], axis=1)
        solution, *_ = np.linalg.lstsq(
            design * np.sqrt(weights)[:, None],
            (standard_deviations[inner] ** 2) * np.sqrt(weights),
            rcond=None,
        )
        a0, b0 = solution
    else:
        a0 = 0.0
        b0 = float(np.median(standard_deviations) ** 2)
    a0 = max(a0, 0.0)
    b0 = max(b0, 1e-10)

    q2 = (1.0 / bin_box) ** len(axes)
    eta = 0.01
    y_grid = np.linspace(-0.5, 1.5, 1024)

    def _profile_nll_terms(a, b):
        """Profile each level set over the unknown unclipped intensity."""
        sigma_grid = np.sqrt(np.maximum(a * y_grid + b, 1e-14))
        mean_grid, std_grid = clipped_normal_moments(y_grid, sigma_grid)
        mean_variance = (
            (std_grid**2 * q2)[:, None] + (half_widths**2)[None, :] + 1e-16
        )
        std_variance = (
            (std_grid**2)[:, None] / (2.0 * counts[None, :])
            + (eta * std_grid[:, None]) ** 2
            + 1e-16
        )
        objective = (
            means[None, :] - mean_grid[:, None]
        ) ** 2 / mean_variance + (
            standard_deviations[None, :] - std_grid[:, None]
        ) ** 2 / std_variance
        locations = np.argmin(objective, axis=0)
        centers = np.clip(locations, 1, len(y_grid) - 2)
        columns = np.arange(objective.shape[1])
        f0 = objective[centers - 1, columns]
        f1 = objective[centers, columns]
        f2 = objective[centers + 1, columns]
        denominator = f0 - 2.0 * f1 + f2
        delta = np.where(
            np.abs(denominator) > 1e-30,
            0.5 * (f0 - f2) / np.maximum(denominator, 1e-30),
            0.0,
        )
        minimum = f1 - 0.25 * (f0 - f2) * np.clip(delta, -1.0, 1.0)
        return np.maximum(np.minimum(minimum, f1), 0.0), locations

    def residuals(parameters):
        """Return profiled level-set residuals for the optimizer."""
        nll, _ = _profile_nll_terms(*parameters)
        return np.sqrt(nll)

    result = least_squares(
        residuals,
        x0=[a0, b0],
        bounds=([0.0, -1e-4], [1.0, 1.0]),
        loss=fit_loss,
        f_scale=2.0,
        xtol=1e-14,
        ftol=1e-14,
    )
    a_hat, b_hat = result.x

    _, fitted_locations = _profile_nll_terms(a_hat, b_hat)
    y_fit = y_grid[fitted_locations]
    sigma_fit = np.sqrt(np.maximum(a_hat * y_fit + b_hat, 1e-12))
    _, model_standard_deviations = clipped_normal_moments(y_fit, sigma_fit)

    return NoiseEstimate(
        a=float(a_hat),
        b=float(b_hat),
        level_means=means,
        level_stds=standard_deviations,
        level_counts=counts,
        model_stds=model_standard_deviations,
        smooth_fraction=smooth_fraction,
    )


def plot_fit(estimate, ax=None, title=None):
    """Plot level-set estimates and fitted clipped/unclipped curves."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    sizes = 8 + 40 * estimate.level_counts / estimate.level_counts.max()
    ax.scatter(
        estimate.level_means,
        estimate.level_stds,
        s=sizes,
        alpha=0.6,
        label="level-set estimates",
        zorder=3,
    )

    intensity = np.linspace(-0.05, 1.05, 400)
    clipped_mean, clipped_std = estimate.clipped_curve(intensity)
    ax.plot(
        clipped_mean,
        clipped_std,
        "r-",
        lw=2,
        label=f"clipped fit: a={estimate.a:.3e}, b={estimate.b:.3e}",
    )
    interior = np.linspace(0.0, 1.0, 200)
    ax.plot(
        interior,
        estimate.sigma(interior),
        "k--",
        lw=1.2,
        label="unclipped $\\sqrt{ay+b}$",
    )

    ax.set_xlabel("local mean of clipped data")
    ax.set_ylabel("local std")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
