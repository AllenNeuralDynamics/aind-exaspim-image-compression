"""
Dynamic-range-preserving intensity transforms for exaSPIM denoising and
compression.

Provides fixed, stateless forward/inverse transforms that map raw uint16
counts to a bounded, network-friendly domain and back. Two families are
implemented:

    * ``AsinhTransform``   - HDR-style asinh compression (log-like tail).
    * ``AnscombeTransform`` - generalized Anscombe variance-stabilizing
      transform for Poisson-Gaussian noise (sqrt-like tail).

The same transform object is meant to be used identically during training,
validation, and inference. Neither transform applies a hard brightness clip
below the physical sensor maximum, so bright structure keeps a distinct,
invertible value instead of being flattened by a percentile clip.

"""

import numpy as np


class IntensityTransform:
    """
    Abstract base class for count <-> normalized intensity transforms.
    """

    def forward(self, x):
        """
        Maps raw counts to the normalized (approximately [0, 1]) domain.

        Parameters
        ----------
        x : numpy.ndarray
            Image in raw count units.

        Returns
        -------
        numpy.ndarray
            Image in the normalized domain.
        """
        raise NotImplementedError

    def inverse(self, y):
        """
        Maps normalized values back to raw uint16 counts.

        Parameters
        ----------
        y : numpy.ndarray
            Image in the normalized domain.

        Returns
        -------
        numpy.ndarray
            Image in raw count units, clipped to the physical range.
        """
        raise NotImplementedError

    def inverse_float(self, y):
        """Maps normalized values to unclipped floating-point counts."""
        raise NotImplementedError


class AsinhTransform(IntensityTransform):
    """
    HDR-style asinh intensity transform.

    The transform is approximately linear for ``(x - offset) << scale`` and
    approximately logarithmic for ``(x - offset) >> scale``, so it is
    monotonic and invertible over the whole range with no plateau.

    ``scale`` is the dynamic-range knob: larger values stay linear longer
    (more faithful absolute range, less headroom); smaller values compress
    the bright tail harder (more headroom).

    The normalized output is only *approximately* [0, 1]: sub-background
    voxels (``x < offset``) map to small negative values. This is intentional
    (it preserves noise-floor symmetry) and harmless downstream. The only
    hard bound is the physical clamp applied in ``inverse``.

    Attributes
    ----------
    offset : float
        Background / black-point in counts.
    scale : float
        Count scale of the linear-to-log knee.
    max_count : float
        Physical sensor maximum used as the normalization reference.
    """

    def __init__(self, offset=0.0, scale=32.0, max_count=65535.0):
        """
        Instantiates an AsinhTransform.

        Parameters
        ----------
        offset : float, optional
            Background / black-point in counts. Default is 0.0.
        scale : float, optional
            Count scale of the linear-to-log knee. Default is 32.0.
        max_count : float, optional
            Physical sensor maximum used as the normalization reference.
            Default is 65535.0.
        """
        self.offset = float(offset)
        self.scale = float(scale)
        self.max_count = float(max_count)
        self._norm = float(
            np.arcsinh((self.max_count - self.offset) / self.scale)
        )

    def forward(self, x):
        """
        Maps raw counts to the normalized asinh domain.

        Parameters
        ----------
        x : numpy.ndarray
            Image in raw count units.

        Returns
        -------
        numpy.ndarray
            Normalized image (approximately [0, 1]), float32.
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.arcsinh((x - self.offset) / self.scale) / self._norm
        return y.astype(np.float32)

    def inverse_float(self, y):
        """Maps normalized asinh values to floating-point counts."""
        y = np.asarray(y, dtype=np.float32)
        return self.offset + self.scale * np.sinh(y * self._norm)

    def inverse(self, y):
        """
        Maps normalized asinh values back to raw uint16 counts.

        Parameters
        ----------
        y : numpy.ndarray
            Image in the normalized asinh domain.

        Returns
        -------
        numpy.ndarray
            Image in raw counts, clipped to [0, max_count], uint16.
        """
        counts = self.inverse_float(y)
        counts = np.clip(counts, 0, self.max_count)
        return np.rint(counts).astype(np.uint16)


class AnscombeTransform(IntensityTransform):
    """
    Generalized Anscombe variance-stabilizing transform.

    Models the data as ``x = gain * Poisson + Normal(offset, read_noise^2)``
    (Makitalo & Foi). The transform is sqrt-like, so it compresses the bright
    tail more gently than asinh while making the noise approximately
    homoscedastic. It reduces to the standard Anscombe transform
    ``2 * sqrt(x + 3/8)`` when ``gain=1``, ``read_noise=0``, ``offset=0``.

    The inverse is a closed form whose constant depends on
    ``unbiased_inverse``: the algebraic inverse (3/8) exactly round-trips the
    forward transform, while the asymptotically unbiased inverse (1/8) is
    appropriate for inverting denoised (expectation) values and therefore
    does not round-trip exactly.

    Attributes
    ----------
    gain : float
        Detector gain in counts per photo-electron.
    read_noise : float
        Gaussian read-noise standard deviation in counts.
    offset : float
        Dark / pedestal offset in counts.
    max_count : float
        Physical sensor maximum used as the normalization reference.
    unbiased_inverse : bool
        Whether the inverse uses the asymptotically unbiased constant.
    """

    def __init__(
        self,
        gain=1.0,
        read_noise=0.0,
        offset=0.0,
        max_count=65535.0,
        unbiased_inverse=True,
    ):
        """
        Instantiates an AnscombeTransform.

        Parameters
        ----------
        gain : float, optional
            Detector gain in counts per photo-electron. Default is 1.0.
        read_noise : float, optional
            Gaussian read-noise standard deviation in counts. Default is 0.0.
        offset : float, optional
            Dark / pedestal offset in counts. Default is 0.0.
        max_count : float, optional
            Physical sensor maximum used as the normalization reference.
            Default is 65535.0.
        unbiased_inverse : bool, optional
            If True, use the asymptotically unbiased inverse (constant 1/8),
            appropriate for inverting denoised values. If False, use the
            exact algebraic inverse (constant 3/8), which round-trips the
            forward transform. Default is True.
        """
        self.gain = float(gain)
        self.read_noise = float(read_noise)
        self.offset = float(offset)
        self.max_count = float(max_count)
        self.unbiased_inverse = bool(unbiased_inverse)
        self._c_inv = 1.0 / 8.0 if unbiased_inverse else 3.0 / 8.0
        self._norm = float(
            self._gat(np.asarray(self.max_count, dtype=np.float32))
        )

    def _gat(self, x):
        """
        Evaluates the unnormalized generalized Anscombe transform.

        Parameters
        ----------
        x : numpy.ndarray
            Image in raw count units.

        Returns
        -------
        numpy.ndarray
            Variance-stabilized values (before normalization).
        """
        arg = (
            self.gain * (x - self.offset)
            + (3.0 / 8.0) * self.gain ** 2
            + self.read_noise ** 2
        )
        return (2.0 / self.gain) * np.sqrt(np.maximum(arg, 0.0))

    def forward(self, x):
        """
        Maps raw counts to the normalized Anscombe domain.

        Parameters
        ----------
        x : numpy.ndarray
            Image in raw count units.

        Returns
        -------
        numpy.ndarray
            Normalized image (approximately [0, 1]), float32.
        """
        gat = self._gat(np.asarray(x, dtype=np.float32))
        return (gat / self._norm).astype(np.float32)

    def inverse_float(self, y):
        """Maps normalized Anscombe values to floating-point counts."""
        d = np.clip(np.asarray(y, dtype=np.float32), 0.0, None) * self._norm
        arg = (d * self.gain / 2.0) ** 2
        return self.offset + (
            arg - self._c_inv * self.gain ** 2 - self.read_noise ** 2
        ) / self.gain

    def inverse(self, y):
        """
        Maps normalized Anscombe values back to raw uint16 counts.

        Parameters
        ----------
        y : numpy.ndarray
            Image in the normalized Anscombe domain.

        Returns
        -------
        numpy.ndarray
            Image in raw counts, clipped to [0, max_count], uint16.
        """
        counts = self.inverse_float(y)
        counts = np.clip(counts, 0, self.max_count)
        return np.rint(counts).astype(np.uint16)


class LinearClipTransform(IntensityTransform):
    """
    Linear normalization with a hard brightness clip.

    Provided as a fixed, stateless baseline for A/B comparison against the
    compressive transforms. It reproduces the original normalize-and-clip
    behavior (with globally-frozen ``mn``/``mx`` instead of per-patch
    percentiles), which flattens the bright tail above ``clip`` into a
    non-invertible plateau. It is the thing the compressive transforms are
    meant to beat, not a recommended default.

    Attributes
    ----------
    mn : float
        Lower normalization reference in counts (maps to 0).
    mx : float
        Upper normalization reference in counts (maps to 1).
    clip : float
        Upper bound applied in the normalized domain.
    max_count : float
        Physical sensor maximum used to clamp the inverse.
    """

    def __init__(self, mn=0.0, mx=1000.0, clip=8.0, max_count=65535.0):
        """
        Instantiates a LinearClipTransform.

        Parameters
        ----------
        mn : float, optional
            Lower normalization reference in counts. Default is 0.0.
        mx : float, optional
            Upper normalization reference in counts. Default is 1000.0.
        clip : float, optional
            Upper bound applied in the normalized domain. Default is 8.0.
        max_count : float, optional
            Physical sensor maximum used to clamp the inverse. Default is
            65535.0.
        """
        self.mn = float(mn)
        self.mx = float(mx)
        self.clip = float(clip)
        self.max_count = float(max_count)

    def forward(self, x):
        """
        Maps raw counts to the normalized, clipped domain.

        Parameters
        ----------
        x : numpy.ndarray
            Image in raw count units.

        Returns
        -------
        numpy.ndarray
            Normalized image clipped to [0, clip], float32.
        """
        x = np.asarray(x, dtype=np.float32)
        y = (x - self.mn) / (self.mx - self.mn + 1e-8)
        return np.clip(y, 0.0, self.clip).astype(np.float32)

    def inverse_float(self, y):
        """Maps normalized linear values to floating-point counts."""
        y = np.asarray(y, dtype=np.float32)
        return y * (self.mx - self.mn) + self.mn

    def inverse(self, y):
        """
        Maps normalized values back to raw uint16 counts.

        Parameters
        ----------
        y : numpy.ndarray
            Image in the normalized domain.

        Returns
        -------
        numpy.ndarray
            Image in raw counts, clipped to [0, max_count], uint16.
        """
        counts = self.inverse_float(y)
        counts = np.clip(counts, 0, self.max_count)
        return np.rint(counts).astype(np.uint16)


class OffsetTransform(IntensityTransform):
    """Applies a raw-count offset around a frozen trained transform.

    This exact composition is used for inference on images that still contain
    their background pedestal. It deliberately leaves the base transform's
    normalization constants unchanged::

        forward(x) = base.forward(x - offset)
        inverse(y) = base.inverse_float(y) + offset

    Changing an AsinhTransform or AnscombeTransform's own ``offset`` parameter
    would also change its normalization denominator and therefore would not
    reproduce the mapping used for offset-subtracted training patches.
    """

    def __init__(self, base_transform, offset=0.0):
        self.base_transform = base_transform
        self.offset = float(offset)
        self.max_count = float(base_transform.max_count)

    def __getattr__(self, name):
        """Expose non-offset parameters such as scale and gain from the base."""
        return getattr(self.base_transform, name)

    def forward(self, x):
        """Subtracts the pedestal, then applies the trained transform."""
        x = np.asarray(x, dtype=np.float32)
        return self.base_transform.forward(x - self.offset)

    def inverse_float(self, y):
        """Inverts through the trained transform and restores the pedestal."""
        return self.base_transform.inverse_float(y) + self.offset

    def inverse(self, y):
        """Returns pedestal-restored, physically clipped uint16 counts."""
        counts = self.inverse_float(y)
        counts = np.clip(counts, 0, self.max_count)
        return np.rint(counts).astype(np.uint16)


def estimate_offset(sample, percentile=1.0, ignore_zeros=True):
    """
    Estimates a robust background / black-point (counts).

    Parameters
    ----------
    sample : numpy.ndarray
        Sample of raw counts (e.g., a coarse multiscale level or a volume).
    percentile : float, optional
        Low percentile used as the background estimate. Default is 1.0.
    ignore_zeros : bool, optional
        If True, exclude exactly-zero voxels so that zero-padding outside the
        imaged volume does not drag the estimate to 0. Default is True.

    Returns
    -------
    float
        Estimated background offset in counts.
    """
    sample = np.asarray(sample, dtype=np.float32).reshape(-1)
    if ignore_zeros:
        nonzero = sample[sample > 0]
        if nonzero.size:
            sample = nonzero
    return float(np.percentile(sample, percentile))


def build_transform(cfg):
    """
    Builds an intensity transform from a config dict.

    Params are treated as frozen constants; any data-calibrated value must
    already be baked into ``cfg`` (see ``calibrate_transform``) so that
    training and inference construct the identical transform. The originating
    (frozen) config is stamped onto the returned instance as ``.cfg`` so it
    can be serialized alongside a model checkpoint.

    Parameters
    ----------
    cfg : dict
        Config of the form ``{"kind": "asinh" | "anscombe" | "linear",
        "params": {...}}``. An offset composition is represented as
        ``{"kind": "offset", "base": <transform cfg>, "params": {...}}``.

    Returns
    -------
    IntensityTransform
        The constructed transform.

    Raises
    ------
    ValueError
        If ``cfg["kind"]`` is not a recognized transform kind.
    """
    kind = cfg["kind"]
    params = cfg.get("params", {})
    if kind == "asinh":
        transform = AsinhTransform(**params)
    elif kind == "anscombe":
        transform = AnscombeTransform(**params)
    elif kind == "linear":
        transform = LinearClipTransform(**params)
    elif kind == "offset":
        transform = OffsetTransform(build_transform(cfg["base"]), **params)
    else:
        raise ValueError(f"Unknown transform kind: {kind}")
    transform.cfg = {**cfg, "params": dict(params)}
    return transform


def calibrate_transform(cfg, sample):
    """
    Freezes data-driven params into a transform config, once, globally.

    Only the black-point ``offset`` is calibrated, from a low percentile of
    the sample; the scale/knee (asinh) and gain/read_noise (Anscombe) are not
    taken from high signal percentiles. The input ``cfg`` is not mutated; the
    returned cfg is what should be serialized with the model and reused
    verbatim at inference.

    Parameters
    ----------
    cfg : dict
        Transform config, optionally containing a ``"calibrate"`` block of
        the form ``{"offset": bool, "offset_percentile": float}``.
    sample : numpy.ndarray
        Representative sample of raw counts used for calibration.

    Returns
    -------
    dict
        A new config with calibrated params frozen in.
    """
    cfg = {**cfg, "params": dict(cfg.get("params", {}))}
    calib = cfg.get("calibrate", {})
    if calib.get("offset", False):
        cfg["params"]["offset"] = estimate_offset(
            sample, percentile=calib.get("offset_percentile", 1.0)
        )
    return cfg


def with_offset(transform, offset):
    """
    Composes a raw-count background offset around a trained transform.

    Used at inference when training patches had their per-brain offsets
    subtracted before the frozen transform. The returned mapping is exactly
    ``transform.forward(x - offset)``; the inverse adds the offset back after
    applying the frozen inverse. In particular, this does not alter the asinh
    or Anscombe normalization denominator.

    Parameters
    ----------
    transform : IntensityTransform
        A transform built via ``build_transform`` (so it carries ``.cfg``).
    offset : float
        Background offset in counts.

    Returns
    -------
    IntensityTransform
        A new transform with the given offset.
    """
    if isinstance(transform, OffsetTransform):
        transform = transform.base_transform
    cfg = getattr(transform, "cfg", None)
    if cfg is None:
        raise ValueError(
            "transform has no cfg; construct it via build_transform"
        )
    offset = float(offset)
    if cfg["kind"] == "linear":
        # Applying the per-volume offset before the trained linear transform,
        # ``base.forward(x - offset)``, is equivalent to shifting both linear
        # bounds.  Shifting both also makes inverse() restore the offset in the
        # returned raw counts.  LinearClipTransform deliberately has no
        # ``offset`` constructor argument.
        params = dict(cfg.get("params", {}))
        params["mn"] = float(transform.mn) + offset
        params["mx"] = float(transform.mx) + offset
        return build_transform({**cfg, "params": params})
    return build_transform(
        {
            "kind": "offset",
            "base": cfg,
            "params": {"offset": offset},
        }
    )
