"""
Validation metrics for scoring neurite preservation vs. compression.

These operate in raw count space (after a transform's inverse) so they mean
the same thing regardless of which intensity transform is used. They split
voxels into foreground and background with a robust intensity mask and
measure, separately, whether bright signal is preserved (foreground, vs. the
raw counts) and whether background is cleaned like the BM4D teacher
(background, vs. the target counts). The foreground/background split uses a
caller-supplied mask -- the segmentation labels unioned with the traced
skeleton during training and validation (see make_segmentation_mask,
make_skeleton_mask), or the robust intensity threshold (make_foreground_mask)
when no annotations are available.

"""

import numpy as np
from scipy import ndimage


# Weights for the checkpoint-selection score. cratio defaults to 0.0 so
# selection is purely fidelity-driven; raise it to trade fidelity for
# compression once the operating point is chosen.
DEFAULT_CHECKPOINT_WEIGHTS = {
    "fg_mae": 1.0,
    "bg_mae": 0.2,
    "top_pct_error": 0.5,
    "cratio": 0.0,
}

LEGACY_METRIC_KEYS = (
    "fg_mae",
    "bg_mae",
    "top_pct_error",
    "top_pct_preservation",
    "mip_max_error",
    "false_bright_rate",
)

FOREGROUND_INTENSITY_BINS = (
    (0.0, 1000.0, "0-1k"),
    (1000.0, 5000.0, "1k-5k"),
    (5000.0, 20000.0, "5k-20k"),
    (20000.0, 60000.0, "20k-60k"),
    (60000.0, np.inf, "60k+"),
)

BACKGROUND_NOISE_BINS = (
    (0.0, 25.0, "0-25"),
    (25.0, 50.0, "25-50"),
    (50.0, 100.0, "50-100"),
    (100.0, 200.0, "100-200"),
    (200.0, np.inf, "200+"),
)

HALO_DISTANCE_BANDS = (
    (0.0, 2.0, "0-2"),
    (2.0, 5.0, "2-5"),
    (5.0, 10.0, "5-10"),
    (10.0, 20.0, "10-20"),
    (20.0, np.inf, "20+"),
)


def make_foreground_mask(raw, k=6.0, dilate=1):
    """
    Builds a robust foreground mask from image intensity.

    Uses a median + k * (robust sigma) threshold so it is insensitive to the
    bright tail, then dilates to include neurite boundaries.

    Parameters
    ----------
    raw : numpy.ndarray
        Image in raw count units.
    k : float, optional
        Threshold in robust standard deviations above the median. Default is
        6.0.
    dilate : int, optional
        Number of binary-dilation iterations. Default is 1.

    Returns
    -------
    numpy.ndarray
        Boolean foreground mask with the same shape as "raw".
    """
    raw = np.asarray(raw, dtype=np.float32)
    med = np.median(raw)
    mad = np.median(np.abs(raw - med)) + 1e-6
    sigma = 1.4826 * mad
    mask = raw > (med + k * sigma)
    if dilate > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilate)
    return mask


def local_autocorr(raw, mask, lag=2):
    """
    Mean spatial autocorrelation of "raw" at a given lag over masked voxels.

    Averages the Pearson correlation between each voxel and its +lag neighbor
    along every axis, restricted to pairs that both fall in "mask". Real
    neurites are PSF-blurred, so the signal stays correlated over several
    voxels; spatially incoherent noise decorrelates immediately.

    Lag 2 (the default) is the discriminating scale. Some bright, blocky
    processing artifacts are correlated at lag 1 (adjacent noisy voxels track
    each other) and so pass a lag-1 test, but their correlation collapses by
    lag 2, whereas real signal -- correlated across the ~2-3 voxel PSF -- stays
    high at lag 2. Measured separation: artifacts <= 0.30 at lag 2, real
    neurites >= 0.53.

    Parameters
    ----------
    raw : numpy.ndarray
        Image patch in counts.
    mask : numpy.ndarray
        Boolean mask selecting the voxels to score.
    lag : int, optional
        Neighbor offset in voxels. Default is 2.

    Returns
    -------
    float
        Mean autocorrelation in [-1, 1]. Returns 1.0 (maximally coherent) when
        it cannot be measured, so callers never reject a segment on an
        undefined score.
    """
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    vals = []
    for ax in range(raw.ndim):
        lo = [slice(None)] * raw.ndim
        hi = [slice(None)] * raw.ndim
        lo[ax] = slice(0, -lag)
        hi[ax] = slice(lag, None)
        sel = mask[tuple(lo)] & mask[tuple(hi)]
        if sel.sum() < 2:
            continue
        x = raw[tuple(lo)][sel]
        y = raw[tuple(hi)][sel]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        vals.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(vals)) if vals else 1.0


def highfreq_energy_fraction(raw, mask, smooth=None, smooth_sigma=1.0):
    """
    Fraction of the masked signal's variance that is high spatial frequency.

    Computes ``var(raw - gaussian_smooth(raw)) / var(raw)`` over the masked
    voxels. Salt-and-pepper noise puts almost all of its energy in the
    high-frequency residual (~0.6-0.8); smooth neurite signal puts almost none
    there (~0.0-0.25). Complements "local_autocorr" (the two are inversely
    related) so a segment can be required to fail both before it is rejected.

    Parameters
    ----------
    raw : numpy.ndarray
        Image patch in counts.
    mask : numpy.ndarray
        Boolean mask selecting the voxels to score.
    smooth : numpy.ndarray, optional
        Precomputed Gaussian-smoothed "raw" (reused across segments of a
        patch). When None it is computed from "raw" with "smooth_sigma".
    smooth_sigma : float, optional
        Gaussian sigma used when "smooth" is not supplied. Default is 1.0.

    Returns
    -------
    float
        High-frequency energy fraction, >= 0. Returns 0.0 (maximally coherent)
        when the masked variance is degenerate.
    """
    raw = np.asarray(raw, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if smooth is None:
        smooth = ndimage.gaussian_filter(raw, sigma=smooth_sigma)
    v = raw[mask]
    if v.var() < 1e-12:
        return 0.0
    hf = (raw - smooth)[mask]
    return float(hf.var() / v.var())


def make_segmentation_mask(labels, dilate=0):
    """
    Builds a foreground mask from segmentation labels.

    Foreground is the labeled neurites alone, so bright non-neuronal structures
    (noise, off-target label) are left for the BM4D teacher to denoise rather
    than preserved as raw counts. The labels are used as-is by default: they
    already mark neurite voxels, so dilating them would annex surrounding
    background into the preserved region. Dilation is opt-in (dilate > 0) to
    feather labeled boundaries / partial-volume edges when wanted.

    Note: segments that are bright, spatially incoherent processing artifacts
    are handled by rejecting the whole patch at sampling time (see
    patch_has_incoherent_segment / TrainDataset), not by editing this mask, so
    a corrupted raw patch never becomes a training example at all.

    Parameters
    ----------
    labels : numpy.ndarray
        Segmentation label patch; 0 is background and any positive id is a
        labeled object.
    dilate : int, optional
        Number of binary-dilation iterations. Default is 0 (labels as-is).

    Returns
    -------
    numpy.ndarray
        Boolean foreground mask with the same shape as "labels".
    """
    mask = np.asarray(labels) > 0
    if dilate > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilate)
    return mask


def patch_has_incoherent_segment(
    labels,
    raw,
    min_autocorr=0.4,
    max_highfreq_frac=0.35,
    min_segment_voxels=50,
    smooth_sigma=1.0,
    coherence_lag=2,
):
    """
    Tests whether a patch contains a spatially incoherent artifact segment.

    Some regions of the raw image are salt-and-pepper noise from an upstream
    processing artifact -- bright, blocky voxel noise that the segmenter labels
    as an object. Such a segment is brighter than real signal (so an intensity
    threshold cannot catch it) but spatially incoherent, whereas a real neurite
    is PSF-blurred and stays correlated across the ~2-3 voxel PSF. Because the
    artifact corrupts the raw input itself -- not just the label -- a patch that
    contains one is a poor training example even after the label is removed, so
    callers reject and resample the whole patch.

    A segment counts as an artifact only when it fails BOTH coherence tests
    (lag-"coherence_lag" autocorrelation below "min_autocorr" AND high-frequency
    energy fraction above "max_highfreq_frac"), so a thin, faint but smooth
    neurite -- low autocorrelation from its thinness, but low high-frequency
    energy -- is not mistaken for one. Segments smaller than
    "min_segment_voxels" are ignored (too few voxels to score reliably).

    Parameters
    ----------
    labels : numpy.ndarray
        Segmentation label patch; 0 is background, positive ids are objects.
    raw : numpy.ndarray
        Image patch in counts, aligned with "labels".
    min_autocorr : float, optional
        Lag-"coherence_lag" autocorrelation at or above which a segment is
        considered coherent (real). Default is 0.4.
    max_highfreq_frac : float, optional
        High-frequency energy fraction at or below which a segment is
        considered coherent (real). Default is 0.35.
    min_segment_voxels : int, optional
        Segments with fewer voxels than this are ignored. Default is 50.
    smooth_sigma : float, optional
        Gaussian sigma for the high-frequency split. Default is 1.0.
    coherence_lag : int, optional
        Voxel lag at which the autocorrelation is measured. Default is 2 (the
        scale that separates real PSF-correlated signal from blocky artifacts
        that only correlate at lag 1).

    Returns
    -------
    bool
        True if any scorable segment is a spatially incoherent artifact.
    """
    labels = np.asarray(labels)
    mask = labels > 0
    if not mask.any():
        return False
    raw = np.asarray(raw, dtype=np.float64)
    smooth = ndimage.gaussian_filter(raw, sigma=smooth_sigma)
    for lid in np.unique(labels[mask]):
        if lid == 0:
            continue
        seg = labels == lid
        if seg.sum() < min_segment_voxels:
            continue
        autocorr = local_autocorr(raw, seg, lag=coherence_lag)
        if autocorr >= min_autocorr:
            continue
        if highfreq_energy_fraction(raw, seg, smooth=smooth) > max_highfreq_frac:
            return True
    return False


def make_skeleton_mask(points, start, patch_shape, dilate=2):
    """
    Builds a foreground mask from ground-truth skeleton points in a patch.

    Rasterizes traced neurite nodes before dilating them to an approximate
    neurite radius, so ground-truth signal is preserved even where the
    segmentation does not label it. SWCs are expected to contain a node at
    every 3D voxel step. Raw intensity is never consulted, so noise is not
    picked up.

    Parameters
    ----------
    points : numpy.ndarray
        Skeleton points as an (N, 3) array of voxel coordinates in the brain
        volume, in the same axis order as the patch.
    start : Sequence[int]
        Lower corner of the patch in the brain volume (center - patch_shape //
        2 per axis), matching img_util.get_slices.
    patch_shape : Tuple[int]
        Shape of the patch.
    dilate : int, optional
        Binary-dilation iterations approximating the neurite radius, in voxels
        and treated isotropically (anisotropy is ignored). Default is 2.

    Returns
    -------
    numpy.ndarray
        Boolean foreground mask with shape "patch_shape".
    """
    start = np.asarray(start)
    stop = start + np.asarray(patch_shape)
    pts = np.asarray(points)
    inside = np.all((pts >= start) & (pts < stop), axis=1)
    mask = np.zeros(tuple(patch_shape), dtype=bool)
    local = (pts[inside] - start).astype(int)
    if local.size:
        mask[local[:, 0], local[:, 1], local[:, 2]] = True

    if dilate > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilate)
    return mask


def foreground_background_mae(pred, ref, fg_mask):
    """
    Computes the mean absolute error split by a foreground mask.

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted image in counts.
    ref : numpy.ndarray
        Reference image in counts.
    fg_mask : numpy.ndarray
        Boolean foreground mask.

    Returns
    -------
    Tuple[float]
        Foreground MAE and background MAE. A side with no voxels reports 0.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    fg = np.asarray(fg_mask, dtype=bool)
    err = np.abs(pred - ref)
    fg_mae = float(err[fg].mean()) if fg.any() else 0.0
    bg_mae = float(err[~fg].mean()) if (~fg).any() else 0.0
    return fg_mae, bg_mae


def mip_max_error(pred, raw):
    """
    Computes the absolute error between the maxima of two images.

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted image in counts.
    raw : numpy.ndarray
        Raw image in counts.

    Returns
    -------
    float
        Absolute difference between the maximum intensities.
    """
    return float(abs(np.max(pred) - np.max(raw)))


def false_bright_rate(pred, raw, fg_mask, k=6.0):
    """
    Computes the fraction of background voxels the model made bright.

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted image in counts.
    raw : numpy.ndarray
        Raw image in counts (used to set the brightness threshold).
    fg_mask : numpy.ndarray
        Boolean foreground mask.
    k : float, optional
        Threshold in robust standard deviations above the median. Default is
        6.0.

    Returns
    -------
    float
        Fraction of background voxels where "pred" exceeds the threshold.
    """
    pred = np.asarray(pred, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    bg = ~np.asarray(fg_mask, dtype=bool)
    if not bg.any():
        return 0.0
    med = np.median(raw)
    mad = np.median(np.abs(raw - med)) + 1e-6
    thr = med + k * 1.4826 * mad
    return float(np.mean(pred[bg] > thr))


def evaluate_example(pred, raw, target, fg_mask, pct=0.1):
    """
    Computes the full metric dictionary for a single example, in counts.

    Foreground fidelity is measured against the raw counts (preserve signal);
    background cleanup is measured against the BM4D target (clean like the
    teacher).

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted image in counts.
    raw : numpy.ndarray
        Raw noisy image in counts.
    target : numpy.ndarray
        BM4D-denoised target image in counts.
    fg_mask : numpy.ndarray
        Boolean foreground mask.
    pct : float, optional
        Top-percentile fraction used for the bright-tail metrics. Default is
        0.1 (i.e., the top 0.1%).

    Returns
    -------
    dict
        Metric name to scalar value.
    """
    fg_mae, _ = foreground_background_mae(pred, raw, fg_mask)
    _, bg_mae = foreground_background_mae(pred, target, fg_mask)

    q = 100.0 - pct
    raw_top = float(np.percentile(np.asarray(raw, dtype=np.float64), q))
    pred_top = float(np.percentile(np.asarray(pred, dtype=np.float64), q))
    return {
        "fg_mae": fg_mae,
        "bg_mae": bg_mae,
        "top_pct_error": abs(pred_top - raw_top),
        "top_pct_preservation": pred_top / (raw_top + 1e-8),
        "mip_max_error": mip_max_error(pred, raw),
        "false_bright_rate": false_bright_rate(pred, raw, fg_mask),
    }


def _value_bin(value, bins):
    """Return the fixed half-open bin label containing a scalar value."""
    for lower, upper, label in bins:
        if lower <= value < upper:
            return label
    return bins[-1][2]


def _flat_background_mask(raw, foreground, saturation, percentile=50.0):
    """Select low-gradient background for high-frequency noise estimates."""
    gradient = ndimage.gaussian_gradient_magnitude(
        np.asarray(raw, dtype=np.float32), sigma=1.0
    )
    candidates = ~foreground & ~saturation
    if not candidates.any():
        return candidates
    threshold = float(np.percentile(gradient[candidates], percentile))
    return candidates & (gradient <= threshold)


def _edge_metrics(raw_mask, pred_mask, region):
    """Measure symmetric bright-edge displacement and regional width change."""
    raw_mask = np.asarray(raw_mask, dtype=bool)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    region = np.asarray(region, dtype=bool)
    raw_edge = raw_mask ^ ndimage.binary_erosion(raw_mask)
    pred_edge = pred_mask ^ ndimage.binary_erosion(pred_mask)
    raw_edge &= region
    pred_edge &= region
    distances = list()
    if raw_edge.any() and pred_mask.any():
        distances.extend(ndimage.distance_transform_edt(~pred_mask)[raw_edge])
    if pred_edge.any() and raw_mask.any():
        distances.extend(ndimage.distance_transform_edt(~raw_mask)[pred_edge])
    edge_displacement = (
        float(np.mean(distances)) if len(distances) else float("nan")
    )
    raw_width = int(np.count_nonzero(raw_mask & region))
    pred_width = int(np.count_nonzero(pred_mask & region))
    width_change = (
        (pred_width - raw_width) / raw_width
        if raw_width
        else float("nan")
    )
    return edge_displacement, float(width_change)


def evaluate_stratified_example(
    pred,
    raw,
    target,
    fg_mask,
    brain_id,
    center,
    offset=0.0,
    noise_params=None,
    max_count=65535.0,
    saturation_margin=64.0,
):
    """Compute legacy, brightness, provenance, saturation, and halo metrics."""
    pred = np.asarray(pred, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    foreground = np.asarray(fg_mask, dtype=bool)
    offset = float(offset)
    raw_sensor = raw + offset
    pred_sensor = pred + offset
    saturation_threshold = float(max_count) - float(saturation_margin)
    saturation = raw_sensor >= saturation_threshold
    flat_background = _flat_background_mask(raw, foreground, saturation)
    raw_hf = raw - ndimage.gaussian_filter(raw, sigma=1.0)
    pred_hf = pred - ndimage.gaussian_filter(pred, sigma=1.0)
    background_noise_std = (
        float(np.std(raw_hf[flat_background]))
        if flat_background.any()
        else float("nan")
    )
    foreground_present = bool(foreground.any())
    foreground_intensity = (
        float(np.percentile(raw[foreground], 95.0))
        if foreground_present
        else float("nan")
    )
    foreground_bin = (
        _value_bin(foreground_intensity, FOREGROUND_INTENSITY_BINS)
        if foreground_present
        else "blank"
    )
    background_noise_bin = (
        _value_bin(background_noise_std, BACKGROUND_NOISE_BINS)
        if np.isfinite(background_noise_std)
        else "unmeasured"
    )
    calibrated_background_sigma = float("nan")
    if noise_params is not None:
        params = np.asarray(noise_params, dtype=np.float32)
        if params.shape == (2,) and np.isfinite(params[1]) and params[1] >= 0:
            calibrated_background_sigma = float(np.sqrt(params[1]))

    halo = dict()
    if saturation.any():
        distance = ndimage.distance_transform_edt(~saturation)
        core_raw_mean = float(np.mean(raw[saturation]))
        core_pred_mean = float(np.mean(pred[saturation]))
        core_preservation = core_pred_mean / (core_raw_mean + 1e-8)
        bright_threshold = max(
            float(np.percentile(raw_sensor, 99.0)),
            saturation_threshold * 0.5,
        )
        raw_bright = raw_sensor >= bright_threshold
        pred_bright = pred_sensor >= bright_threshold
        for lower, upper, label in HALO_DISTANCE_BANDS:
            band = (distance >= lower) & (distance < upper)
            flat_band = band & (
                flat_background | (saturation if lower == 0 else False)
            )
            input_hf_std = (
                float(np.std(raw_hf[flat_band]))
                if flat_band.any()
                else float("nan")
            )
            output_hf_std = (
                float(np.std(pred_hf[flat_band]))
                if flat_band.any()
                else float("nan")
            )
            edge_displacement, width_change = _edge_metrics(
                raw_bright, pred_bright, band
            )
            raw_mean = float(np.mean(raw[band])) if band.any() else float("nan")
            pred_mean = (
                float(np.mean(pred[band])) if band.any() else float("nan")
            )
            halo[label] = {
                "voxel_count": int(np.count_nonzero(band)),
                "flat_voxel_count": int(np.count_nonzero(flat_band)),
                "input_hf_std": input_hf_std,
                "output_hf_std": output_hf_std,
                "output_input_hf_ratio": output_hf_std / (input_hf_std + 1e-8),
                "mean_intensity_bias": pred_mean - raw_mean,
                "intensity_preservation": pred_mean / (raw_mean + 1e-8),
                "saturated_core_preservation": (
                    core_preservation if lower == 0 else float("nan")
                ),
                "bright_edge_displacement": edge_displacement,
                "bright_width_change": width_change,
            }
    else:
        for _, _, label in HALO_DISTANCE_BANDS:
            halo[label] = {
                "voxel_count": 0,
                "flat_voxel_count": 0,
                "input_hf_std": float("nan"),
                "output_hf_std": float("nan"),
                "output_input_hf_ratio": float("nan"),
                "mean_intensity_bias": float("nan"),
                "intensity_preservation": float("nan"),
                "saturated_core_preservation": float("nan"),
                "bright_edge_displacement": float("nan"),
                "bright_width_change": float("nan"),
            }

    return {
        "legacy": evaluate_example(pred, raw, target, foreground),
        "provenance": {
            "brain_id": str(brain_id),
            "center": [int(value) for value in center],
        },
        "strata": {
            "foreground_presence": (
                "foreground" if foreground_present else "blank"
            ),
            "foreground_intensity_bin": foreground_bin,
            "background_noise_bin": background_noise_bin,
            "saturation": "saturated" if saturation.any() else "unsaturated",
        },
        "patch_metrics": {
            "foreground_voxel_count": int(np.count_nonzero(foreground)),
            "foreground_intensity_p95": foreground_intensity,
            "background_noise_std": background_noise_std,
            "calibrated_background_sigma": calibrated_background_sigma,
            "saturated_voxel_count": int(np.count_nonzero(saturation)),
        },
        "halo": halo,
    }


def _aggregate_record_group(records):
    """Aggregate legacy and patch scalars without blank-foreground dilution."""
    legacy = dict()
    for key in LEGACY_METRIC_KEYS:
        eligible = records
        if key == "fg_mae":
            eligible = [
                record
                for record in records
                if record["strata"]["foreground_presence"] == "foreground"
            ]
        values = [record["legacy"][key] for record in eligible]
        values = [value for value in values if np.isfinite(value)]
        legacy[key] = float(np.mean(values)) if values else float("nan")
    patch_metrics = dict()
    for key in (
        "foreground_intensity_p95",
        "background_noise_std",
        "calibrated_background_sigma",
        "saturated_voxel_count",
    ):
        values = [record["patch_metrics"][key] for record in records]
        values = [value for value in values if np.isfinite(value)]
        patch_metrics[key] = (
            float(np.mean(values)) if values else float("nan")
        )
    return {
        **legacy,
        **patch_metrics,
        "patch_count": len(records),
        "foreground_patch_count": sum(
            record["strata"]["foreground_presence"] == "foreground"
            for record in records
        ),
    }


def aggregate_stratified_metrics(records):
    """Aggregate validation records by brain, brightness, noise, and saturation."""
    if not records:
        return {"overall": {}, "patches": []}

    def grouped_records(path):
        groups = dict()
        for record in records:
            value = record
            for key in path:
                value = value[key]
            groups.setdefault(str(value), []).append(record)
        return groups

    def grouped(path):
        return {
            label: _aggregate_record_group(group)
            for label, group in sorted(grouped_records(path).items())
        }

    def aggregate_halo(group):
        halo = dict()
        for _, _, label in HALO_DISTANCE_BANDS:
            band_rows = [record["halo"][label] for record in group]
            aggregate = {
                "voxel_count": int(
                    sum(row["voxel_count"] for row in band_rows)
                ),
                "flat_voxel_count": int(
                    sum(row["flat_voxel_count"] for row in band_rows)
                ),
            }
            for key in (
                "input_hf_std",
                "output_hf_std",
                "output_input_hf_ratio",
                "mean_intensity_bias",
                "intensity_preservation",
                "saturated_core_preservation",
                "bright_edge_displacement",
                "bright_width_change",
            ):
                weighted = [
                    (row[key], row["voxel_count"])
                    for row in band_rows
                    if row["voxel_count"] and np.isfinite(row[key])
                ]
                denominator = sum(weight for _, weight in weighted)
                aggregate[key] = (
                    float(
                        sum(value * weight for value, weight in weighted)
                        / denominator
                    )
                    if denominator
                    else float("nan")
                )
            halo[label] = aggregate
        return halo

    brain_groups = grouped_records(("provenance", "brain_id"))

    return {
        "overall": _aggregate_record_group(records),
        "by_brain": grouped(("provenance", "brain_id")),
        "by_foreground_presence": grouped(
            ("strata", "foreground_presence")
        ),
        "by_foreground_intensity": grouped(
            ("strata", "foreground_intensity_bin")
        ),
        "by_background_noise": grouped(("strata", "background_noise_bin")),
        "by_saturation": grouped(("strata", "saturation")),
        "halo_distance_bands": aggregate_halo(records),
        "halo_by_brain": {
            brain_id: aggregate_halo(group)
            for brain_id, group in sorted(brain_groups.items())
        },
        "patches": records,
    }


def checkpoint_score(metrics, cratio, weights=None):
    """
    Computes the checkpoint-selection score (lower is better).

    Combines count-scale fidelity terms and rewards compression through a
    negative weight. With ``weights["cratio"] == 0`` (default) the score is
    purely fidelity-driven.

    Parameters
    ----------
    metrics : dict
        Aggregated metric dictionary, as produced by "evaluate_example".
    cratio : float
        Compression ratio (higher is better).
    weights : dict, optional
        Term weights. Defaults to "DEFAULT_CHECKPOINT_WEIGHTS".

    Returns
    -------
    float
        Checkpoint-selection score.
    """
    w = DEFAULT_CHECKPOINT_WEIGHTS if weights is None else weights
    return (
        w.get("fg_mae", 0.0) * metrics["fg_mae"]
        + w.get("bg_mae", 0.0) * metrics["bg_mae"]
        + w.get("top_pct_error", 0.0) * metrics["top_pct_error"]
        - w.get("cratio", 0.0) * cratio
    )
