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


def make_segmentation_mask(labels, dilate=0):
    """
    Builds a foreground mask from segmentation labels.

    Foreground is the labeled neurites alone, so bright non-neuronal structures
    (noise, off-target label) are left for the BM4D teacher to denoise rather
    than preserved as raw counts. The labels are used as-is by default: they
    already mark neurite voxels, so dilating them would annex surrounding
    background into the preserved region. Dilation is opt-in (dilate > 0) to
    feather labeled boundaries / partial-volume edges when wanted.

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


def make_skeleton_mask(points, start, patch_shape, dilate=2):
    """
    Builds a foreground mask from ground-truth skeleton points in a patch.

    Rasterizes the traced neurite centerline points that fall within the patch
    and dilates them to an approximate neurite radius, so ground-truth signal
    is preserved even where the segmentation does not label it. Raw intensity
    is never consulted, so noise is not picked up.

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
