"""
Visualize precomputed patches (raw, BM4D teacher, target, foreground mask).

Renders a grid of cached patches so the foreground masks -- now built from the
segmentation labels unioned with the traced skeleton (see
data_handling.foreground_mask) -- can be eyeballed against the actual signal:
is the mask covering neurites, and is it staying off bright non-neuronal
structures (noise, off-target label)?

Works on either cache produced by scripts/precompute.py (train or val); both
share the same layout::

    raw.npy      float16  (N, *patch_shape)   offset-subtracted counts
    teacher.npy  float16  (N, *patch_shape)   clipped BM4D denoising
    fg.npy       uint8    (N, *patch_shape)   foreground mask (0/1)

Each patch is one row with five count-space panels:

    raw | teacher | target | fg mask | raw + mask overlay

where ``target = where(fg, raw, teacher)`` is exactly what the model trains
against (with preserve_foreground). 3D patches are reduced to 2D with a
maximum-intensity projection (``--mode mip``, the default) or a center slice
(``--mode slice``) along the chosen ``--axis``. raw/teacher/target share one
percentile contrast window (computed from raw) so the denoising effect and the
preserved foreground are directly comparable.

Examples
--------
    # 8 random patches from the training cache
    python scripts/visualize_patches.py /results/patch_cache

    # foreground-rich patches (so the masks are actually visible), center slice
    python scripts/visualize_patches.py /results/val_patch_cache \
        --sort-by-fg --mode slice --out val_preview.png

    # specific patches by index
    python scripts/visualize_patches.py /results/patch_cache --indices 0 5 42
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never open a window

import matplotlib.pyplot as plt
import numpy as np

from aind_exaspim_image_compression.utils import util

PANELS = ("raw", "teacher", "target", "fg mask", "raw + mask")


def load_cache(cache_dir):
    """
    Memory-maps the cached arrays and reads the stamped transform cfg.

    Parameters
    ----------
    cache_dir : str
        Directory holding raw.npy, teacher.npy, and fg.npy.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, dict]
        (raw, teacher, fg) memmaps and the transform cfg (or None if absent).
    """
    raw = np.load(os.path.join(cache_dir, "raw.npy"), mmap_mode="r")
    teacher = np.load(os.path.join(cache_dir, "teacher.npy"), mmap_mode="r")
    fg = np.load(os.path.join(cache_dir, "fg.npy"), mmap_mode="r")
    cfg_path = os.path.join(cache_dir, "transform.json")
    cfg = util.read_json(cfg_path) if os.path.exists(cfg_path) else None
    return raw, teacher, fg, cfg


def pick_indices(fg, n, indices, sort_by_fg, seed):
    """
    Chooses which cached patches to render.

    Parameters
    ----------
    fg : numpy.ndarray
        Foreground-mask memmap with shape (N, *patch_shape).
    n : int
        Number of patches to render when indices are not given explicitly.
    indices : list of int or None
        Explicit patch indices; overrides n / sort_by_fg when provided.
    sort_by_fg : bool
        When True, sample a candidate pool at random and keep the n patches
        with the largest foreground fraction (only the pool is read, not the
        whole cache), so the masks are actually visible.
    seed : int
        RNG seed for reproducible selection.

    Returns
    -------
    list of int
        Selected patch indices.
    """
    pool_size = len(fg)
    if indices:
        return [i for i in indices if 0 <= i < pool_size]

    rng = np.random.default_rng(seed)
    n = min(n, pool_size)
    if not sort_by_fg:
        return sorted(rng.choice(pool_size, size=n, replace=False).tolist())

    # Score a bounded random candidate pool by foreground fraction so we do not
    # read the entire (multi-GB) fg array just to rank patches.
    n_candidates = min(pool_size, max(20 * n, 200))
    candidates = rng.choice(pool_size, size=n_candidates, replace=False)
    candidates.sort()
    fracs = np.array([np.asarray(fg[i]).mean() for i in candidates])
    top = candidates[np.argsort(fracs)[::-1][:n]]
    return sorted(top.tolist())


def project(vol, mode, axis):
    """
    Reduces a 3D patch to 2D by MIP or a center slice along an axis.

    Parameters
    ----------
    vol : numpy.ndarray
        3D patch.
    mode : str
        "mip" for a maximum-intensity projection, "slice" for the center slice.
    axis : int
        Axis to project or slice along.

    Returns
    -------
    numpy.ndarray
        2D projection.
    """
    if mode == "mip":
        return vol.max(axis=axis)
    center = vol.shape[axis] // 2
    return np.take(vol, center, axis=axis)


def stretch(img2d, lo, hi):
    """Percentile-window a 2D image to [0, 1] for display."""
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img2d.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def overlay(gray2d, mask2d, alpha=0.45):
    """
    Tints the foreground red over a grayscale background.

    Parameters
    ----------
    gray2d : numpy.ndarray
        Background image already normalized to [0, 1].
    mask2d : numpy.ndarray
        Boolean foreground projection with the same shape as gray2d.
    alpha : float, optional
        Opacity of the red tint. Default is 0.45.

    Returns
    -------
    numpy.ndarray
        (H, W, 3) RGB image.
    """
    rgb = np.stack([gray2d, gray2d, gray2d], axis=-1)
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[mask2d] = (1.0 - alpha) * rgb[mask2d] + alpha * red
    return rgb


def render(cache_dir, out_path, n, indices, sort_by_fg, mode, axis, seed,
           low_pct, high_pct):
    """
    Builds and saves the patch-preview grid; prints a per-patch summary.

    Parameters
    ----------
    cache_dir : str
        Cache directory to visualize.
    out_path : str
        Output PNG path.
    n : int
        Number of patches when indices are not given.
    indices : list of int or None
        Explicit patch indices.
    sort_by_fg : bool
        Prefer foreground-rich patches (see pick_indices).
    mode : str
        "mip" or "slice".
    axis : int
        Projection/slice axis.
    seed : int
        Selection RNG seed.
    low_pct, high_pct : float
        Contrast percentiles (computed on the raw projection, reused for
        teacher and target).
    """
    raw, teacher, fg, cfg = load_cache(cache_dir)
    idxs = pick_indices(fg, n, indices, sort_by_fg, seed)
    if not idxs:
        raise SystemExit(f"No patches to render from {cache_dir}")

    n_rows, n_cols = len(idxs), len(PANELS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows), squeeze=False,
    )

    print(f"Cache: {cache_dir}  | pool: {len(raw)}  | transform: {cfg}")
    print(f"{'idx':>7}  {'fg%':>6}  {'raw_max':>9}  {'teach_max':>9}")
    for r, i in enumerate(idxs):
        raw_p = np.asarray(raw[i], dtype=np.float32)
        teach_p = np.asarray(teacher[i], dtype=np.float32)
        fg_p = np.asarray(fg[i]).astype(bool)
        target_p = np.where(fg_p, raw_p, teach_p)

        raw_m = project(raw_p, mode, axis)
        teach_m = project(teach_p, mode, axis)
        target_m = project(target_p, mode, axis)
        fg_m = project(fg_p.astype(np.uint8), mode, axis).astype(bool)

        lo, hi = np.percentile(raw_m, (low_pct, high_pct))
        raw_s = stretch(raw_m, lo, hi)
        panels = [
            (raw_s, "gray"),
            (stretch(teach_m, lo, hi), "gray"),
            (stretch(target_m, lo, hi), "gray"),
            (fg_m.astype(np.float32), "magma"),
            (overlay(raw_s, fg_m), None),
        ]
        for c, (data, cmap) in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(PANELS[c], fontsize=10)
        fg_frac = 100.0 * fg_p.mean()
        axes[r][0].set_ylabel(f"#{i}\n{fg_frac:.1f}% fg", fontsize=9)
        print(f"{i:>7}  {fg_frac:>6.2f}  {raw_p.max():>9.1f}  {teach_p.max():>9.1f}")

    title = f"{os.path.basename(os.path.normpath(cache_dir))}  |  {mode} axis={axis}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    util.mkdir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cache_dir", help="Cache directory (train or val).")
    parser.add_argument(
        "--out", default="patch_preview.png", help="Output PNG path."
    )
    parser.add_argument(
        "--n", type=int, default=8, help="Number of patches to render."
    )
    parser.add_argument(
        "--indices", type=int, nargs="+", default=None,
        help="Explicit patch indices (overrides --n / --sort-by-fg).",
    )
    parser.add_argument(
        "--sort-by-fg", action="store_true",
        help="Prefer foreground-rich patches so the masks are visible.",
    )
    parser.add_argument(
        "--mode", choices=("mip", "slice"), default="mip",
        help="Reduce 3D patches by max-intensity projection or center slice.",
    )
    parser.add_argument(
        "--axis", type=int, default=0, help="Projection/slice axis (0=z)."
    )
    parser.add_argument("--seed", type=int, default=0, help="Selection seed.")
    parser.add_argument("--low-pct", type=float, default=1.0)
    parser.add_argument("--high-pct", type=float, default=99.9)
    args = parser.parse_args()

    render(
        args.cache_dir, args.out, args.n, args.indices, args.sort_by_fg,
        args.mode, args.axis, args.seed, args.low_pct, args.high_pct,
    )


if __name__ == "__main__":
    main()
