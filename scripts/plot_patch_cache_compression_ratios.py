"""Plot per-patch compression-ratio distributions for patch caches.

Compression matches training: patches are cast to uint16 and compressed as a
single 64^3 chunk with Blosc Zstd, clevel 5, and byte shuffle.  The ratio is
uncompressed bytes / compressed bytes, so larger values are better.
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numcodecs import blosc


DEFAULT_CACHES = (
    Path("/root/capsule/data/denoise_net_patch_cache_2026_07_13"),
    Path("/root/capsule/data/denoise_net_patch_cache_2026_07_15"),
)
SPLITS = ("patch_cache", "val_patch_cache")
ARRAYS = ("raw", "teacher")
COLORS = {"2026_07_13": "#4477AA", "2026_07_15": "#EE6677"}
FALLBACK_COLORS = (
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
)
THREAD_LOCAL = threading.local()


def _cache_colors(cache_dates) -> dict[str, str]:
    """Return consistent colors for both known and newly supplied caches."""
    dates = sorted(set(cache_dates))
    colors = {date: COLORS[date] for date in dates if date in COLORS}
    unknown_dates = [date for date in dates if date not in colors]
    colors.update(
        {
            date: FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
            for index, date in enumerate(unknown_dates)
        }
    )
    return colors


def _codec():
    """Return one codec per worker thread."""
    codec = getattr(THREAD_LOCAL, "codec", None)
    if codec is None:
        codec = blosc.Blosc(
            cname="zstd", clevel=5, shuffle=blosc.SHUFFLE
        )
        THREAD_LOCAL.codec = codec
    return codec


def _ratio(array: np.ndarray) -> float:
    chunk = np.ascontiguousarray(array, dtype=np.uint16)
    return chunk.nbytes / len(_codec().encode(chunk))


def measure_cache(cache_root: Path, split: str, workers: int) -> pd.DataFrame:
    """Measure raw and teacher compression ratios for every cached patch."""
    cache_dir = cache_root / split
    raw = np.load(cache_dir / "raw.npy", mmap_mode="r")
    teacher = np.load(cache_dir / "teacher.npy", mmap_mode="r")
    if raw.shape != teacher.shape:
        raise ValueError(f"Shape mismatch in {cache_dir}: {raw.shape} != {teacher.shape}")

    def measure_one(index: int) -> tuple[float, float]:
        return _ratio(raw[index]), _ratio(teacher[index])

    # Each Blosc call uses one thread; parallelism happens across patches.
    blosc.set_nthreads(1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        measured = list(executor.map(measure_one, range(len(raw))))

    ratios = np.asarray(measured)
    date = cache_root.name.removeprefix("denoise_net_patch_cache_")
    frame = pd.DataFrame(
        {
            "cache_date": date,
            "split": "train" if split == "patch_cache" else "validation",
            "patch_index": np.arange(len(raw)),
            "raw_cratio": ratios[:, 0],
            "teacher_cratio": ratios[:, 1],
        }
    )
    frame["teacher_raw_gain"] = frame["teacher_cratio"] / frame["raw_cratio"]
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (date, split), group in frame.groupby(["cache_date", "split"], sort=True):
        for kind in ARRAYS:
            values = group[f"{kind}_cratio"].to_numpy()
            rows.append(
                {
                    "cache_date": date,
                    "split": split,
                    "array": kind,
                    "n": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "p05": np.percentile(values, 5),
                    "p25": np.percentile(values, 25),
                    "median": np.median(values),
                    "p75": np.percentile(values, 75),
                    "p95": np.percentile(values, 95),
                    "max": np.max(values),
                }
            )
    return pd.DataFrame(rows)


def _panel_hist(ax, groups, colors, xlabel: str, title: str) -> None:
    all_values = np.concatenate([values for _, values in groups])
    lo, hi = np.percentile(all_values, [0.5, 99.5])
    if lo == hi:
        lo, hi = np.min(all_values), np.max(all_values) + 1
    bins = np.geomspace(lo, hi, 70)
    for label, values in groups:
        color = colors[label]
        median = np.median(values)
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 100 / len(values)),
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"{label} (median {median:.2f})",
        )
        ax.axvline(median, color=color, linewidth=1, linestyle="--", alpha=0.8)
    ax.set_xlim(lo, hi)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(f"{xlabel} (log scale)")
    ax.set_ylabel("Patches per bin (%)")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=8)
    ax.text(
        0.99,
        0.98,
        "x-axis: pooled 0.5th–99.5th percentiles",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="0.35",
    )


def plot_histograms(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = _cache_colors(frame["cache_date"])
    for row, split in enumerate(("train", "validation")):
        split_frame = frame[frame["split"] == split]
        for col, kind in enumerate(ARRAYS):
            groups = [
                (
                    date,
                    split_frame[split_frame["cache_date"] == date][
                        f"{kind}_cratio"
                    ].to_numpy(),
                )
                for date in sorted(frame["cache_date"].unique())
            ]
            _panel_hist(
                axes[row, col],
                groups,
                colors,
                "Compression ratio (uncompressed / compressed)",
                f"{split.title()} — {kind.title()} patches",
            )
    fig.suptitle(
        "Patch-cache compression-ratio distributions\n"
        "Blosc Zstd level 5, byte shuffle, uint16, one 64³ chunk per patch",
        fontsize=14,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ecdfs(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = _cache_colors(frame["cache_date"])
    for row, split in enumerate(("train", "validation")):
        for col, kind in enumerate(ARRAYS):
            ax = axes[row, col]
            subset = frame[frame["split"] == split]
            for date in sorted(frame["cache_date"].unique()):
                values = np.sort(
                    subset[subset["cache_date"] == date][f"{kind}_cratio"].to_numpy()
                )
                y = np.arange(1, len(values) + 1) / len(values)
                ax.plot(values, y, color=colors[date], label=date, linewidth=1.8)
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_title(f"{split.title()} — {kind.title()} patches")
            ax.set_xlabel("Compression ratio (log scale)")
            ax.set_ylabel("Cumulative fraction")
            ax.grid(alpha=0.2, which="both")
            ax.legend(frameon=False)
    fig.suptitle(
        "Patch-cache compression-ratio ECDFs\n"
        "Blosc Zstd level 5, byte shuffle, uint16, one 64³ chunk per patch",
        fontsize=14,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_gains(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = _cache_colors(frame["cache_date"])
    for ax, split in zip(axes, ("train", "validation")):
        subset = frame[frame["split"] == split]
        groups = [
            (
                date,
                subset[subset["cache_date"] == date]["teacher_raw_gain"].to_numpy(),
            )
            for date in sorted(frame["cache_date"].unique())
        ]
        _panel_hist(
            ax,
            groups,
            colors,
            "Teacher / raw compression-ratio gain",
            split.title(),
        )
        ax.axvline(1, color="0.25", linewidth=1, linestyle=":")
    fig.suptitle("Per-patch compression gain after teacher denoising", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", nargs="*", type=Path, default=DEFAULT_CACHES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/capsule/results/patch_cache_compression_ratios"),
    )
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for cache in args.caches:
        for split in SPLITS:
            print(f"Measuring {cache.name}/{split} ...", flush=True)
            frames.append(measure_cache(cache, split, args.workers))

    frame = pd.concat(frames, ignore_index=True)
    summary = summarize(frame)
    frame.to_csv(args.output_dir / "per_patch_compression_ratios.csv", index=False)
    summary.to_csv(args.output_dir / "compression_ratio_summary.csv", index=False)
    plot_histograms(frame, args.output_dir / "compression_ratio_distributions.png")
    plot_ecdfs(frame, args.output_dir / "compression_ratio_ecdfs.png")
    plot_gains(frame, args.output_dir / "teacher_compression_gain.png")
    metadata = {
        "definition": "uncompressed uint16 bytes / compressed bytes",
        "codec": "Blosc Zstd",
        "compression_level": 5,
        "shuffle": "byte shuffle",
        "chunk_shape": [64, 64, 64],
        "caches": [str(path) for path in args.caches],
        "rows": len(frame),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("\nSummary:\n", summary.to_string(index=False), flush=True)
    print(f"\nWrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
