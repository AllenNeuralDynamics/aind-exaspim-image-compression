"""
Reads denoised zarrs produced by run_bm4d.py and computes per-block metrics
(compression ratio, compressed file size, SSIM vs raw) for every experiment
subdir (sigma{value}_{dtype}) under dest_prefix. Writes one CSV per
experiment and a summary CSV with mean ± std across blocks.
"""

import csv
import os
import re

import boto3
import numpy as np
from numcodecs import blosc
from skimage.metrics import structural_similarity

from aind_exaspim_image_compression.utils.img_util import compute_cratio, read as read_image
from aind_exaspim_image_compression.utils.util import list_s3_prefixes, parse_cloud_path


def get_s3_size_mb(bucket, prefix):
    """Sum the compressed sizes of all S3 objects under a prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    total_bytes = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            total_bytes += obj["Size"]
    return total_bytes / 1e6


def to_uint8(volume, p999=None):
    """Clip to the 99.9th percentile and rescale to uint8."""
    vol = volume.astype(np.float32)
    if p999 is None:
        p999 = np.percentile(vol, 99.9)
    return (np.clip(vol, 0, p999) * (255.0 / p999)).astype(np.uint8)


def list_zarr_paths(bucket_name, prefix):
    """Recursively find all .zarr directories under a prefix."""
    search_prefix = prefix if prefix.endswith("/") else prefix + "/"
    zarr_entries = []
    to_visit = [search_prefix]
    while to_visit:
        current = to_visit.pop()
        for subprefix in list_s3_prefixes(bucket_name, current):
            name = subprefix.rstrip("/").split("/")[-1]
            if name.endswith(".zarr"):
                s3_path = f"s3://{bucket_name}/{subprefix.rstrip('/')}"
                rel = subprefix.rstrip("/")[len(search_prefix):]
                zarr_entries.append((s3_path, rel))
            else:
                to_visit.append(subprefix)
    return zarr_entries


def analyze(source_prefix, dest_prefix, output_dir):
    src_bucket, src_prefix = parse_cloud_path(source_prefix)
    dst_bucket, dst_prefix = parse_cloud_path(dest_prefix)
    codec = blosc.Blosc(cname="zstd", clevel=5, shuffle=blosc.SHUFFLE)

    os.makedirs(output_dir, exist_ok=True)

    # Build lookup: rel_path -> raw s3_path
    raw_lookup = {rel: path for path, rel in list_zarr_paths(src_bucket, src_prefix)}

    # Compute raw reference stats (cratio, S3 file size) once per block
    print("Computing raw reference stats ...")
    raw_rows = []
    for raw_s3_path, rel_path in list_zarr_paths(src_bucket, src_prefix):
        stem = rel_path.split("/")[-1].removesuffix(".zarr")
        raw = np.squeeze(np.asarray(read_image(raw_s3_path + "/0")))
        raw_key = raw_s3_path.removeprefix(f"s3://{src_bucket}/")
        raw_rows.append({
            "block": stem,
            "cratio": round(compute_cratio(raw, codec), 4),
            "file_size_mb": round(get_s3_size_mb(src_bucket, raw_key + "/"), 4),
            "ssim": 1.0,
        })
        print(f"  {stem}: cratio={raw_rows[-1]['cratio']:.2f}x  size={raw_rows[-1]['file_size_mb']:.2f} MB")

    # Write raw per-block CSV (so downstream plots can split by sparse/dense)
    raw_csv_path = os.path.join(output_dir, "raw.csv")
    with open(raw_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["block", "cratio", "file_size_mb", "ssim"])
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"  Raw CSV written: {raw_csv_path}")

    def mean_std(rows, key):
        vals = [r[key] for r in rows]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    cr_mean, cr_std = mean_std(raw_rows, "cratio")
    sz_mean, sz_std = mean_std(raw_rows, "file_size_mb")
    summary_rows = [{
        "experiment": "raw",
        "cratio_mean": cr_mean, "cratio_std": cr_std,
        "file_size_mb_mean": sz_mean, "file_size_mb_std": sz_std,
        "ssim_mean": 1.0, "ssim_std": 0.0,
    }]

    # List sigma* experiment subdirs
    search_prefix = dst_prefix.rstrip("/") + "/"
    exp_prefixes = sorted(
        sp for sp in list_s3_prefixes(dst_bucket, search_prefix)
        if sp.rstrip("/").split("/")[-1].startswith("sigma")
    )

    for exp_prefix in exp_prefixes:
        exp_name = exp_prefix.rstrip("/").split("/")[-1]
        is_uint8 = exp_name.endswith("uint8")
        print(f"\n{'=' * 60}")
        print(f"Experiment: {exp_name}")
        print(f"{'=' * 60}")

        zarr_entries = list_zarr_paths(dst_bucket, exp_prefix.rstrip("/"))
        rows = []

        for zarr_path, rel_path in zarr_entries:
            stem = rel_path.split("/")[-1].removesuffix(".zarr")
            print(f"  Block: {stem}")

            denoised = np.squeeze(np.asarray(read_image(zarr_path + "/0")))

            raw_s3_path = raw_lookup.get(rel_path)
            if raw_s3_path is None:
                print(f"    [SKIP] No matching raw for {rel_path}")
                continue
            raw = np.squeeze(np.asarray(read_image(raw_s3_path + "/0")))

            # Convert to the same dtype space for SSIM
            if is_uint8:
                p999 = np.percentile(raw.astype(np.float32), 99.9)
                raw_cmp = to_uint8(raw, p999=p999)
                denoised_cmp = denoised  # already uint8 on disk
                data_range = 255
            else:
                raw_cmp = raw.astype(np.uint16)
                denoised_cmp = denoised.astype(np.uint16)
                data_range = 65535

            zarr_key = zarr_path.removeprefix(f"s3://{dst_bucket}/")
            cratio = compute_cratio(denoised, codec)
            file_size_mb = get_s3_size_mb(dst_bucket, zarr_key + "/")
            ssim_val = structural_similarity(raw_cmp, denoised_cmp, data_range=data_range)

            print(f"    cratio={cratio:.2f}x  size={file_size_mb:.2f} MB  ssim={ssim_val:.4f}")
            rows.append({
                "block": stem,
                "cratio": round(cratio, 4),
                "file_size_mb": round(file_size_mb, 4),
                "ssim": round(ssim_val, 4),
            })

        # Per-experiment CSV
        csv_path = os.path.join(output_dir, f"{exp_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["block", "cratio", "file_size_mb", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  CSV written: {csv_path}")

        if rows:
            cr_mean, cr_std = mean_std(rows, "cratio")
            sz_mean, sz_std = mean_std(rows, "file_size_mb")
            ss_mean, ss_std = mean_std(rows, "ssim")
            summary_rows.append({
                "experiment": exp_name,
                "cratio_mean": cr_mean, "cratio_std": cr_std,
                "file_size_mb_mean": sz_mean, "file_size_mb_std": sz_std,
                "ssim_mean": ss_mean, "ssim_std": ss_std,
            })

    # Summary CSV
    summary_path = os.path.join(output_dir, "summary.csv")
    fieldnames = [
        "experiment",
        "cratio_mean", "cratio_std",
        "file_size_mb_mean", "file_size_mb_std",
        "ssim_mean", "ssim_std",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        raw_row = [r for r in summary_rows if r["experiment"] == "raw"]
        exp_rows = sorted(
            (r for r in summary_rows if r["experiment"] != "raw"),
            key=lambda r: r["experiment"],
        )
        writer.writerows(raw_row + exp_rows)
    print(f"\nSummary CSV written: {summary_path}")

    # Print summary table
    if summary_rows:
        col_w = max(len(r["experiment"]) for r in summary_rows) + 2
        header = f"{'Experiment':<{col_w}} {'Cratio':>20} {'Size (MB)':>20} {'SSIM':>20}"
        print("\n" + header)
        print("-" * len(header))
        for r in summary_rows:
            print(
                f"{r['experiment']:<{col_w}}"
                f"  {r['cratio_mean']:.2f} ± {r['cratio_std']:.2f}"
                f"  {r['file_size_mb_mean']:.2f} ± {r['file_size_mb_std']:.2f}"
                f"  {r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}"
            )

    write_summary_md(summary_rows, output_dir)


def write_summary_md(summary_rows, output_dir):
    """Write a Markdown report of summary metrics."""
    from datetime import date

    def fmt(mean, std, decimals=2):
        return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

    def bold_best(value, best_value, decimals=2):
        s = f"{value:.{decimals}f}"
        return f"**{s}**" if value == best_value else s

    exp_rows = [r for r in summary_rows if r["experiment"] != "raw"]
    raw_row  = next((r for r in summary_rows if r["experiment"] == "raw"), None)

    best_cratio  = max((r["cratio_mean"]       for r in exp_rows), default=None)
    best_ssim    = max((r["ssim_mean"]          for r in exp_rows), default=None)
    best_size    = min((r["file_size_mb_mean"]  for r in exp_rows), default=None)

    lines = [
        "# BM4D Denoising — Summary Results",
        "",
        f"**Dataset:** mesoscale-anatomy-cell-detection  ",
        f"**Date:** {date.today().isoformat()}  ",
        f"**Metric definitions:** compression ratio (higher = better), "
        f"SSIM vs raw image (higher = better), compressed file size on S3 (lower = better).",
        "",
        "---",
        "",
        "## Results",
        "",
        "| Experiment | Compression Ratio | File Size (MB) | SSIM |",
        "|:-----------|------------------:|---------------:|-----:|",
    ]

    def table_row(r):
        exp = r["experiment"]
        if exp == "raw":
            cratio = fmt(r["cratio_mean"], r["cratio_std"])
            size   = fmt(r["file_size_mb_mean"], r["file_size_mb_std"])
            ssim   = fmt(r["ssim_mean"], r["ssim_std"], decimals=4)
        else:
            cratio = bold_best(r["cratio_mean"], best_cratio) + f" ± {r['cratio_std']:.2f}"
            size   = bold_best(r["file_size_mb_mean"], best_size) + f" ± {r['file_size_mb_std']:.2f}"
            ssim   = bold_best(r["ssim_mean"], best_ssim, decimals=4) + f" ± {r['ssim_std']:.4f}"
        return f"| `{exp}` | {cratio} | {size} | {ssim} |"

    if raw_row:
        lines.append(table_row(raw_row))
    for r in sorted(exp_rows, key=lambda r: r["experiment"]):
        lines.append(table_row(r))

    lines += [
        "",
        "> Best values across sigma experiments are **bolded**.",
        "",
        "---",
        "",
        "## Per-experiment CSVs",
        "",
    ]
    for r in sorted(exp_rows, key=lambda r: r["experiment"]):
        lines.append(f"- [`{r['experiment']}.csv`]({r['experiment']}.csv)")

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Summary report written: {md_path}")


if __name__ == "__main__":
    source_prefix = "s3://aind-benchmark-data/mesoscale-anatomy-cell-detection/test/"
    dest_prefix = "s3://aind-msma-morphology-data/anna.grim/image_compression/smartspim_bm4d/mesoscale-anatomy-cell-detection"
    output_dir = "/results/bm4d_analysis"
    analyze(source_prefix, dest_prefix, output_dir)
