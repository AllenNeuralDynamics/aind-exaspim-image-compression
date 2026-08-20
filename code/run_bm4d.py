"""
Applies BM4D denoising to all .zarr images under an S3 prefix at sigma
values 10, 20, 30, 40 and writes the results to a second S3 prefix.

The input zarrs are OME-Zarr multiscale groups; this script reads the full-
resolution level (key "0"). Output zarrs are plain single-level zarr arrays
with Blosc/zstd compression, written to the destination prefix while
preserving the same relative directory structure.
"""

import json
import sys
from urllib.parse import quote

import boto3
import numpy as np
import zarr
from bm4d import bm4d
from numcodecs import blosc

from aind_exaspim_image_compression.utils.img_util import compute_cratio, read as read_image
from aind_exaspim_image_compression.utils.util import list_s3_prefixes, parse_cloud_path

NG_BASE_URL = "https://neuroglancer-demo.appspot.com/#!"


def main(source_prefix, dest_prefix, cast_to_uint8=False):
    # Parse cloud prefixes
    src_bucket, src_prefix = parse_cloud_path(source_prefix)
    dst_bucket, dst_prefix = parse_cloud_path(dest_prefix)
    codec = blosc.Blosc(cname="zstd", clevel=5, shuffle=blosc.SHUFFLE)

    zarr_entries = list_zarr_paths(src_bucket, src_prefix)
    if not zarr_entries:
        print(f"No .zarr directories found under s3://{src_bucket}/{src_prefix}")
        sys.exit(0)

    # Denoise images
    n = len(zarr_entries)
    print(f"Found {n} zarr image(s)\n")
    results = {}   # {stem: {"raw": float, sigma: float, ...}}
    ng_links = {}  # {stem: url}

    for i, (zarr_path, rel_path) in enumerate(zarr_entries):
        # rel_path looks like "668647/Ex_488_Em_525_Dense.zarr"
        rel_dir = "/".join(rel_path.split("/")[:-1])   # "668647"
        zarr_name = rel_path.split("/")[-1]             # "Ex_488_Em_525_Dense.zarr"
        stem = zarr_name.removesuffix(".zarr")          # "Ex_488_Em_525_Dense"

        print(f"[{i + 1}/{n}] {zarr_name}")
        print(f"  Reading {zarr_path} (level 0)")
        volume = read_image(zarr_path + "/0")

        # BM4D expects a 3-D array; squeeze out T/C singleton dimensions
        vol3d = np.squeeze(volume)
        if vol3d.ndim != 3:
            print(f"  Skipping: expected 3-D after squeeze, got shape {volume.shape}")
            continue

        img_dtype = "uint8" if cast_to_uint8 else "uint16"
        print(f"  Shape: {vol3d.shape}, dtype: {vol3d.dtype}")

        vol3d_stored = to_storage_dtype(vol3d, cast_to_uint8)
        img_size_mb = vol3d_stored.nbytes / 1e6
        raw_cratio = compute_cratio(vol3d_stored, codec)
        print(f"  Size: {img_size_mb:.2f} MB")
        print(f"  Raw cratio: {raw_cratio:.2f}x")
        results[stem] = {"img_size_mb": img_size_mb, "raw": raw_cratio}

        denoised_paths = {}
        for j, sigma in enumerate(sigmas):
            label = f"sigma{sigma}_{img_dtype}"
            rel_out = f"{rel_dir}/{stem}.zarr" if rel_dir else f"{stem}.zarr"
            output_path = f"s3://{dst_bucket}/{dst_prefix}/{label}/{rel_out}"
            print(f"\n  [{j + 1}/{len(sigmas)}] sigma={sigma}")
            print(f"    Denoising ...")
            denoised = to_storage_dtype(denoise_volume(vol3d, sigma), cast_to_uint8)
            cratio = compute_cratio(denoised, codec)
            results[stem][sigma] = cratio
            denoised_paths[label] = output_path
            print(f"    Writing to {output_path} ...")
            write_zarr_volume(denoised, output_path)
            print(f"    Done  |  block={stem}  sigma={sigma}  cratio={cratio:.2f}x")

        ng_link = generate_ng_link(zarr_path, denoised_paths, src_bucket, rel_dir, stem)
        ng_links[stem] = ng_link
        print(f"\n  Neuroglancer: {ng_link}")
        print()

    # Print summary table
    col_w = max(len(s) for s in results) + 2
    header = f"{'Image':<{col_w}} {'Size (MB)':>10} {'Raw':>8}" + "".join(f"  sigma={s:>2}" for s in sigmas)
    print("\n" + header)
    print("-" * len(header))
    for stem, cratios in sorted(results.items()):
        row = f"{stem:<{col_w}} {cratios['img_size_mb']:>9.2f}  {cratios['raw']:>7.2f}x"
        row += "".join(f"  {cratios[s]:>8.2f}x" for s in sigmas)
        print(row)

    print("\nNeuroglancer links (this run):")
    for stem, link in ng_links.items():
        print(f"  {stem}: {link}")

    print("\nNeuroglancer comparison links (all subdirs under dest_prefix):")
    dest_prefix = f"s3://{dst_bucket}/{dst_prefix}"
    for (zarr_path, rel_path) in zarr_entries:
        rel_dir = "/".join(rel_path.split("/")[:-1])
        stem = rel_path.split("/")[-1].removesuffix(".zarr")
        link = generate_comparison_ng_link(zarr_path, dest_prefix, src_bucket, rel_dir, stem)
        print(f"  {stem}: {link}")

    print("\nAll done.")


# --- Helpers ---
def s3_to_https(s3_path):
    """Convert s3://bucket/key to https://bucket.s3.amazonaws.com/key."""
    path = s3_path.removeprefix("s3://")
    bucket, _, key = path.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def generate_ng_link(raw_zarr_path, denoised_paths, src_bucket, rel_dir, stem):
    """
    Build a Neuroglancer link with the raw image and one layer per sigma.

    Reads the existing neuroglancer config JSON from the source bucket to
    inherit the voxel dimensions, then appends a layer for each denoised output.
    Sources use the zarr://https:// format that neuroglancer-demo.appspot.com expects.
    """
    s3 = boto3.client("s3")

    # Try to read the existing config to get voxel dimensions
    img_type = stem.split("_")[-1]  # "Dense" or "Sparse"
    config_key = f"mesoscale-anatomy-cell-detection/test/{rel_dir}/neuroglancer_config_{img_type}.json"
    try:
        obj = s3.get_object(Bucket=src_bucket, Key=config_key)
        config = json.loads(obj["Body"].read())
        dimensions = config.get("dimensions", {"z": [2e-6, "m"], "y": [1.8e-6, "m"], "x": [1.8e-6, "m"]})
    except Exception:
        dimensions = {"z": [2e-6, "m"], "y": [1.8e-6, "m"], "x": [1.8e-6, "m"]}

    layers = [
        {"type": "image", "source": f"zarr://{s3_to_https(raw_zarr_path)}", "name": "raw"},
    ] + [
        {"type": "image", "source": f"zarr://{s3_to_https(path)}", "name": label, "visible": False}
        for label, path in denoised_paths.items()
    ]

    state = {
        "dimensions": dimensions,
        "layers": layers,
        "selectedLayer": {"visible": True, "layer": "raw"},
        "layout": "4panel",
    }
    return NG_BASE_URL + quote(json.dumps(state))


def generate_comparison_ng_link(raw_zarr_path, dest_prefix, src_bucket, rel_dir, stem):
    """
    Build a Neuroglancer link for a single image that includes one layer per
    experiment subdir (sigma{value}_{dtype}) found under dest_prefix.

    Useful for comparing uint8 and uint16 runs side-by-side after both
    experiments have finished.
    """
    dst_bucket, dst_prefix_key = parse_cloud_path(dest_prefix)
    search_prefix = dst_prefix_key.rstrip("/") + "/"
    denoised_paths = {}
    for subprefix in list_s3_prefixes(dst_bucket, search_prefix):
        subdir = subprefix.rstrip("/").split("/")[-1]
        if not subdir.startswith("sigma"):
            continue
        rel_out = f"{rel_dir}/{stem}.zarr" if rel_dir else f"{stem}.zarr"
        path = f"s3://{dst_bucket}/{subprefix.rstrip('/')}/{rel_out}"
        denoised_paths[subdir] = path
    return generate_ng_link(raw_zarr_path, denoised_paths, src_bucket, rel_dir, stem)


def write_zarr_volume(img, output_path, voxel_size_um=(2.0, 1.8, 1.8)):
    """Write img to S3 as an OME-Zarr v2 group matching the raw image format.

    Creates a .zgroup + .zattrs (multiscales) at output_path and writes the
    array at output_path/0 with "/" chunk separators, matching the raw zarr.
    """
    while img.ndim < 5:
        img = img[np.newaxis, ...]

    # Write array at level "0" with "/" separator to match raw format
    zarr.create_array(
        store=output_path.rstrip("/") + "/0",
        shape=img.shape,
        chunks=(1, 1, 64, 64, 64),
        dtype=img.dtype,
        zarr_format=2,
        chunk_key_encoding={"name": "v2", "separator": "/"},
        compressors=blosc.Blosc(cname="zstd", clevel=5, shuffle=blosc.SHUFFLE),
        overwrite=True,
    )[:] = img

    # Write OME-Zarr group metadata so neuroglancer can read it as a multiscale group
    s3 = boto3.client("s3")
    bucket, prefix = parse_cloud_path(output_path)
    prefix = prefix.rstrip("/")

    s3.put_object(Bucket=bucket, Key=f"{prefix}/.zgroup", Body=json.dumps({"zarr_format": 2}))

    z_um, y_um, x_um = voxel_size_um
    multiscales = [{
        "version": "0.4",
        "name": "/",
        "axes": [
            {"name": "t", "type": "time", "unit": "millisecond"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, z_um, y_um, x_um]}
            ]
        }]
    }]
    s3.put_object(Bucket=bucket, Key=f"{prefix}/.zattrs", Body=json.dumps({"multiscales": multiscales}))


def to_storage_dtype(volume, cast_to_uint8):
    """Optionally rescale to uint8 via percentile-based clipping; otherwise return uint16."""
    if cast_to_uint8:
        vol = volume.astype(np.float32)
        p1, p999 = np.percentile(vol, [1, 99.9])
        vol = np.clip(vol - p1, 0, p999 - p1)
        return (vol * (255.0 / (p999 - p1))).astype(np.uint8)
    return volume.astype(np.uint16)


def denoise_volume(volume_3d, sigma):
    """Run BM4D on a uint16 3-D array and return a non-negative uint16 result."""
    denoised = bm4d(volume_3d, sigma)
    return np.maximum(denoised, 0).astype(np.uint16)


def list_zarr_paths(bucket_name, prefix):
    """
    Recursively find all .zarr directories under prefix and return their
    s3:// paths along with the relative path from prefix (used to mirror
    the directory structure in the destination).
    """
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


if __name__ == "__main__":
    # Parameters
    cast_to_uint8 = True
    sigmas = [5] #[10, 20, 50, 100]
    dataset_name = "mesoscale-anatomy-cell-detection"

    # Paths
    source_prefix = "s3://aind-benchmark-data/mesoscale-anatomy-cell-detection/test/"
    dst_prefix = f"s3://aind-msma-morphology-data/anna.grim/image_compression/smartspim_bm4d/{dataset_name}"

    # Run code
    main(source_prefix, dst_prefix, cast_to_uint8=cast_to_uint8)
