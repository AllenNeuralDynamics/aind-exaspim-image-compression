import argparse
import ast
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from numcodecs import blosc

from aind_exaspim_image_compression.inference import (
    build_volume_transform,
    load_model,
    predict,
)
from aind_exaspim_image_compression.utils import img_util, util


def find_session_dirs(root):
    """
    Return all directories under root that directly contain BM4DNet-*.pth
    checkpoints, sorted for deterministic ordering.
    """
    pth_files = glob.glob(os.path.join(root, "**", "BM4DNet-*.pth"), recursive=True)
    dirs = sorted(set(os.path.dirname(p) for p in pth_files))
    if not dirs:
        raise FileNotFoundError(
            f"No BM4DNet-*.pth checkpoints found under {root}"
        )
    return dirs


def find_top_k_checkpoints(session_dir, k):
    """
    Returns the last k checkpoints saved in a training session, ordered by
    epoch (most recent first).

    Checkpoints are named ``BM4DNet-<date>-<update>-<score>.pth`` by
    Trainer.save_model. The update number is parsed to determine save order;
    higher update == saved later.

    Parameters
    ----------
    session_dir : str
        Training session directory holding the BM4DNet-*.pth checkpoints.
    k : int
        Number of checkpoints to return. Clamped to the number available.

    Returns
    -------
    list of str
        Paths to the k most-recently-saved checkpoints, latest first.
    """
    paths = glob.glob(os.path.join(session_dir, "BM4DNet-*.pth"))
    if not paths:
        raise FileNotFoundError(
            f"No BM4DNet-*.pth checkpoints found in {session_dir}"
        )

    def epoch(path):
        m = re.search(r"BM4DNet-\d{8}-(\d+)-", os.path.basename(path))
        if m is None:
            raise ValueError(f"Cannot parse update number from checkpoint: {path}")
        return int(m.group(1))

    return sorted(paths, key=epoch, reverse=True)[:k]


def load_blocks(csv_path):
    """
    Parse the eval dataset spreadsheet, returning a list of block dicts.

    Each dict has keys:
        brain_id    (int) specimen identifier
        voxel       (tuple of int) voxel (z, y, x) coordinate

    Parameters
    ----------
    csv_path : str
        Path to the evaluation CSV file (e.g. evaluation_alpha_dataset.csv).
        The first row is used as the column header.

    Returns
    -------
    List[dict]
    """
    blocks = []
    for _, row in pd.read_csv(csv_path).iterrows():
        coords = ast.literal_eval(row["Voxel"])
        blocks.append(
            {
                "brain_id": int(row["Brain ID"]),
                "voxel": tuple(coords),
            }
        )
    return blocks


def evaluate_block(
    img_path,
    crop_center,
    model,
    transform,
    background_offset,
    output_dir,
    batch_size=32,
    clevel=5,
    crop_shape=(1024, 1024, 1024),
    output_zarr=None,
):
    """
    Evaluate a single block. Model and transform must be pre-loaded.

    Parameters
    ----------
    img_path : str
        Path to the source OME-Zarr level (s3://, gs://, or local). Must point
        to a single 5D (t, c, z, y, x) multiscale level array (e.g. ".../0").
    crop_center : tuple of float
        Neuroglancer (x, y, z) coordinate of the block centre.
    model : torch.nn.Module
        Loaded denoising model in eval mode.
    transform : IntensityTransform
        Intensity transform the model was trained with.
    background_offset : float or None
        Precomputed full-tile background offset. None falls back to estimating
        from this subvolume.
    output_dir : str
        Directory for MIP images and any other per-block outputs.
    output_zarr : str or None
        Destination path for the denoised OME-Zarr (local or cloud). Skipped
        when None.
    crop_shape : tuple of int, optional
        (z, y, x) size of the region to evaluate. Default is (1024, 1024, 1024).
    batch_size : int, optional
        Patch batch size for predict(). Default is 32.
    clevel : int, optional
        Blosc zstd compression level for cratio measurement. Default is 5.
    """
    # Load image and metadata
    img = img_util.read(img_path)
    raw = img_util.get_patch(img, crop_center, crop_shape, is_center=True)
    raw = np.asarray(raw)
    print("Image Shape:", raw.shape)

    source_transform = img_util.get_ome_zarr_level_transform(img_path)
    source_scale = np.asarray(source_transform["scale"])
    source_translation = np.asarray(source_transform["translation"])

    # Load image transform
    volume_transform = build_volume_transform(
        transform,
        raw,
        percentile=0.1,
        offset=background_offset,
    )
    print("Per-volume transform:", volume_transform.cfg)

    # Denoise the whole volume via overlapping tiled prediction.
    denoised = predict(raw, model, volume_transform, batch_size=batch_size)

    # Compute compression ratios
    codec = blosc.Blosc(cname="zstd", clevel=clevel, shuffle=blosc.SHUFFLE)
    raw_cratio = img_util.compute_cratio(raw, codec)
    denoised_cratio = img_util.compute_cratio(denoised, codec)
    print(f"cratio (raw):      {raw_cratio}")
    print(f"cratio (denoised): {denoised_cratio}")
    print(f"cratio gain:       {denoised_cratio / raw_cratio:.2f}x")

    # Save MIPs of raw and denoised
    util.mkdir(output_dir)
    img_util.plot_mips(
        raw, output_path=os.path.join(output_dir, "raw_mips.png")
    )
    img_util.plot_mips(
        denoised, output_path=os.path.join(output_dir, "denoised_mips.png")
    )
    print("MIPs written to:", output_dir)

    # Save result for neuroglancer visualization
    if output_zarr is not None:
        crop_start = np.asarray(crop_center) - np.asarray(crop_shape) // 2
        crop_offset = np.asarray([0, 0, *crop_start])
        output_translation = source_translation + source_scale * crop_offset
        print(
            "Output OME transform (t, c, z, y, x):",
            {
                "scale": tuple(source_scale.tolist()),
                "translation": tuple(output_translation.tolist()),
                "unit": source_transform["spatial_unit"],
            },
        )
        img_util.write_ome_zarr(
            denoised,
            output_zarr,
            scale=source_scale,
            translation=output_translation,
            spatial_unit=source_transform["spatial_unit"],
        )
        print("Denoised Zarr written to:", output_zarr)

    return raw_cratio, denoised_cratio


def evaluate():
    # Load shared resources once
    with open(img_prefixes_path) as f:
        img_prefixes = json.load(f)
    with open(background_offsets_path) as f:
        background_offsets = json.load(f)

    blocks = load_blocks(csv_path)
    print(f"Loaded {len(blocks)} blocks from {csv_path}\n")

    for session_dir in session_dirs:
        session_name = os.path.basename(session_dir.rstrip("/"))
        print(f"\n{'#' * 60}")
        print(f"Session: {session_name}")
        print(f"{'#' * 60}")

        try:
            ckpt_paths = find_top_k_checkpoints(session_dir, top_k)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        print(f"Checkpoints to evaluate ({len(ckpt_paths)}):")
        for p in ckpt_paths:
            print(" ", p)

        session_output_dir = os.path.join(output_dir, session_name)
        session_output_zarr = (
            f"{base_output_zarr}/{session_name}" if base_output_zarr else None
        )
        util.mkdir(session_output_dir)

        all_rows = []

        for ckpt_path in ckpt_paths:
            ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            print(f"\n{'=' * 60}")
            print(f"Checkpoint: {ckpt_name}")
            print(f"{'=' * 60}")

            model, transform = load_model(ckpt_path, device="cuda")
            print("Transform:", transform.cfg)

            ckpt_output_dir = os.path.join(session_output_dir, ckpt_name)
            ckpt_output_zarr = (
                f"{session_output_zarr}/{ckpt_name}" if session_output_zarr else None
            )
            util.mkdir(ckpt_output_dir)

            ckpt_rows = []
            for i, block in enumerate(blocks):
                brain_id = block["brain_id"]
                crop_center = block["voxel"]
                brain_id_str = str(brain_id)

                img_path = os.path.join(img_prefixes.get(brain_id_str), str(0))
                if img_path is None:
                    print(
                        f"[SKIP] Block {i + 1}/{len(blocks)}: brain_id={brain_id}"
                        " (not found in img_prefixes)"
                    )
                    continue

                background_offset = background_offsets.get(brain_id_str)
                block_id = f"{brain_id}_block{i:02d}"
                block_output_dir = os.path.join(ckpt_output_dir, block_id)
                block_output_zarr = (
                    f"{ckpt_output_zarr}/{block_id}.zarr"
                    if ckpt_output_zarr else None
                )
                util.mkdir(block_output_dir)

                print(f"\n--- Block {i + 1}/{len(blocks)}: {block_id} ---")
                print("img_path:", img_path)
                print("crop_center:", crop_center)
                print("background_offset:", background_offset)
                try:
                    raw_cratio, denoised_cratio = evaluate_block(
                        img_path=img_path,
                        crop_center=crop_center,
                        model=model,
                        transform=transform,
                        background_offset=background_offset,
                        output_dir=block_output_dir,
                        output_zarr=block_output_zarr,
                        crop_shape=crop_shape,
                        batch_size=batch_size,
                        clevel=clevel,
                    )
                    ckpt_rows.append(
                        {
                            "block_id": block_id,
                            "brain_id": brain_id,
                            "raw_cratio": raw_cratio,
                            "denoised_cratio": denoised_cratio,
                            "cratio_gain": denoised_cratio / raw_cratio,
                        }
                    )
                except Exception as e:
                    print(f"[ERROR] Block {block_id} failed: {e}")

            if ckpt_rows:
                util.mkdir(ckpt_output_dir)
                ckpt_table_path = os.path.join(ckpt_output_dir, "cratio_results.csv")
                pd.DataFrame(ckpt_rows).to_csv(ckpt_table_path, index=False)
                print(f"\nPer-block table written to: {ckpt_table_path}")
                all_rows.append({"checkpoint": ckpt_name, "rows": ckpt_rows})

        if all_rows:
            summary_rows = []
            for entry in all_rows:
                vals = {
                    col: [r[col] for r in entry["rows"]]
                    for col in ("raw_cratio", "denoised_cratio", "cratio_gain")
                }
                summary_rows.append(
                    {
                        "checkpoint": entry["checkpoint"],
                        "raw_cratio_mean": np.mean(vals["raw_cratio"]),
                        "raw_cratio_std": np.std(vals["raw_cratio"]),
                        "denoised_cratio_mean": np.mean(vals["denoised_cratio"]),
                        "denoised_cratio_std": np.std(vals["denoised_cratio"]),
                        "cratio_gain_mean": np.mean(vals["cratio_gain"]),
                        "cratio_gain_std": np.std(vals["cratio_gain"]),
                    }
                )
            summary_path = os.path.join(session_output_dir, "cratio_summary.csv")
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            print(f"\nSummary table written to: {summary_path}")


if __name__ == "__main__":
    # Parse arguments
    def str_to_bool(v: str) -> bool:
        if v.lower() in ("true", "1", "yes"):
            return True
        if v.lower() in ("false", "0", "no"):
            return False
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v!r}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--crop_shape", type=str, default="1024,1024,1024")
    parser.add_argument("--clevel", type=int, default=5)
    parser.add_argument("--top_k_ckpts", type=int, default=1)
    parser.add_argument("--upload_to_s3", type=str_to_bool, default=False)
    args = parser.parse_args()

    # Parameters
    batch_size = 32
    clevel = args.clevel
    crop_shape = tuple(int(x.strip()) for x in args.crop_shape.split(","))
    top_k = args.top_k_ckpts

    # Paths
    background_offsets_path = "/data/exaspim_background_offsets.json"
    img_prefixes_path = "/data/exaspim_image_prefixes.json"
    output_dir = "/results/"
    session_root = "/data/denoise_net_v2"
    csv_path = f"/data/denoise_evaluation_datasets/evaluation_{args.dataset_name}_dataset.csv"
    util.mkdir(output_dir)

    # S3 upload path
    print(type(args.upload_to_s3), args.upload_to_s3)
    base_output_zarr = (
        f"s3://aind-scratch-data/anna.grim/denoising-experiments/model_evaluation_{args.dataset_name}_V2"
        if args.upload_to_s3 else None
    )

    # Discover all session directories
    session_dirs = find_session_dirs(session_root)
    print(f"Found {len(session_dirs)} session(s) under {session_root}:")
    for d in session_dirs:
        print(" ", d)

    # Run evaluation
    evaluate()
