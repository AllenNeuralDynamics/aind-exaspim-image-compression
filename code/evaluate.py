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


def find_top_k_checkpoints(session_dir, k):
    """
    Returns the top-k checkpoints from a training session, ranked by score.

    Checkpoints are named ``BM4DNet-<date>-<epoch>-<score>.pth`` by
    Trainer.save_model, where <score> is the checkpoint-selection score at save
    time and lower is better. The score can be negative (the cratio term is
    subtracted), so it is parsed with a regex that allows a leading minus rather
    than by splitting on "-".

    Parameters
    ----------
    session_dir : str
        Training session directory holding the BM4DNet-*.pth checkpoints.
    k : int
        Number of checkpoints to return. Clamped to the number available.

    Returns
    -------
    list of str
        Paths to the k lowest-scoring (best) checkpoints, best first.
    """
    paths = glob.glob(os.path.join(session_dir, "BM4DNet-*.pth"))
    if not paths:
        raise FileNotFoundError(
            f"No BM4DNet-*.pth checkpoints found in {session_dir}"
        )

    def score(path):
        m = re.search(r"-(-?\d+\.\d+)\.pth$", os.path.basename(path))
        if m is None:
            raise ValueError(f"Cannot parse score from checkpoint: {path}")
        return float(m.group(1))

    return sorted(paths, key=score)[:k]


def find_best_checkpoint(session_dir):
    """Returns the single best checkpoint from a training session."""
    return find_top_k_checkpoints(session_dir, k=1)[0]


def load_blocks(xlsx_path):
    """
    Parse the eval dataset spreadsheet, returning a list of block dicts.

    Each dict has keys:
        brain_id    (int) specimen identifier
        crop_center (tuple of float) Neuroglancer (x, y, z) coordinate

    Parameters
    ----------
    xlsx_path : str
        Path to the denoise_eval_dataset.xlsx file. The first row in the file
        is a section label; the actual column header is in row 1.

    Returns
    -------
    list of dict
    """
    df = pd.read_excel(xlsx_path, header=1)
    blocks = []
    for _, row in df.iterrows():
        coords = ast.literal_eval(row["XYZ"])
        blocks.append(
            {
                "brain_id": int(row["Brain ID"]),
                "crop_center": tuple(coords),
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
    output_zarr=None,
    crop_shape=(1024, 1024, 1024),
    raw_input=True,
    batch_size=32,
    clevel=5,
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
    raw_input : bool, optional
        True for volumes that were not background-subtracted. Default is True.
    batch_size : int, optional
        Patch batch size for predict(). Default is 32.
    clevel : int, optional
        Blosc zstd compression level for cratio measurement. Default is 5.
    """
    img = img_util.read(img_path)
    source_transform = img_util.get_ome_zarr_level_transform(img_path)
    source_scale = np.asarray(source_transform["scale"])
    source_translation = np.asarray(source_transform["translation"])

    crop_start = (0, 0, 0)
    if crop_center is not None:
        # Neuroglancer reports transformed spatial coordinates in (x, y, z)
        # order. Convert them to this level's integer (z, y, x) voxel indices.
        crop_center_voxel = img_util.ome_zarr_coordinate_to_voxel(
            crop_center, source_transform
        )
        source_offset_zyx = source_translation[2:] / source_scale[2:]
        snapped_center_zyx = source_offset_zyx + np.asarray(crop_center_voxel)
        crop_start, _ = img_util.get_start_end(
            crop_center_voxel, crop_shape, is_center=True
        )
        crop_origin_zyx = source_offset_zyx + np.asarray(crop_start)
        crop_end = np.asarray(crop_start) + np.asarray(crop_shape)
        if np.any(np.asarray(crop_start) < 0) or np.any(
            crop_end > np.asarray(img.shape[2:])
        ):
            raise ValueError(
                "Crop is outside the source level: "
                f"start={tuple(crop_start)}, end={tuple(crop_end)}, "
                f"source_shape={tuple(img.shape[2:])}"
            )
        print(
            "Requested Neuroglancer crop center (x, y, z):",
            tuple(crop_center),
        )
        print(
            "Neuroglancer spatial scale (x, y, z):",
            tuple(source_scale[2:][::-1].tolist()),
            source_transform["spatial_unit"],
        )
        print("Crop center voxel (z, y, x):", crop_center_voxel)
        print(
            "Snapped crop center (x, y, z):",
            tuple(snapped_center_zyx[::-1].tolist()),
        )
        print("Crop origin (z, y, x):", tuple(crop_start))
        print(
            "Neuroglancer crop origin (x, y, z):",
            tuple(crop_origin_zyx[::-1].tolist()),
        )
        raw = np.asarray(
            img_util.get_patch(
                img, crop_center_voxel, crop_shape, is_center=True
            )
        )
    else:
        raw = np.asarray(img[0, 0])
    print("Volume shape:", raw.shape)

    # For a raw (non-background-subtracted) volume, use the supplied full-tile
    # offset. With background_offset=None, fall back to estimating from this
    # test subvolume for debugging only.
    if raw_input:
        volume_transform = build_volume_transform(
            transform,
            raw,
            percentile=0.1,
            offset=background_offset,
        )
        print("Per-volume transform:", volume_transform.cfg)
    else:
        volume_transform = transform

    # Denoise the whole volume via overlapping tiled prediction.
    denoised = predict(raw, model, volume_transform, batch_size=batch_size)

    # Compression ratio, raw vs denoised, with the codec Zarr uses to store
    # chunks. clevel=5 matches the training-time codec (train.py).
    codec = blosc.Blosc(cname="zstd", clevel=clevel, shuffle=blosc.SHUFFLE)
    raw_cratio = img_util.compute_cratio(raw, codec)
    denoised_cratio = img_util.compute_cratio(denoised, codec)
    print(f"cratio (raw):      {raw_cratio}")
    print(f"cratio (denoised): {denoised_cratio}")
    print(f"cratio gain:       {denoised_cratio / raw_cratio:.2f}x")

    # Save side-by-side MIPs (XY/XZ/YZ) of the raw and denoised volumes.
    util.mkdir(output_dir)
    img_util.plot_mips(
        raw, output_path=os.path.join(output_dir, "raw_mips.png")
    )
    img_util.plot_mips(
        denoised, output_path=os.path.join(output_dir, "denoised_mips.png")
    )
    print("MIPs written to:", output_dir)

    if output_zarr is not None:
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
    # Resolve the list of checkpoints to evaluate. An explicit checkpoint_path
    # always wins; otherwise the top-k from session_dir are selected.
    if checkpoint_path is not None:
        ckpt_paths = [checkpoint_path]
    else:
        ckpt_paths = find_top_k_checkpoints(session_dir, top_k)
    print(f"Checkpoints to evaluate ({len(ckpt_paths)}):")
    for p in ckpt_paths:
        print(" ", p)

    # Load per-brain lookup tables. Keys are brain_id strings in both files.
    with open(img_prefixes_path) as f:
        img_prefixes = json.load(f)
    with open(background_offsets_path) as f:
        background_offsets = json.load(f)

    blocks = load_blocks(xlsx_path)
    print(f"\nLoaded {len(blocks)} blocks from {xlsx_path}\n")

    all_rows = []

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n{'=' * 60}")
        print(f"Checkpoint: {ckpt_name}")
        print(f"{'=' * 60}")

        # Load the model once per checkpoint; reuse it for every block.
        model, transform = load_model(ckpt_path, device=device)
        print("Transform:", transform.cfg)

        ckpt_output_dir = os.path.join(output_dir, ckpt_name)
        ckpt_output_zarr = (
            f"{base_output_zarr}/{ckpt_name}" if base_output_zarr else None
        )

        ckpt_rows = []
        for i, block in enumerate(blocks):
            brain_id = block["brain_id"]
            crop_center = block["crop_center"]
            brain_id_str = str(brain_id)

            img_path = img_prefixes.get(brain_id_str)
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

            print(f"\n--- Block {i + 1}/{len(blocks)}: {block_id} ---")
            print("img_path:", img_path)
            print("crop_center (x, y, z):", crop_center)
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
                    raw_input=raw_input,
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
        util.mkdir(output_dir)
        summary_path = os.path.join(output_dir, "cratio_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary table written to: {summary_path}")


if __name__ == "__main__":
    # Checkpoint. Set checkpoint_path to a .pth to evaluate that file
    # explicitly (top_k is ignored). Otherwise point session_dir at a training
    # session and set top_k to evaluate that many of the best checkpoints.
    session_dir = "/root/capsule/results/training-sessions/session-20260710_1719"
    checkpoint_path = "/root/capsule/data/bm4dnet-training-session-20260714_2305/BM4DNet-20260716-49--162.266248.pth"
    top_k = 1  # number of best checkpoints to evaluate from session_dir

    # Eval dataset spreadsheet (first row is a section label; header is row 1).
    xlsx_path = "/root/capsule/data/denoise_evaluation_datasets/denoise_eval_dataset.xlsx"

    # Per-brain lookup tables (keys are brain_id strings).
    img_prefixes_path = "/root/capsule/data/exaspim_image_prefixes.json"
    background_offsets_path = "/root/capsule/data/exaspim_background_offsets.json"

    # Region shape (z, y, x); each dim must be >= the model patch size (64).
    crop_shape = (1024, 1024, 1024)

    # Use raw_input=True for volumes that were not background-subtracted.
    raw_input = True

    # Output. Block outputs land in subdirectories of output_dir named
    # "<brain_id>_block<i>". Set base_output_zarr to an S3 prefix (or local
    # path) to also persist each denoised block as an OME-Zarr; None to skip.
    output_dir = "/results/evaluation"
    base_output_zarr = "s3://aind-scratch-data/anna.grim/denoising-experiments/test/BM4DNet-20260716-49--162.266248"

    device = "cuda"
    batch_size = 32
    clevel = 5

    evaluate()
