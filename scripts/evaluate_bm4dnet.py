import glob
import os
import re

import numpy as np
from numcodecs import blosc

from aind_exaspim_image_compression.inference import (
    build_volume_transform,
    load_model,
    predict,
)
from aind_exaspim_image_compression.utils import img_util, util


def find_best_checkpoint(session_dir):
    """
    Returns the best checkpoint written by a training session.

    Checkpoints are named ``BM4DNet-<date>-<epoch>-<score>.pth`` by
    Trainer.save_model, where <score> is the checkpoint-selection score at save
    time and lower is better. The score can be negative (the cratio term is
    subtracted), so it is parsed with a regex that allows a leading minus rather
    than by splitting on "-".

    Parameters
    ----------
    session_dir : str
        Training session directory holding the BM4DNet-*.pth checkpoints.

    Returns
    -------
    str
        Path to the lowest-scoring (best) checkpoint.
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

    return min(paths, key=score)


def evaluate():
    # Resolve the checkpoint to evaluate (explicit path wins over auto-select).
    ckpt_path = checkpoint_path or find_best_checkpoint(session_dir)
    print("Checkpoint:", ckpt_path)

    # Load the model together with the intensity transform it was trained with
    # (load_model rebuilds the transform from the checkpoint's "transform" cfg).
    model, transform = load_model(ckpt_path, device=device)
    print("Transform:", transform.cfg)

    # Read the image. img_util.read handles s3://, gs://, and local zarr; point
    # img_path at a single 5D (t, c, z, y, x) multiscale level array (e.g.
    # ".../image.zarr/0"). Slicing the lazy zarr in get_patch fetches only the
    # requested region, so a crop avoids pulling the whole (huge) volume from S3.
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
        snapped_center_zyx = source_offset_zyx + np.asarray(
            crop_center_voxel
        )
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

    # Optionally persist the denoised volume as a Zarr array. output_zarr may be
    # a local path or a cloud URL (s3://.../denoised.zarr); it is written with
    # the same zstd/clevel codec used to measure cratio, and reads back via
    # img_util.read at "<output_zarr>" (a plain array, no "/0" suffix). Writing
    # to S3 needs credentials (the default AWS chain), unlike the anonymous
    # public read of the input.
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


if __name__ == "__main__":
    # Checkpoint. Point session_dir at a training session (the folder holding
    # the BM4DNet-*.pth files) to auto-select the best checkpoint. Set
    # checkpoint_path to a .pth to evaluate that file explicitly instead.
    session_dir = "/root/capsule/results/training-sessions/session-20260710_1719"
    checkpoint_path = "/root/capsule/results/training-sessions/session-20260710_1719/BM4DNet-20260710-499--19.965923.pth"

    # Test image. Any zarr readable by img_util.read, including an s3:// path;
    # give the full path to a single 5D multiscale level array.
    img_path = "s3://aind-open-data/exaSPIM_826511_2026-06-02_15-10-47/SPIM.ome.zarr/tile_000010_ch_488.zarr/0"

    # Region to evaluate. crop_center is the numeric (x, y, z) position shown by
    # Neuroglancer; the physical scale displayed beside each coordinate is read
    # from the source OME-Zarr. The position is converted to the nearest source
    # voxel before cropping. Each crop_shape dimension must be >= the model patch
    # size (64). Set crop_center=None only for a small, pre-cropped input volume.
    crop_center = (22464, -15914, 18711)
    crop_shape = (1024, 1024, 1024)

    # Use raw_input=True for volumes that were not background-subtracted.
    raw_input = True
    # Prefer the background offset precomputed from the full image tile's
    # lower-resolution data. None estimates from this test subvolume instead.
    background_offset = 37

    # Output + misc
    output_dir = "/results/evaluation"
    # Where to persist the denoised volume as an OME-Zarr. Local path or a
    # cloud path (e.g. "s3://BUCKET/PATH/denoised.zarr"). Set to None to skip.
    output_zarr = "s3://aind-scratch-data/cameron.arshadi/denoising-experiments/outputs/BM4DNet-20260710-499--19.965923/826511_raw_crop.zarr"
    device = "cuda"
    batch_size = 32
    clevel = 5

    evaluate()
