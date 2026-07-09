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
    if crop_center is not None:
        raw = np.asarray(
            img_util.get_patch(img, crop_center, crop_shape, is_center=True)
        )
    else:
        raw = np.asarray(img[0, 0])
    print("Volume shape:", raw.shape)

    # For a raw (non-background-subtracted) volume, estimate this volume's
    # background offset so it lands in the same background-at-zero space the
    # model was trained on (mirrors the per-brain offset subtracted in training).
    # Set raw_input=False if the input is already background-subtracted.
    if raw_input:
        volume_transform = build_volume_transform(transform, raw)
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
        img_util.write_ome_zarr(denoised, output_zarr)
        print("Denoised Zarr written to:", output_zarr)


if __name__ == "__main__":
    # Checkpoint. Point session_dir at a training session (the folder holding
    # the BM4DNet-*.pth files) to auto-select the best checkpoint. Set
    # checkpoint_path to a .pth to evaluate that file explicitly instead.
    session_dir = "/root/capsule/data/bm4dnet-training-session-20260709_1354"
    checkpoint_path = None

    # Test image. Any zarr readable by img_util.read, including an s3:// path;
    # give the full path to a single 5D multiscale level array.
    img_path = "s3://aind-benchmark-data/3d-image-compression/blocks/block_001/input.zarr/0"

    # Region to evaluate. crop_center=(z, y, x) with crop_shape denoises only a
    # bounded sub-volume (reads just that region from S3); each dim of crop_shape
    # must be >= the model patch size (64). Set crop_center=None to denoise the
    # entire volume -- only safe for small, pre-cropped test blocks, since a
    # full-resolution zarr will not fit in memory.
    # crop_center = (256, 256, 256)
    crop_center = None
    crop_shape = (256, 256, 256)

    # raw_input=True estimates a per-volume background offset (use for raw
    # volumes that were NOT background-subtracted).
    raw_input = True

    # Output + misc
    output_dir = "/results/evaluation"
    # Where to persist the denoised volume as an OME-Zarr. Local path or a
    # cloud path (e.g. "s3://BUCKET/PATH/denoised.zarr"). Set to None to skip.
    output_zarr = "s3://aind-scratch-data/cameron.arshadi/denoising-experiments/outputs/BM4DNet-20260709-405--163.534489/block_001.zarr"
    device = "cuda"
    batch_size = 32
    clevel = 5

    evaluate()
