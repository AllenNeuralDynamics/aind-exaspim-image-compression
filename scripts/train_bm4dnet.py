import multiprocessing as mp
import os

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.machine_learning.train import Trainer
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
)
from aind_exaspim_image_compression.machine_learning.unet3d import UNet
from aind_exaspim_image_compression.utils import util

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""


def train():
    # Fully-cached path: with both a training and a validation cache, no cloud
    # reads or BM4D happen at startup. Rebuild the exact transform the caches
    # were built with (stamped in the val cache) so the cached patches and the
    # model share the identical mapping; per-brain offsets are already baked
    # into the cached counts.
    if cache_dir and val_cache_dir:
        transform_path = os.path.join(val_cache_dir, "transform.json")
        cached_cfg = (
            util.read_json(transform_path)
            if os.path.exists(transform_path)
            else transform_cfg
        )
        transform = build_transform(cached_cfg)
        train_dataset = data_handling.CachedPatchDataset(
            cache_dir,
            transform=transform,
            preserve_foreground=preserve_foreground,
            n_examples_per_epoch=n_train_examples_per_epoch,
        )
        val_dataset = data_handling.CachedValidateDataset(
            val_cache_dir,
            transform=transform,
            preserve_foreground=preserve_foreground,
        )
        print("Transform:", transform.cfg)
        print(
            "Training from cache:", cache_dir,
            "| pool size:", len(train_dataset.raw),
        )
        print(
            "Validating from cache:", val_cache_dir,
            "| examples:", len(val_dataset),
        )
    else:
        # Load Brain IDs and per-brain background offsets
        brain_ids = util.read_txt(brain_ids_path)
        offsets = util.read_json(offsets_path) if offsets_path else None

        # Datasets. The per-brain offset is subtracted from each patch, then
        # one shared transform (offset 0) maps every brain to a
        # background-at-zero space; the transform cfg is serialized with each
        # checkpoint.
        train_dataset, val_dataset = data_handling.init_datasets(
            brain_ids,
            img_prefixes_path,
            patch_shape,
            foreground_sampling_rate=foreground_sampling_rate,
            min_foreground_voxels=min_foreground_voxels,
            min_segmentation_volume=min_segmentation_volume,
            n_train_examples_per_epoch=n_train_examples_per_epoch,
            n_validate_examples=n_validate_examples,
            offsets=offsets,
            preserve_foreground=preserve_foreground,
            segmentation_prefixes_path=segmentation_prefixes_path,
            sigma_bm4d=sigma_bm4d,
            swc_pointers=swc_pointers,
            transform_cfg=transform_cfg,
        )
        print("Transform:", train_dataset.transform.cfg)
        print("# Brains with Skeletons:", len(train_dataset.skeletons))
        print("# Brains with Segmentations:", len(train_dataset.segmentations))

        # Train from the precomputed patch cache when available (GPU-bound).
        # Reuse the transform init_datasets built so the cache and validation
        # share the identical mapping; validation stays on the cloud dataset.
        if cache_dir:
            train_dataset = data_handling.CachedPatchDataset(
                cache_dir,
                transform=train_dataset.transform,
                preserve_foreground=preserve_foreground,
                n_examples_per_epoch=n_train_examples_per_epoch,
            )
            print(
                "Training from cache:", cache_dir,
                "| pool size:", len(train_dataset.raw),
            )

    # Run. Cached patches are cheap, so load them in-thread (num_workers=0);
    # the cloud dataset needs the process pool for parallel BM4D.
    trainer = Trainer(
        output_dir,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        model=model,
        fg_weight=fg_weight,
        checkpoint_weights=checkpoint_weights,
        num_workers=0 if cache_dir else None,
    )
    if resume_path is not None:
        trainer.load_pretrained_weights(resume_path)
    trainer.run(train_dataset, val_dataset)


if __name__ == "__main__":
    # Paths
    brain_ids_path = "/data/train_brain_ids.txt"
    img_prefixes_path = "/data/exaspim_image_prefixes.json"
    output_dir = "/results/training-sessions"
    segmentation_prefixes_path = (
        "/data/exaspim_segmentation_prefixes.json"
    )
    # Per-brain background offsets from estimate_background_offsets.py. Set to
    # None to disable per-brain offset subtraction.
    offsets_path = "/data/exaspim_background_offsets.json"
    # Precomputed patch cache from precompute_patches.py. Leave None to sample
    # + BM4D live from the cloud (slow, GPU-starved); after precomputing, set
    # this to the cache dir (e.g. "/results/patch_cache") to train GPU-bound.
    cache_dir = "/root/capsule/data/denoise_net_patch_cache_10K_2026_07_09"
    # Precomputed validation cache from precompute_val_patches.py. When set
    # alongside cache_dir, training runs fully offline (no cloud reads or BM4D
    # at startup) and the GPU is busy almost immediately. Leave None to build
    # the validation set live from the cloud.
    val_cache_dir = None
    util.mkdir(output_dir)

    # Resume path. Checkpoints from before the normalization overhaul are NOT
    # compatible: GroupNorm changed the state_dict keys, the residual output
    # changed the semantics, and the old model was trained under percentile
    # normalization. Train from scratch (None), or point this at a NEW-format
    # checkpoint (a dict of {"model", "transform"}) to resume.
    resume_path = None

    # SWC Pointer
    swc_pointers = {
        "bucket_name": "allen-nd-goog",
        "path": "ground_truth_tracings",
    }

    # Intensity transform, shared by train and inference. Options:
    #   {"kind": "asinh",    "params": {"offset": 35.0, "scale": 32.0}}
    #   {"kind": "anscombe", "params": {"gain": 8.0, "read_noise": 5.0,
    #                                   "offset": 35.0}}
    #   {"kind": "linear",   "params": {"mn": 0.0, "mx": 1000.0, "clip": 8.0}}
    # Per-brain offsets (offsets_path) are subtracted at the dataset, so the
    # transform offset stays 0 to avoid double-subtracting. scale is the
    # linear->log knee (tune from the noise floor).
    transform_cfg = {
        "kind": "asinh",
        "params": {"offset": 0.0, "scale": 32.0},
    }

    # Model (new defaults: residual output + GroupNorm)
    model = UNet()

    # Training parameters
    batch_size = 32
    foreground_sampling_rate = 0.5
    lr = 1e-4
    max_epochs = 400
    n_train_examples_per_epoch = 300
    n_validate_examples = 60
    patch_shape = (64, 64, 64)
    sigma_bm4d = 24

    # Signal-preserving loss + target/sampling (Parts E/F). fg_weight is
    # aggressive; sweep it against foreground fraction. preserve_foreground
    # keeps raw counts on the foreground so BM4D cannot erase neurites.
    fg_weight = 20.0
    preserve_foreground = True
    min_foreground_voxels = 50
    min_segmentation_volume = 200

    # Checkpoint selection (Part C). None => fidelity-only (cratio weight 0).
    # Once you pick the compression-vs-fidelity operating point, set e.g.
    #   checkpoint_weights = dict(
    #       fg_mae=1.0, bg_mae=0.2, top_pct_error=0.5, cratio=200.0
    #   )
    checkpoint_weights = None

    # Main
    mp.set_start_method("spawn", force=True)
    train()
