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

_REQUIRED_CACHE_FILES = ("raw.npy", "teacher.npy", "fg.npy", "transform.json")


def _load_cached_transform(train_cache_dir, val_cache_dir):
    """Validates both patch caches and returns their shared transform."""
    cache_dirs = {
        "train_cache_dir": train_cache_dir,
        "val_cache_dir": val_cache_dir,
    }
    transform_cfgs = {}
    for name, cache_dir in cache_dirs.items():
        if not cache_dir:
            raise ValueError(f"{name} is required for training")
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(
                f"{name} does not exist or is not a directory: {cache_dir}"
            )
        missing = [
            filename
            for filename in _REQUIRED_CACHE_FILES
            if not os.path.isfile(os.path.join(cache_dir, filename))
        ]
        if missing:
            raise FileNotFoundError(
                f"{name} is missing required cache files: "
                + ", ".join(missing)
            )
        transform_cfgs[name] = util.read_json(
            os.path.join(cache_dir, "transform.json")
        )

    if transform_cfgs["train_cache_dir"] != transform_cfgs["val_cache_dir"]:
        raise ValueError(
            "train and validation patch caches use different transforms"
        )
    return build_transform(transform_cfgs["train_cache_dir"])


def train(train_cache_dir, val_cache_dir):
    """Trains and validates exclusively from precomputed patch caches."""
    # Per-brain offsets and the BM4D teacher are already baked into the cached
    # counts. Both caches must use the identical count-space transform.
    transform = _load_cached_transform(train_cache_dir, val_cache_dir)
    train_dataset = data_handling.CachedPatchDataset(
        train_cache_dir,
        transform=transform,
        preserve_foreground=preserve_foreground,
    )
    val_dataset = data_handling.CachedValidateDataset(
        val_cache_dir,
        transform=transform,
        preserve_foreground=preserve_foreground,
    )
    print("Transform:", transform.cfg)
    print(
        "Training from cache:", train_cache_dir,
        "| pool size:", len(train_dataset.raw),
    )
    print(
        "Validating from cache:", val_cache_dir,
        "| examples:", len(val_dataset),
    )

    # Cached patches are cheap to load, so read them in-thread.
    trainer = Trainer(
        output_dir,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        model=model,
        fg_weight=fg_weight,
        checkpoint_weights=checkpoint_weights,
        num_workers=0,
        val_every=val_every,
        seed=seed,
    )

    # Persist the run configuration next to the checkpoints/tensorboard so each
    # session is reproducible (the Trainer merges in its own hyperparameters).
    trainer.save_config(
        {
            "train_cache_dir": train_cache_dir,
            "val_cache_dir": val_cache_dir,
            "resume_path": resume_path,
            "transform_cfg": transform.cfg,
            "preserve_foreground": preserve_foreground,
        }
    )

    if resume_path is not None:
        trainer.load_pretrained_weights(resume_path)
    trainer.run(train_dataset, val_dataset)


if __name__ == "__main__":
    # Paths
    output_dir = "/results/training-sessions"
    # Both patch caches are required. Build them with precompute.py --split
    # train and precompute.py --split val before starting training.
    train_cache_dir = (
        "/root/capsule/data/denoise_net_patch_cache_2026_07_13/patch_cache"
    )
    val_cache_dir = (
        "/root/capsule/data/denoise_net_patch_cache_2026_07_13/val_patch_cache"
    )
    util.mkdir(output_dir)

    # Resume path. Checkpoints from before the normalization overhaul are NOT
    # compatible: GroupNorm changed the state_dict keys, the residual output
    # changed the semantics, and the old model was trained under percentile
    # normalization. Train from scratch (None), or point this at a NEW-format
    # checkpoint (a dict of {"model", "transform"}) to resume.
    resume_path = None

    # Model (new defaults: residual output + GroupNorm)
    model = UNet()

    # Training parameters
    batch_size = 32
    lr = 1e-3
    max_epochs = 30
    # Validate (and consider a checkpoint) after every full-cache epoch.
    val_every = 1
    seed = 42

    # Signal-preserving loss + target/sampling (Parts E/F). preserve_foreground
    # keeps raw counts on the foreground so BM4D cannot erase neurites; that
    # makes the foreground target equal to the (noisy) input, so a large
    # fg_weight rewards the identity map -- the net stops denoising and its
    # output compresses no better than raw. Keep fg_weight modest (~1-3) so
    # background denoising, not foreground copying, dominates the loss.
    fg_weight = 0
    preserve_foreground = False

    # Checkpoint selection (Part C). None => fidelity-only (cratio weight 0),
    # which cannot see compression and happily selects a non-denoising model
    # (identity minimizes fg_mae because the fg target is the raw input). Give
    # cratio a nonzero weight so selection rewards the compression the project
    # exists for. cratio is the operating-point knob: raise it to trade
    # fidelity for compression, lower it to protect faint neurites.
    checkpoint_weights = dict(
        fg_mae=1.0, bg_mae=0.2, top_pct_error=0.5, cratio=10.0
    )

    # Main
    train(train_cache_dir, val_cache_dir)
