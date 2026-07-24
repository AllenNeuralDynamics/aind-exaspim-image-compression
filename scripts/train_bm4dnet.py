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


def _normalize_cache_dirs(cache_dir, name):
    """Returns one or more cache paths as a nonempty list of strings."""
    if cache_dir is None:
        raise ValueError(f"{name} is required for training")
    if isinstance(cache_dir, (str, os.PathLike)):
        cache_dirs = [cache_dir]
    else:
        try:
            cache_dirs = list(cache_dir)
        except TypeError as error:
            raise TypeError(
                f"{name} must be a path or an iterable of paths"
            ) from error

    if not cache_dirs:
        raise ValueError(f"{name} is required for training")

    normalized = []
    for index, path in enumerate(cache_dirs):
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"{name}[{index}] is not a path: {path!r}")
        normalized.append(os.fspath(path))
    return normalized


def _load_cached_transform(train_cache_dir, val_cache_dir):
    """Validates all patch caches and returns their shared transform."""
    cache_groups = {
        "train_cache_dir": _normalize_cache_dirs(
            train_cache_dir, "train_cache_dir"
        ),
        "val_cache_dir": _normalize_cache_dirs(val_cache_dir, "val_cache_dir"),
    }
    transform_cfg = None
    for name, cache_dirs in cache_groups.items():
        for index, cache_dir in enumerate(cache_dirs):
            label = name if len(cache_dirs) == 1 else f"{name}[{index}]"
            if not os.path.isdir(cache_dir):
                raise FileNotFoundError(
                    f"{label} does not exist or is not a directory: "
                    f"{cache_dir}"
                )
            missing = [
                filename
                for filename in _REQUIRED_CACHE_FILES
                if not os.path.isfile(os.path.join(cache_dir, filename))
            ]
            if missing:
                raise FileNotFoundError(
                    f"{label} is missing required cache files: "
                    + ", ".join(missing)
                )
            current_cfg = util.read_json(
                os.path.join(cache_dir, "transform.json")
            )
            if transform_cfg is None:
                transform_cfg = current_cfg
            elif current_cfg != transform_cfg:
                raise ValueError(
                    "train and validation patch caches use different "
                    f"transforms: {label}"
                )
    return build_transform(transform_cfg)


def train(train_cache_dir, val_cache_dir):
    """Trains and validates exclusively from precomputed patch caches."""
    # Per-brain offsets and the BM4D teacher are already baked into the cached
    # counts. Both caches must use the identical count-space transform.
    train_cache_dirs = _normalize_cache_dirs(
        train_cache_dir, "train_cache_dir"
    )
    val_cache_dirs = _normalize_cache_dirs(val_cache_dir, "val_cache_dir")
    transform = _load_cached_transform(train_cache_dirs, val_cache_dirs)
    train_cache_arg = (
        train_cache_dirs[0] if len(train_cache_dirs) == 1 else train_cache_dirs
    )
    val_cache_arg = (
        val_cache_dirs[0] if len(val_cache_dirs) == 1 else val_cache_dirs
    )
    train_dataset = data_handling.CachedPatchDataset(
        train_cache_arg,
        transform=transform,
        preserve_foreground=preserve_foreground,
    )
    val_dataset = data_handling.CachedValidateDataset(
        val_cache_arg,
        transform=transform,
        preserve_foreground=preserve_foreground,
    )
    print("Transform:", transform.cfg)
    print(
        "Training from cache:",
        train_cache_arg,
        "| pool size:",
        len(train_dataset),
    )
    print(
        "Validating from cache:",
        val_cache_arg,
        "| examples:",
        len(val_dataset),
    )

    # Cached patches are cheap to load, so read them in-thread.
    trainer = Trainer(
        output_dir,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        model=model,
        use_amp=use_amp,
        use_amp_validation=use_amp_validation,
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
            "train_cache_dir": train_cache_arg,
            "val_cache_dir": val_cache_arg,
            "resume_path": resume_path,
            "transform_cfg": transform.cfg,
            "preserve_foreground": preserve_foreground,
            "use_amp": use_amp,
            "use_amp_validation": use_amp_validation,
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
    max_epochs = 50
    # Use AMP for optimization, but keep validation/checkpoint selection in
    # FP32 so its cratio matches production FP32 inference.
    use_amp = False
    use_amp_validation = False
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
