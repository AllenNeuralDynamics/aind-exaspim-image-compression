import hashlib
import json
import os

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.machine_learning.losses import build_loss
from aind_exaspim_image_compression.machine_learning.train import Trainer
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
)
from aind_exaspim_image_compression.machine_learning.unet3d import UNet
from aind_exaspim_image_compression.utils import util

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

_REQUIRED_CACHE_FILES = ("raw.npy", "teacher.npy", "fg.npy", "transform.json")


def _cache_provenance(cache_dir):
    """Return a stable config hash and complete cache-generation record."""
    config_path = os.path.join(cache_dir, "config.json")
    config = util.read_json(config_path) if os.path.isfile(config_path) else None
    transform = util.read_json(os.path.join(cache_dir, "transform.json"))
    hash_payload = {"config": config, "transform": transform}
    digest = hashlib.sha256(
        json.dumps(
            hash_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "path": os.path.abspath(cache_dir),
        "config_sha256": digest,
        "config": config,
    }


def _validate_cache_provenance(train_record, val_record):
    """Reject train/validation caches with incompatible generation settings."""
    train_cfg = train_record["config"]
    val_cfg = val_record["config"]
    if train_cfg is None or val_cfg is None:
        return
    shared_keys = (
        "transform_cfg",
        "teacher_mode",
        "sigma_bm4d",
        "gat_sigma_multiplier",
        "noise_models",
        "saturation_margins",
        "count_dtype",
        "heldout_regions",
    )
    mismatched = [
        key
        for key in shared_keys
        if key in train_cfg and key in val_cfg and train_cfg[key] != val_cfg[key]
    ]
    if mismatched:
        raise ValueError(
            "train and validation caches have incompatible provenance: "
            + ", ".join(mismatched)
        )


def _experiment_provenance(train_cache_dir, val_cache_dir):
    """Assemble cache, teacher, noise, sampling, and held-out provenance."""
    train_record = _cache_provenance(train_cache_dir)
    val_record = _cache_provenance(val_cache_dir)
    _validate_cache_provenance(train_record, val_record)
    train_source = train_record["config"] or {}
    val_source = val_record["config"] or {}

    def sampling_record(source):
        return {
            key: source.get(key)
            for key in (
                "brain_sampling_weights",
                "brain_sampling_distribution",
                "sampling_rois_path",
                "sampling_rois",
                "bright_sampling_weights",
                "exclude_heldout",
            )
        }

    return {
        "caches": {"train": train_record, "validation": val_record},
        "teacher": {
            key: train_source.get(key)
            for key in (
                "teacher_mode",
                "sigma_bm4d",
                "gat_sigma_multiplier",
            )
        },
        "noise_models": {
            "source_path": train_source.get("noise_models_path"),
            "models": train_source.get("noise_models"),
            "saturation_margins": train_source.get("saturation_margins"),
        },
        "sampling": {
            "train": sampling_record(train_source),
            "validation": sampling_record(val_source),
        },
        "heldout_regions": {
            "source_path": train_source.get("heldout_regions_path"),
            "regions": train_source.get("heldout_regions"),
        },
    }


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
    provenance = _experiment_provenance(train_cache_dir, val_cache_dir)
    configured_loss = globals().get(
        "loss_cfg",
        {
            "legacy_weight": 1.0,
            "count_weight": 0.0,
            "legacy": {"fg_weight": fg_weight},
        },
    )
    criterion = build_loss(configured_loss, transform)
    require_noise_metadata = criterion.requires_count_metadata
    train_kwargs = {}
    val_kwargs = {}
    if require_noise_metadata:
        train_kwargs["require_noise_metadata"] = True
        val_kwargs["require_noise_metadata"] = True
    train_dataset = data_handling.CachedPatchDataset(
        train_cache_dir,
        transform=transform,
        preserve_foreground=preserve_foreground,
        n_examples_per_epoch=n_train_examples_per_epoch,
        **train_kwargs,
    )
    val_dataset = data_handling.CachedValidateDataset(
        val_cache_dir,
        transform=transform,
        preserve_foreground=preserve_foreground,
        **val_kwargs,
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
        criterion=criterion,
        checkpoint_weights=checkpoint_weights,
        num_workers=0,
        val_every=val_every,
    )

    # Persist the run configuration next to the checkpoints/tensorboard so each
    # session is reproducible (the Trainer merges in its own hyperparameters).
    trainer.save_config(
        {
            "train_cache_dir": train_cache_dir,
            "val_cache_dir": val_cache_dir,
            "resume_path": resume_path,
            "transform_cfg": transform.cfg,
            "n_train_examples_per_epoch": n_train_examples_per_epoch,
            "preserve_foreground": preserve_foreground,
            "loss_config": criterion.cfg,
            "provenance": provenance,
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
        "/root/capsule/data/denoise_net_patch_cache_2026_07_10/patch_cache"
    )
    val_cache_dir = (
        "/root/capsule/data/denoise_net_patch_cache_2026_07_10/"
        "val_patch_cache"
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
    max_epochs = 500
    n_train_examples_per_epoch = 300
    # Validate (and consider a checkpoint) every this many epochs. A larger
    # cached validation set is cheap to store but CPU-bound to score, so keep
    # this above 1 to avoid the metrics dominating epoch time.
    val_every = 3

    # Signal-preserving loss + target/sampling (Parts E/F). preserve_foreground
    # keeps raw counts on the foreground so BM4D cannot erase neurites; that
    # makes the foreground target equal to the (noisy) input, so a large
    # fg_weight rewards the identity map -- the net stops denoising and its
    # output compresses no better than raw. Keep fg_weight modest (~1-3) so
    # background denoising, not foreground copying, dominates the loss.
    fg_weight = 0
    preserve_foreground = True

    # Loss A/B mode. Keep cache paths, sampling, seeds, and held-out regions
    # fixed when comparing these weights. count_weight > 0 requires a Part 3
    # cache with finite per-patch noise metadata.
    loss_cfg = {
        "legacy_weight": 1.0,
        "count_weight": 0.0,
        "legacy": {"fg_weight": fg_weight, "eps": 1e-3},
        "count": {
            "sigma_floor": 2.0,
            "saturation_margin": 64,
            "saturation_dilate": 1,
            "standardized_error_cap": None,
            "eps": 1e-3,
        },
    }

    # Checkpoint selection (Part C). None => fidelity-only (cratio weight 0),
    # which cannot see compression and happily selects a non-denoising model
    # (identity minimizes fg_mae because the fg target is the raw input). Give
    # cratio a nonzero weight so selection rewards the compression the project
    # exists for. cratio is the operating-point knob: raise it to trade
    # fidelity for compression, lower it to protect faint neurites.
    checkpoint_weights = dict(
        fg_mae=1.0, bg_mae=0.2, top_pct_error=0.5, cratio=2.0
    )

    # Main
    train(train_cache_dir, val_cache_dir)
