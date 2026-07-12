import hashlib
import json
import os
import random

import numpy as np
import torch

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


def _resolve_cached_transform(
    train_cache_dir, val_cache_dir, transform_override=None
):
    """Validate cache transforms before applying an explicit safe override."""
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
    cache_cfg = transform_cfgs["train_cache_dir"]
    cache_transform = build_transform(cache_cfg)
    if transform_override is None:
        resolved_cfg = cache_cfg
    else:
        if not isinstance(transform_override, dict):
            raise ValueError("transform override must be a configuration object")
        if transform_override.get("calibrate"):
            raise ValueError(
                "transform override cannot calibrate from cached patches"
            )
        if transform_override.get("kind") == "offset":
            raise ValueError(
                "transform override cannot restore a raw-count offset"
            )
        params = transform_override.get("params", {})
        if float(params.get("offset", 0.0)) != 0.0:
            raise ValueError(
                "cached counts are offset-subtracted; transform override "
                "offset must be zero"
            )
        override_transform = build_transform(transform_override)
        if override_transform.max_count != cache_transform.max_count:
            raise ValueError(
                "transform override cannot change the physical max_count"
            )
        resolved_cfg = transform_override
    return build_transform(resolved_cfg), {
        "cache_transform_cfg": cache_cfg,
        "transform_override_cfg": transform_override,
        "resolved_transform_cfg": resolved_cfg,
    }


def _load_cached_transform(
    train_cache_dir, val_cache_dir, transform_override=None
):
    """Compatibility wrapper returning only the resolved cache transform."""
    transform, _ = _resolve_cached_transform(
        train_cache_dir, val_cache_dir, transform_override
    )
    return transform


def _validate_experiment_config(config):
    """Validate the serializable top-level training experiment contract."""
    required = {
        "paths",
        "teacher",
        "transform",
        "loss",
        "model",
        "noise_model_path",
        "sampling",
        "seed",
        "training",
        "target",
        "checkpoint_weights",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(
            "experiment configuration is missing fields: "
            + ", ".join(sorted(missing))
        )
    unknown = set(config) - required
    if unknown:
        raise ValueError(
            "experiment configuration has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    paths = config["paths"]
    for key in ("output_dir", "train_cache_dir", "val_cache_dir"):
        if not paths.get(key):
            raise ValueError(f"experiment paths.{key} is required")
    if int(config["model"].get("in_channels", 1)) != 1:
        raise ValueError(
            "this training path currently supports model.in_channels=1; "
            "the optional noise-map channel is a separate experiment"
        )
    try:
        json.dumps(config, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "experiment configuration must be JSON serializable"
        ) from error
    return config


def _validate_expected_cache_settings(config, provenance):
    """Check explicit teacher/noise/sampling expectations against each cache."""
    teacher_mode = config["teacher"].get("mode")
    cached_teacher_mode = provenance["teacher"].get("teacher_mode")
    if teacher_mode is not None and teacher_mode != cached_teacher_mode:
        raise ValueError(
            f"configured teacher mode {teacher_mode!r} does not match cache "
            f"mode {cached_teacher_mode!r}"
        )
    configured_noise_path = config.get("noise_model_path")
    cached_noise_path = provenance["noise_models"].get("source_path")
    if (
        configured_noise_path is not None
        and configured_noise_path != cached_noise_path
    ):
        raise ValueError(
            "configured noise_model_path does not match cache provenance"
        )

    expected_sampling = config["sampling"]
    cache_records = provenance["caches"]
    comparisons = {
        "brain_sampling_weights": "brain_sampling_weights",
        "sampling_rois": "sampling_rois",
        "train_regions": "sampling_rois",
        "validation_regions": "sampling_rois",
    }
    for configured_key, cache_key in comparisons.items():
        expected = expected_sampling.get(configured_key)
        if expected is None:
            continue
        split = "validation" if configured_key == "validation_regions" else "train"
        cache_config = cache_records[split]["config"] or {}
        if expected != cache_config.get(cache_key):
            raise ValueError(
                f"configured sampling.{configured_key} does not match "
                f"{split} cache provenance"
            )


def _seed_experiment(seed):
    """Seed cached-patch selection and model initialization reproducibly."""
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(experiment_config):
    """Train exclusively from caches under one serializable experiment."""
    config = _validate_experiment_config(experiment_config)
    _seed_experiment(config["seed"])
    paths = config["paths"]
    train_cache_dir = paths["train_cache_dir"]
    val_cache_dir = paths["val_cache_dir"]
    output_dir = paths["output_dir"]
    resume_path = paths.get("resume_path")
    util.mkdir(output_dir)

    # Per-brain offsets and the BM4D teacher are already baked into the cached
    # counts. Both caches must use the identical count-space transform.
    transform, transform_record = _resolve_cached_transform(
        train_cache_dir,
        val_cache_dir,
        transform_override=config["transform"].get("override"),
    )
    provenance = _experiment_provenance(train_cache_dir, val_cache_dir)
    _validate_expected_cache_settings(config, provenance)
    criterion = build_loss(config["loss"], transform)
    require_noise_metadata = criterion.requires_count_metadata
    preserve_foreground = bool(config["target"]["preserve_foreground"])
    training = config["training"]
    train_kwargs = {}
    val_kwargs = {}
    if require_noise_metadata:
        train_kwargs["require_noise_metadata"] = True
        val_kwargs["require_noise_metadata"] = True
    train_dataset = data_handling.CachedPatchDataset(
        train_cache_dir,
        transform=transform,
        preserve_foreground=preserve_foreground,
        n_examples_per_epoch=training["n_train_examples_per_epoch"],
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

    model_cfg = dict(config["model"])
    model_cfg.pop("in_channels", None)
    model = UNet(**model_cfg)
    # Cached patches are cheap to load, so read them in-thread.
    trainer = Trainer(
        output_dir,
        batch_size=training["batch_size"],
        device=training.get("device", "cuda"),
        lr=training["lr"],
        max_epochs=training["max_epochs"],
        model=model,
        criterion=criterion,
        use_amp=training.get("use_amp", True),
        checkpoint_weights=config["checkpoint_weights"],
        num_workers=0,
        val_every=training["val_every"],
    )

    # Persist the run configuration next to the checkpoints/tensorboard so each
    # session is reproducible (the Trainer merges in its own hyperparameters).
    trainer.save_config(
        {
            "train_cache_dir": train_cache_dir,
            "val_cache_dir": val_cache_dir,
            "resume_path": resume_path,
            "transform_cfg": transform.cfg,
            "transform_record": transform_record,
            "n_train_examples_per_epoch": training[
                "n_train_examples_per_epoch"
            ],
            "preserve_foreground": preserve_foreground,
            "loss_config": criterion.cfg,
            "provenance": provenance,
            "experiment": config,
        }
    )

    if resume_path is not None:
        trainer.load_pretrained_weights(resume_path)
    trainer.run(train_dataset, val_dataset)


EXPERIMENT_CONFIG = {
    "paths": {
        "output_dir": "/results/training-sessions",
        "train_cache_dir": (
            "/root/capsule/data/denoise_net_patch_cache_2026_07_10/patch_cache"
        ),
        "val_cache_dir": (
            "/root/capsule/data/denoise_net_patch_cache_2026_07_10/"
            "val_patch_cache"
        ),
        "resume_path": None,
    },
    # None accepts the teacher stamped into the cache. Set an explicit mode to
    # make the script reject a cache built with a different teacher.
    "teacher": {"mode": None},
    # Cache transform metadata is compared before this explicit count-space
    # override is applied. Set override to None to use the cache transform.
    "transform": {
        "override": {
            "kind": "asinh",
            "params": {"offset": 0.0, "scale": 60.0},
        }
    },
    "loss": {
        "legacy_weight": 1.0,
        "count_weight": 0.0,
        "legacy": {"fg_weight": 0.0, "eps": 1e-3},
        "count": {
            "sigma_floor": 2.0,
            "saturation_margin": 64,
            "saturation_dilate": 1,
            "standardized_error_cap": None,
            "eps": 1e-3,
        },
    },
    "model": {
        "in_channels": 1,
        "width_multiplier": 1,
        "trilinear": True,
        "residual": True,
    },
    "noise_model_path": None,
    "sampling": {
        "brain_sampling_weights": None,
        "sampling_rois": None,
        "train_regions": None,
        "validation_regions": None,
    },
    "seed": 42,
    "training": {
        "batch_size": 32,
        "device": "cuda",
        "use_amp": True,
        "lr": 1e-3,
        "max_epochs": 500,
        "n_train_examples_per_epoch": 300,
        "val_every": 3,
    },
    "target": {"preserve_foreground": False},
    "checkpoint_weights": {
        "fg_mae": 1.0,
        "bg_mae": 0.2,
        "top_pct_error": 0.5,
        "cratio": 2.0,
    },
}


if __name__ == "__main__":
    train(EXPERIMENT_CONFIG)
