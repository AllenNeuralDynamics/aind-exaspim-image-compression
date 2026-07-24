import argparse
import os

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.machine_learning.train import Trainer
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
)
from aind_exaspim_image_compression.machine_learning.unet3d import N2V2UNet, UNet
from aind_exaspim_image_compression.utils import util

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

_REQUIRED_CACHE_FILES = ("raw.npy", "teacher.npy", "fg.npy", "transform.json")

CACHE_DATASETS = {
    "alpha": "/data/denoise_net_alpha_patch_cache_2026_07_23",
    "beta": "/data/denoise_net_beta_patch_cache_2026_07_23",
    "beta2": "/data/denoise_net_beta2_patch_cache_2026_07_13",
}

MODELS = {
    "unet": UNet(),
    "n2v2_unet": N2V2UNet(),
}

def train():
    """Trains and validates exclusively from precomputed patch caches."""
    # Load transform
    transform = build_transform(config["transform_cfg"])
    print("Transform:", transform.cfg)

    # Datasets
    train_cache_dirs, val_cache_dirs = get_cache_dirs()
    train_dataset = data_handling.CachedPatchDataset(
        train_cache_dirs,
        transform=transform,
        preserve_foreground=preserve_foreground,
    )
    val_dataset = data_handling.CachedValidateDataset(
        val_cache_dirs,
        transform=transform,
        preserve_foreground=preserve_foreground,
    )

    print(
        f"Training from cache: {train_cache_dirs}"
        f"| examples: {len(train_dataset)}"
    )
    print(
        f"Validating from cache: {val_cache_dirs}"
        f"| examples: {len(val_dataset)}"
    )

    # Model
    model = MODELS[args.model_name]

    # Trainer
    trainer = Trainer(
        output_dir,
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        model=model,
        use_amp=use_amp_train,
        use_amp_validation=use_amp_validation,
        fg_weight=fg_weight,
        checkpoint_weights=checkpoint_weights,
        num_workers=0,
        val_every=val_every,
        seed=seed,
    )

    trainer.save_config(
        {
            "train_cache_dir": train_cache_dirs,
            "val_cache_dir": val_cache_dirs,
            "resume_path": resume_path,
            "transform_cfg": transform.cfg,
            "preserve_foreground": preserve_foreground,
            "use_amp": use_amp_train,
            "use_amp_validation": use_amp_validation,
        }
    )

    if resume_path is not None:
        trainer.load_pretrained_weights(resume_path)

    # Start training
    trainer.run(train_dataset, val_dataset)


def get_cache_dirs():
    # Set cache dataset names to retrieve
    dataset_names = (
        CACHE_DATASETS.keys()
        if args.dataset_name == "all"
        else [args.dataset_name]
    )

    # Get paths to cache datasets
    train_cache_dirs = list()
    val_cache_dirs = list()
    for name in dataset_names:
        train_cache_dirs.append(f"{CACHE_DATASETS[name]}/patch_cache")
        val_cache_dirs.append(f"{CACHE_DATASETS[name]}/val_patch_cache")
    return train_cache_dirs, val_cache_dirs


if __name__ == "__main__":
    # Extract reproducible run parameters
    parser = argparse.ArgumentParser(description="Process named arguments")
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=False)
    args = parser.parse_args()

    # Paths
    output_dir = "/results/training-sessions"
    util.mkdir(output_dir)

    # Load experiment config
    config_name = args.config_name.strip()
    input_config_path = f"/data/{config_name}.json"
    output_config_path = os.path.join(output_dir, f"{config_name}.json")

    config = util.read_json(input_config_path)
    util.write_json(output_config_path, config)
    print("Experiment Config:", config)

    # Resume path. Checkpoints from before the normalization overhaul are NOT
    # compatible: GroupNorm changed the state_dict keys, the residual output
    # changed the semantics, and the old model was trained under percentile
    # normalization. Train from scratch (None), or point this at a NEW-format
    # checkpoint (a dict of {"model", "transform"}) to resume.
    resume_path = None

    # Training parameters
    batch_size = config["batch_size"]
    lr = config["lr"]
    max_epochs = config["max_epochs"]

    use_amp_train = config["use_amp_train"]
    use_amp_validation = config["use_amp_val"]
    val_every = config["val_every"]
    seed = config["seed"]

    # Signal-preserving loss + target/sampling
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
    train()
