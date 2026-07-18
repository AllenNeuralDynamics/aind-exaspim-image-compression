import argparse
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


def train(train_cache_dir, val_cache_dir):
    """Trains and validates exclusively from precomputed patch caches."""
    # Per-brain offsets and the BM4D teacher are already baked into the cached
    # counts. Both caches must use the identical count-space transform.
    transform = build_transform(config["transform_cfg"])
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
        use_amp=use_amp_train,
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
            "train_cache_dir": train_cache_dir,
            "val_cache_dir": val_cache_dir,
            "resume_path": resume_path,
            "transform_cfg": transform.cfg,
            "preserve_foreground": preserve_foreground,
            "use_amp": use_amp_train,
            "use_amp_validation": use_amp_validation,
        }
    )

    if resume_path is not None:
        trainer.load_pretrained_weights(resume_path)
    trainer.run(train_dataset, val_dataset)


if __name__ == "__main__":
    # Extract reproducible run parameters
    parser = argparse.ArgumentParser(description="Process named arguments")
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--cached_dataset_type", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--use_amp_train", type=bool, required=False)
    args, _ = parser.parse_known_args()  # ignore cross-mode flags (e.g. --dataset_name)

    # Paths
    output_dir = "/results/training-sessions"
    util.mkdir(output_dir)    
    # Both patch caches are required. Build them with precompute.py --split
    # train and precompute.py --split val before starting training.

    # Set dataset paths
    if args.cached_dataset_type == "alpha":
        raise Exception("Not so fast, alpha datasets not implemented yet!")
    elif args.cached_dataset_type == "beta":
        train_cache_dir = (
            "/root/capsule/data/denoise_net_patch_cache_2026_07_13/patch_cache"
        )
        val_cache_dir = (
            "/root/capsule/data/denoise_net_patch_cache_2026_07_13/val_patch_cache"
        )
    else:
        raise Exception("Not so fast, alpha and beta datasets not implemented yet!")

    # Load experiment config file
    config_name = args.config_name.strip()
    input_config_path = f"/root/capsule/data/{config_name}.json"
    config = util.read_json(input_config_path)
    print("Experiment Config:", config)

    output_config_path = os.path.join(output_dir, f"{config_name}.json")
    util.write_json(output_config_path, config)

    # Resume path. Checkpoints from before the normalization overhaul are NOT
    # compatible: GroupNorm changed the state_dict keys, the residual output
    # changed the semantics, and the old model was trained under percentile
    # normalization. Train from scratch (None), or point this at a NEW-format
    # checkpoint (a dict of {"model", "transform"}) to resume.
    resume_path = None

    # Model (new defaults: residual output + GroupNorm)
    if args.model_name == "unet":
        model = UNet()
    elif args.model_name == "unet_blurpool_rmskip0":
        model = UNet(maxblurpool=True, remove_top_skip=True)

    # Training parameters
    batch_size = config["batch_size"]
    lr = config["lr"]
    max_epochs = 50
    # Use AMP for optimization, but keep validation/checkpoint selection in
    # FP32 so its cratio matches production FP32 inference.
    use_amp_train = args.use_amp_train
    use_amp_validation = False
    # Validate (and consider a checkpoint) after every full-cache epoch.
    val_every = config["val_every"]
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
