"""Tests for the cache-only BM4DNet training entrypoint."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch


class CachedTrainingTest(unittest.TestCase):
    """Tests the required train and validation cache contract."""

    @classmethod
    def setUpClass(cls):
        """Loads the training script without running its main block."""
        script = Path(__file__).parents[1] / "scripts" / "train_bm4dnet.py"
        cls.namespace = runpy.run_path(str(script))

    def _make_cache(self, root, name, transform=None, omit=()):
        """Creates the minimal on-disk cache contract for a test."""
        cache_dir = root / name
        cache_dir.mkdir()
        transform = transform or {
            "kind": "asinh",
            "params": {"offset": 0.0, "scale": 32.0},
        }
        files = {
            "raw.npy": b"",
            "teacher.npy": b"",
            "fg.npy": b"",
            "transform.json": json.dumps(transform),
        }
        for filename, contents in files.items():
            if filename in omit:
                continue
            path = cache_dir / filename
            if isinstance(contents, bytes):
                path.write_bytes(contents)
            else:
                path.write_text(contents)
        return cache_dir

    def test_both_cache_paths_are_required(self):
        """Neither cache may be omitted from a training run."""
        load_transform = self.namespace["_load_cached_transform"]
        with self.assertRaisesRegex(ValueError, "train_cache_dir is required"):
            load_transform(None, "/validation")

        with self.assertRaisesRegex(ValueError, "val_cache_dir is required"):
            with patch("os.path.isdir", return_value=True), patch(
                "os.path.isfile", return_value=True
            ), patch.object(
                self.namespace["util"], "read_json", return_value={}
            ):
                load_transform("/training", None)

    def test_cache_contract_requires_all_files(self):
        """A partial cache fails before dataset construction."""
        load_transform = self.namespace["_load_cached_transform"]
        with self.subTest("missing cache directory"):
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                load_transform("/missing-training-cache", "/validation")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cache = self._make_cache(
                root, "train", omit=("teacher.npy", "transform.json")
            )
            val_cache = self._make_cache(root, "val")
            with self.assertRaisesRegex(
                FileNotFoundError, "teacher.npy, transform.json"
            ):
                load_transform(str(train_cache), str(val_cache))

    def test_cache_transforms_must_match(self):
        """Every train and validation cache must use the same transform."""
        load_transform = self.namespace["_load_cached_transform"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cache = self._make_cache(root, "train-0")
            extra_train_cache = self._make_cache(root, "train-1")
            val_cache = self._make_cache(
                root,
                "val",
                transform={
                    "kind": "asinh",
                    "params": {"offset": 0.0, "scale": 16.0},
                },
            )
            with self.assertRaisesRegex(ValueError, "different transforms"):
                load_transform(
                    [str(train_cache), str(extra_train_cache)],
                    [str(val_cache)],
                )

    def test_each_cache_in_an_iterable_is_validated(self):
        """Missing files in any iterable entry fail before construction."""
        load_transform = self.namespace["_load_cached_transform"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cache = self._make_cache(root, "train-0")
            incomplete = self._make_cache(
                root, "train-1", omit=("teacher.npy",)
            )
            val_cache = self._make_cache(root, "val")
            with self.assertRaisesRegex(
                FileNotFoundError, r"train_cache_dir\[1\].*teacher.npy"
            ):
                load_transform(
                    [str(train_cache), str(incomplete)], str(val_cache)
                )

    def test_training_uses_only_cached_datasets(self):
        """Training constructs both cache adapters and records cache config."""
        train = self.namespace["train"]
        globals_ = train.__globals__
        settings = {
            "output_dir": "/results",
            "batch_size": 2,
            "lr": 1e-3,
            "max_epochs": 3,
            "model": object(),
            "use_amp": True,
            "use_amp_validation": False,
            "fg_weight": 0,
            "checkpoint_weights": {"fg_mae": 1.0},
            "val_every": 2,
            "seed": 42,
            "preserve_foreground": True,
            "resume_path": None,
        }
        previous = {key: globals_.get(key) for key in settings}
        globals_.update(settings)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                train_cache = self._make_cache(root, "train-0")
                extra_train_cache = self._make_cache(root, "train-1")
                val_cache = self._make_cache(root, "val")
                transform = self.namespace["_load_cached_transform"](
                    [str(train_cache), str(extra_train_cache)],
                    str(val_cache),
                )
                train_dataset = MagicMock(spec_set=["__len__"])
                train_dataset.__len__.return_value = 2
                val_dataset = MagicMock()
                val_dataset.__len__.return_value = 3
                val_dataset.transform = transform
                trainer = MagicMock()
                trainer_factory = MagicMock(return_value=trainer)

                data_handling = self.namespace["data_handling"]
                with patch.object(
                    data_handling,
                    "CachedPatchDataset",
                    return_value=train_dataset,
                ) as cached_train, patch.object(
                    data_handling,
                    "CachedValidateDataset",
                    return_value=val_dataset,
                ) as cached_val, patch.object(
                    data_handling, "init_datasets"
                ) as init_datasets, patch.dict(
                    globals_, {"Trainer": trainer_factory}
                ):
                    train(
                        [str(train_cache), str(extra_train_cache)],
                        str(val_cache),
                    )

                init_datasets.assert_not_called()
                cached_train.assert_called_once_with(
                    [str(train_cache), str(extra_train_cache)],
                    transform=ANY,
                    preserve_foreground=True,
                )
                cached_val.assert_called_once_with(
                    str(val_cache),
                    transform=ANY,
                    preserve_foreground=True,
                )
                trainer.run.assert_called_once_with(train_dataset, val_dataset)
                config = trainer.save_config.call_args.args[0]
                self.assertEqual(
                    config["train_cache_dir"],
                    [str(train_cache), str(extra_train_cache)],
                )
                self.assertEqual(config["val_cache_dir"], str(val_cache))
                self.assertEqual(config["transform_cfg"], transform.cfg)
                self.assertTrue(config["use_amp"])
                self.assertFalse(config["use_amp_validation"])
                self.assertNotIn("n_train_examples_per_epoch", config)
                self.assertNotIn("brain_ids_path", config)
                self.assertNotIn("sigma_bm4d", config)
                trainer_call = trainer_factory.call_args
                self.assertEqual(trainer_call.kwargs["seed"], 42)
                self.assertTrue(trainer_call.kwargs["use_amp"])
                self.assertFalse(
                    trainer_call.kwargs["use_amp_validation"]
                )
        finally:
            for key, value in previous.items():
                if value is None:
                    globals_.pop(key, None)
                else:
                    globals_[key] = value


if __name__ == "__main__":
    unittest.main()
