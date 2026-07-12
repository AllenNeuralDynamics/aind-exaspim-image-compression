"""Tests for the cache-only BM4DNet training entrypoint."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch


class CachedTrainingTest(unittest.TestCase):
    """Tests the required train and validation cache contract."""

    @classmethod
    def setUpClass(cls):
        """Loads the training script without running its main block."""
        script = Path(__file__).parents[1] / "scripts" / "train_bm4dnet.py"
        cls.namespace = runpy.run_path(str(script))

    def _make_cache(self, root, name, transform=None, config=None, omit=()):
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
        if config is not None:
            files["config.json"] = json.dumps(config)
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
        """Train and validation caches cannot use different transforms."""
        load_transform = self.namespace["_load_cached_transform"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_cache = self._make_cache(root, "train")
            val_cache = self._make_cache(
                root,
                "val",
                transform={
                    "kind": "asinh",
                    "params": {"offset": 0.0, "scale": 16.0},
                },
            )
            with self.assertRaisesRegex(ValueError, "different transforms"):
                load_transform(str(train_cache), str(val_cache))

    def test_cache_provenance_hash_is_stable_and_mismatch_is_rejected(self):
        """Cache IDs are deterministic and teacher mismatches fail early."""
        cache_provenance = self.namespace["_cache_provenance"]
        experiment_provenance = self.namespace["_experiment_provenance"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "teacher_mode": "raw_bm4d",
                "sigma_bm4d": 24,
                "count_dtype": "float32",
            }
            train_cache = self._make_cache(root, "train", config=config)
            val_cache = self._make_cache(root, "val", config=config)
            train_record = cache_provenance(str(train_cache))
            val_record = cache_provenance(str(val_cache))
            self.assertEqual(
                train_record["config_sha256"], val_record["config_sha256"]
            )

            (val_cache / "config.json").write_text(
                json.dumps({**config, "sigma_bm4d": 16})
            )
            with self.assertRaisesRegex(ValueError, "sigma_bm4d"):
                experiment_provenance(str(train_cache), str(val_cache))

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
            "fg_weight": 0,
            "checkpoint_weights": {"fg_mae": 1.0},
            "val_every": 2,
            "preserve_foreground": True,
            "n_train_examples_per_epoch": 7,
            "resume_path": None,
        }
        previous = {key: globals_.get(key) for key in settings}
        globals_.update(settings)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                train_cache = self._make_cache(root, "train")
                val_cache = self._make_cache(root, "val")
                transform = self.namespace["_load_cached_transform"](
                    str(train_cache), str(val_cache)
                )
                train_dataset = SimpleNamespace(
                    raw=[object(), object()], transform=transform
                )
                val_dataset = MagicMock()
                val_dataset.__len__.return_value = 3
                val_dataset.transform = transform
                trainer = MagicMock()

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
                    globals_, {"Trainer": MagicMock(return_value=trainer)}
                ):
                    train(str(train_cache), str(val_cache))

                init_datasets.assert_not_called()
                cached_train.assert_called_once_with(
                    str(train_cache),
                    transform=ANY,
                    preserve_foreground=True,
                    n_examples_per_epoch=7,
                )
                cached_val.assert_called_once_with(
                    str(val_cache),
                    transform=ANY,
                    preserve_foreground=True,
                )
                trainer.run.assert_called_once_with(train_dataset, val_dataset)
                config = trainer.save_config.call_args.args[0]
                self.assertEqual(config["train_cache_dir"], str(train_cache))
                self.assertEqual(config["val_cache_dir"], str(val_cache))
                self.assertEqual(config["transform_cfg"], transform.cfg)
                self.assertIn("provenance", config)
                self.assertIn("train", config["provenance"]["caches"])
                self.assertIn("validation", config["provenance"]["caches"])
                self.assertNotIn("brain_ids_path", config)
                self.assertNotIn("sigma_bm4d", config)
        finally:
            for key, value in previous.items():
                if value is None:
                    globals_.pop(key, None)
                else:
                    globals_[key] = value


if __name__ == "__main__":
    unittest.main()
