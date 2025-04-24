"""
Created on Thu Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training and inference.

"""

from abc import ABC, abstractmethod
from careamics.transforms.n2v_manipulate import N2VManipulate
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed
)
from scipy.spatial import distance
from torch.utils.data import Dataset
from tqdm import tqdm

import logging
import numpy as np
import os
import pytorch_lightning as L
import random
import torch

from aind_exaspim_image_compression.utils import img_util, util
from aind_exaspim_image_compression.utils.swc_util import Reader


logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        bboxes_path,
        img_prefixes_path,
        swc_dir,
        batch_size=16,
        foreground_sampling_rate=0.2,
        n_upds=100,
        n_validate_examples=200,
        patch_shape=(64, 64, 64)
    ):
        # Call parent class
        super(DataModule, self).__init__()

        # Instance attributes
        self.batch_size = batch_size
        self.foreground_sampling_rate = foreground_sampling_rate
        self.n_upds = n_upds
        self.n_validate_examples = n_validate_examples
        self.patch_shape = patch_shape

        # Paths
        self.bboxes_path = bboxes_path
        self.img_prefixes_path = img_prefixes_path
        self.swc_dir = swc_dir

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset = init_datasets(
                self.bboxes_path,
                self.img_prefixes_path,
                self.swc_dir,
                self.patch_shape,
                self.n_validate_examples,
                self.foreground_sampling_rate,
            )

    def train_dataloader(self):
        train_dataloader = TrainN2VDataLoader(
            self.train_dataset, batch_size=self.batch_size, n_upds=self.n_upds
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = ValidateN2VDataLoader(
            self.val_dataset, batch_size=self.batch_size
        )
        return val_dataloader


# --- Custom Datasets ---
class TrainDataset(Dataset):
    def __init__(
        self,
        patch_shape,
        transform,
        anisotropy=(0.748, 0.748, 1.0),
        foreground_sampling_rate=0.2
    ):
        # Call parent class
        super(TrainDataset, self).__init__()

        # Class attributes
        self.anisotropy = anisotropy
        self.foreground_sampling_rate = foreground_sampling_rate
        self.patch_shape = patch_shape
        self.swc_reader = Reader()
        self.transform = transform

        # Data structures
        self.bboxes = dict()
        self.foreground = dict()
        self.imgs = dict()

    def ingest_img(self, brain_id, img_prefix, bbox, swc_pointer=None):
        # Adjust bbox
        for i in range(3):
            bbox["min"][i] += self.patch_shape[i]
            bbox["max"][i] -= self.patch_shape[i]        

        # Store result
        self.bboxes[brain_id] = bbox
        self.foreground[brain_id] = self.ingest_swcs(swc_pointer)
        self.imgs[brain_id] = img_util.open_img(img_prefix)

    def ingest_swcs(self, swc_pointer):
        if swc_pointer:
            # Initializations
            swc_dicts = self.swc_reader.read(swc_pointer)
            n_points = np.sum([len(swc_dict["xyz"]) for swc_dict in swc_dicts])

            # Extract foreground voxels
            start = 0
            foreground = np.zeros((n_points, 3), dtype=np.int32)
            for swc_dict in swc_dicts:
                end = start + len(swc_dict["xyz"])
                foreground[start:end] = self.to_voxels(swc_dict["xyz"])
                start = end
        else:
            foreground = set()
        return foreground
        
    def __len__(self):
        """
        Counts the number of whole-brain images in dataset.

        Parameters
        ----------
        None

        Returns
        -------
        Number of whole-brain images in dataset.

        """
        return len(self.imgs)

    def __getitem__(self, dummy_input):
        brain_id = self.sample_brain()
        voxel = self.sample_voxel(brain_id)
        return self.transform(self.get_patch(brain_id, voxel))

    def sample_brain(self):
        return util.sample_once(self.imgs.keys())

    def sample_voxel(self, brain_id):
        if random.random() < self.foreground_sampling_rate:
            idx = random.randint(0, len(self.foreground[brain_id]) - 1)
            shift = np.random.randint(0, 32, size=3)
            return tuple(self.foreground[brain_id][idx] + shift)
        else:
            voxel = list()
            for i in range(3):
                lower = self.bboxes[brain_id]["min"][i]
                upper = self.bboxes[brain_id]["max"][i]
                voxel.append(random.randint(lower, upper))
            return tuple(voxel)

    # --- Helpers ---
    def get_patch(self, brain_id, voxel):
        s, e = img_util.get_start_end(voxel, self.patch_shape)
        patch = self.imgs[brain_id][0, 0, s[0]: e[0], s[1]: e[1], s[2]: e[2]]
        return patch #/ np.percentile(patch, 99.9)

    def to_voxels(self, xyz_arr):
        for i in range(3):
            xyz_arr[:, i] = xyz_arr[:, i] / self.anisotropy[i]
        return np.flip(xyz_arr, axis=1).astype(int)

    
class ValidateDataset(Dataset):
    def __init__(self, patch_shape, transform):
        # Call parent class
        super(ValidateDataset, self).__init__()

        # Instance attributes
        self.patch_shape = patch_shape
        self.transform = transform

        # Data structures
        self.imgs = dict()
        self.examples = list()

    def __len__(self):
        """
        Counts the number of whole-brain images in dataset.

        Parameters
        ----------
        None

        Returns
        -------
        Number of whole-brain images in dataset.

        """
        return len(self.examples)

    def ingest_img(self, brain_id, img_prefix):
        self.imgs[brain_id] = img_util.open_img(img_prefix)

    def ingest_example(self, brain_id, voxel):
        self.examples.append((brain_id, voxel))

    def __getitem__(self, idx):
        brain_id, voxel = self.examples[idx]
        return self.transform(self.get_patch(brain_id, voxel))

    # --- Helpers ---
    def get_patch(self, brain_id, voxel):
        s, e = img_util.get_start_end(voxel, self.patch_shape)
        patch = self.imgs[brain_id][0, 0, s[0]: e[0], s[1]: e[1], s[2]: e[2]]
        return patch / np.percentile(patch, 99.9)


# --- Custom Dataloaders ---
class DataLoader(ABC):
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.

    """

    def __init__(self, dataset, batch_size=16):
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_shape = dataset.patch_shape

    def __iter__(self):
        for idx in self._get_iterator():
            yield self._load_batch(idx)

    @abstractmethod
    def _get_iterator(self):
        pass

    @abstractmethod
    def _load_batch(self, idx):
        pass


class TrainN2VDataLoader(DataLoader):
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches to train Noise2Void (N2V).

    """

    def __init__(self, dataset, batch_size=16, n_upds=100):
        # Call parent class
        super().__init__(dataset, batch_size)

        # Instance attributes
        self.n_upds = n_upds

    def _get_iterator(self):
        return range(self.n_upds)

    def _load_batch(self, dummy_input):
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for _ in range(self.batch_size):
                threads.append(
                    executor.submit(self.dataset.__getitem__, -1)
                )

            # Process results
            masked_patches = np.zeros((self.batch_size, 1,) + self.patch_shape)
            patches = np.zeros((self.batch_size, 1,) + self.patch_shape)
            masks = np.zeros((self.batch_size, 1,) + self.patch_shape)
            for i, thread in enumerate(as_completed(threads)):
                masked_patch, patch, mask = thread.result()
                masked_patches[i, 0, ...] = masked_patch
                patches[i, 0, ...] = patch
                masks[i, 0, ...] = mask
        return to_tensor(masked_patches), to_tensor(patches), to_tensor(masks)


class TrainBM4DDataLoader(DataLoader):
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.

    """

    def __init__(self, dataset, batch_size=8, n_upds=20):
        """
        Constructs a multithreaded data loader.

        Parameters
        ----------
        dataset : Dataset.ProposalDataset
            Instance of custom dataset.
        batch_size : int
            Number of samples per batch.

        Returns
        -------
        None

        """
        # Call parent class
        super().__init__(dataset, batch_size)

        # Instance attributes
        self.n_upds = n_upds

    def _get_iterator(self):
        return range(self.n_upds)

    def _load_batch(self, dummy_input):
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = list()
            for _ in range(self.batch_size):
                processes.append(
                    executor.submit(self.dataset.__getitem__, -1)
                )

            # Process results
            noise_patches = np.zeros((self.batch_size, 1,) + self.patch_shape)
            clean_patches = np.zeros((self.batch_size, 1,) + self.patch_shape)
            for i, process in enumerate(as_completed(processes)):
                noise, clean = process.result()
                noise_patches[i, 0, ...] = noise
                clean_patches[i, 0, ...] = clean
        return to_tensor(noise_patches), to_tensor(clean_patches)


class ValidateN2VDataLoader(DataLoader):
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.

    """

    def __init__(self, dataset, batch_size=8):
        super().__init__(dataset, batch_size)

    def _get_iterator(self):
        return range(0, len(self.dataset), self.batch_size)

    def _load_batch(self, start_idx):
        # Compute batch size
        n_remaining_examples = len(self.dataset) - start_idx
        batch_size = min(self.batch_size, n_remaining_examples)

        # Generate batch
        with ThreadPoolExecutor() as executor:
            # Assign threads
            threads = list()
            for idx_shift in range(batch_size):
                idx = start_idx + idx_shift
                threads.append(
                    executor.submit(self.dataset.__getitem__, idx)
                )

            # Process results
            masked_patches = np.zeros((batch_size, 1,) + self.patch_shape)
            patches = np.zeros((batch_size, 1,) + self.patch_shape)
            masks = np.zeros((batch_size, 1,) + self.patch_shape)
            for i, thread in enumerate(as_completed(threads)):
                masked_patch, patch, mask = thread.result()
                masked_patches[i, 0, ...] = masked_patch
                patches[i, 0, ...] = patch
                masks[i, 0, ...] = mask
        return to_tensor(masked_patches), to_tensor(patches), to_tensor(masks)


class ValidateBM4DDataLoader(DataLoader):
    """
    DataLoader that uses multiprocessing to fetch image patches from the cloud
    to form batches.

    """

    def __init__(self, dataset, batch_size=8):
        super().__init__(dataset, batch_size)

    def _get_iterator(self):
        return range(0, len(self.dataset), self.batch_size)

    def _load_batch(self, start_idx):
        # Compute batch size
        n_remaining_examples = len(self.dataset) - start_idx
        batch_size = min(self.batch_size, n_remaining_examples)

        # Generate batch
        with ProcessPoolExecutor() as executor:
            # Assign processs
            processes = list()
            for idx_shift in range(batch_size):
                idx = start_idx + idx_shift
                processes.append(
                    executor.submit(self.dataset.__getitem__, idx)
                )

            # Process results
            noise_patches = np.zeros((self.batch_size, 1,) + self.patch_shape)
            clean_patches = np.zeros((self.batch_size, 1,) + self.patch_shape)
            for i, process in enumerate(as_completed(processes)):
                noise, clean = process.result()
                noise_patches[i, 0, ...] = noise
                clean_patches[i, 0, ...] = clean
        return to_tensor(noise_patches), to_tensor(clean_patches)


# --- Helpers ---
def init_datasets(
    bboxes_path,
    img_prefixes_path,
    swc_dir,
    patch_shape,
    n_validate_examples,
    foreground_sampling_rate=0.5,
    method="n2v",
):
    # Initializations
    bboxes = util.read_json(bboxes_path)
    img_prefixes = util.read_json(img_prefixes_path)
    if method == "n2v":
        transform = N2VManipulate()
    elif method == "bm4d":
        transform = img_util.BM4D()
    else:
        raise ValueError(f"Method {method} is not recognized!")

    val_dataset = ValidateDataset(patch_shape, transform)
    train_dataset = TrainDataset(
        patch_shape,
        transform,
        foreground_sampling_rate=foreground_sampling_rate
    )

    # Load data
    for brain_id, bbox in tqdm(bboxes.items(), desc="Load Data"):
        img_prefix = img_prefixes[brain_id] + str(0)
        swc_pointer = os.path.join(swc_dir, brain_id)
        train_dataset.ingest_img(brain_id, img_prefix, bbox, swc_pointer)
        val_dataset.ingest_img(brain_id, img_prefix)
        
    # Generate validation examples
    for _ in range(n_validate_examples):
        brain_id = train_dataset.sample_brain()
        voxel = train_dataset.sample_voxel(brain_id)
        val_dataset.ingest_example(brain_id, voxel)

    return train_dataset, val_dataset


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float)
