"""
Created on Thu Dec 5 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training and inference.

"""

from abc import ABC, abstractmethod
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import random
import torch

from aind_exaspim_image_compression.utils import img_util, util
from aind_exaspim_image_compression.utils.swc_util import Reader


# --- Custom Datasets ---
class TrainDataset(Dataset):
    def __init__(
        self,
        patch_shape,
        transform,
        anisotropy=(0.748, 0.748, 1.0),
        boundary_buffer=4000,
        foreground_sampling_rate=0.5,
    ):
        # Call parent class
        super(TrainDataset, self).__init__()

        # Class attributes
        self.anisotropy = anisotropy
        self.boundary_buffer = boundary_buffer
        self.foreground_sampling_rate = foreground_sampling_rate
        self.patch_shape = patch_shape
        self.swc_reader = Reader()
        self.transform = transform

        # Data structures
        self.foreground = dict()
        self.imgs = dict()

    def ingest_img(self, brain_id, img_path, swc_pointer):
        self.foreground[brain_id] = self.ingest_swcs(swc_pointer)
        self.imgs[brain_id] = img_util.read(img_path)

    def ingest_swcs(self, swc_pointer):
        if swc_pointer:
            # Initializations
            swc_dicts = self.swc_reader.read(swc_pointer)
            n_points = np.sum([len(d["xyz"]) for d in swc_dicts])

            # Extract foreground voxels
            if n_points > 0:
                start = 0
                foreground = np.zeros((n_points, 3), dtype=np.int32)
                for swc_dict in swc_dicts:
                    end = start + len(swc_dict["xyz"])
                    foreground[start:end] = self.to_voxels(swc_dict["xyz"])
                    start = end
                return foreground
        return set()

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
        sample_foreground = random.random() < self.foreground_sampling_rate
        if sample_foreground and len(self.foreground[brain_id]) > 0:
            idx = random.randint(0, len(self.foreground[brain_id]) - 1)
            shift = np.random.randint(0, 16, size=3)
            return tuple(self.foreground[brain_id][idx] + shift)
        else:
            voxel = list()
            for s in self.imgs[brain_id].shape[2::]:
                upper = s - self.boundary_buffer
                voxel.append(random.randint(self.boundary_buffer, upper))
            return tuple(voxel)

    # --- Helpers ---
    def get_patch(self, brain_id, voxel):
        s, e = img_util.get_start_end(voxel, self.patch_shape)
        return self.imgs[brain_id][0, 0, s[0]: e[0], s[1]: e[1], s[2]: e[2]]

    def to_voxels(self, xyz_arr):
        for i in range(3):
            xyz_arr[:, i] = xyz_arr[:, i] / self.anisotropy[i]
        return np.flip(xyz_arr, axis=1).astype(int)

    def update_foreground_sampling_rate(self, foreground_sampling_rate):
        self.foreground_sampling_rate = foreground_sampling_rate


class ValidateDataset(Dataset):
    def __init__(self, patch_shape, transform):
        # Call parent class
        super(ValidateDataset, self).__init__()

        # Instance attributes
        self.patch_shape = patch_shape
        self.transform = transform

        # Data structures
        self.ids = list()
        self.imgs = dict()
        self.denoised = list()
        self.noise = list()
        self.mn_mxs = list()

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
        return len(self.ids)

    def ingest_img(self, brain_id, img_path):
        self.imgs[brain_id] = img_util.read(img_path)

    def ingest_example(self, brain_id, voxel):
        # Get clean image
        noise, denoised, mn_mx = self.transform(
            self.get_patch(brain_id, voxel)
        )

        # Store results
        self.ids.append((brain_id, voxel))
        self.denoised.append(denoised)
        self.noise.append(noise)
        self.mn_mxs.append(mn_mx)

    def __getitem__(self, idx):
        return self.noise[idx], self.denoised[idx], self.mn_mxs[idx]

    # --- Helpers ---
    def get_patch(self, brain_id, voxel):
        s, e = img_util.get_start_end(voxel, self.patch_shape)
        patch = self.imgs[brain_id][
            0, 0, s[0]: e[0], s[1]: e[1], s[2]: e[2]
        ]
        return patch


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
                threads.append(executor.submit(self.dataset.__getitem__, -1))

            # Process results
            shape = (self.batch_size, 1,) + self.patch_shape
            masked_patches = np.zeros(shape)
            patches = np.zeros(shape)
            masks = np.zeros(shape)
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
                processes.append(executor.submit(self.dataset.__getitem__, -1))

            # Process results
            shape = (self.batch_size, 1,) + self.patch_shape
            noise_patches = np.zeros(shape)
            clean_patches = np.zeros(shape)
            for i, process in enumerate(as_completed(processes)):
                noise, clean, _ = process.result()
                noise_patches[i, 0, ...] = noise
                clean_patches[i, 0, ...] = clean
        return to_tensor(noise_patches), to_tensor(clean_patches), None


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
                threads.append(executor.submit(self.dataset.__getitem__, idx))

            # Process results
            shape = (batch_size, 1,) + self.patch_shape
            masked_patches = np.zeros(shape)
            patches = np.zeros(shape)
            masks = np.zeros(shape)
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
            shape = (batch_size, 1,) + self.patch_shape
            noise_patches = np.zeros(shape)
            clean_patches = np.zeros(shape)
            mn_mxs = np.zeros((self.batch_size, 2))
            for i, process in enumerate(as_completed(processes)):
                noise, clean, mn_mx = process.result()
                noise_patches[i, 0, ...] = noise
                clean_patches[i, 0, ...] = clean
                mn_mxs[i, :] = mn_mx
        return to_tensor(noise_patches), to_tensor(clean_patches), mn_mxs


# --- Helpers ---
def init_datasets(
    brain_ids,
    img_paths_json,
    patch_shape,
    n_validate_examples,
    foreground_sampling_rate=0.5,
    swc_dict=None
):
    # Initializations
    transform = img_util.BM4D()
    train_dataset = TrainDataset(
        patch_shape,
        transform,
        foreground_sampling_rate=foreground_sampling_rate,
    )
    val_dataset = ValidateDataset(patch_shape, transform)

    # Load data
    for brain_id in tqdm(brain_ids, desc="Load Data"):
        img_path = img_util.get_img_prefix(brain_id, img_paths_json)
        img_path += str(0)
        if swc_dict:
            swc_pointer = deepcopy(swc_dict)
            swc_pointer["path"] += f"/{brain_id}/world"
        else:
            swc_pointer = None

        train_dataset.ingest_img(brain_id, img_path, swc_pointer)
        val_dataset.ingest_img(brain_id, img_path)

    # Generate validation examples
    for _ in range(n_validate_examples):
        brain_id = train_dataset.sample_brain()
        voxel = train_dataset.sample_voxel(brain_id)
        val_dataset.ingest_example(brain_id, voxel)
    return train_dataset, val_dataset


def to_tensor(arr):
    """
    Converts a numpy array to a tensor.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Array converted to tensor.

    """
    return torch.tensor(arr, dtype=torch.float)
