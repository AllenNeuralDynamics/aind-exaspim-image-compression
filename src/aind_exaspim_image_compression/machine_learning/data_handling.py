"""
Created on Jan 3 12:30:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training and inference.

"""

from aind_exaspim_dataset_utils.s3_util import get_img_prefix
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import random
import torch

from aind_exaspim_image_compression.utils import img_util, util
from aind_exaspim_image_compression.utils.img_util import BM4D
from aind_exaspim_image_compression.utils.swc_util import Reader


# --- Custom Datasets ---
class TrainDataset(Dataset):
    """
    A PyTorch Dataset for sampling 3D patches from whole-brain images and
    applying the BM4D denoising algorithm. The dataset's __getitem__ method
    returns both the original and denoised patches. Optionally, the patch
    sampling maybe biased toward foreground regions by using the voxel
    coordinates from SWC files that represent neuron tracings.
    """

    def __init__(
        self,
        patch_shape,
        anisotropy=(0.748, 0.748, 1.0),
        boundary_buffer=4000,
        foreground_sampling_rate=0.5,
        n_examples_per_epoch=100,
        sigma=50,
    ):
        # Call parent class
        super(TrainDataset, self).__init__()

        # Class attributes
        self.anisotropy = anisotropy
        self.boundary_buffer = boundary_buffer
        self.denoise_bm4d = BM4D(sigma=sigma)
        self.foreground_sampling_rate = foreground_sampling_rate
        self.n_examples_per_epoch = n_examples_per_epoch
        self.patch_shape = patch_shape
        self.swc_reader = Reader()

        # Data structures
        self.foreground = dict()
        self.imgs = dict()

    # --- Ingest data ---
    def ingest_brain(self, brain_id, img_path, swc_pointer):
        self.foreground[brain_id] = self.load_swcs(swc_pointer)
        self.imgs[brain_id] = img_util.read(img_path)

    def load_swcs(self, swc_pointer):
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

    # --- Core Routines ---
    def __getitem__(self, dummy_input):
        brain_id = self.sample_brain()
        voxel = self.sample_voxel(brain_id)
        return self.denoise_bm4d(self.get_patch(brain_id, voxel))

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
    def __len__(self):
        """
        Counts the number of whole-brain images in the dataset.

        Returns
        -------
        int
            Number of whole-brain images in the dataset.
        """
        return self.n_examples_per_epoch

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

    def __init__(self, patch_shape):
        # Call parent class
        super(ValidateDataset, self).__init__()

        # Instance attributes
        self.patch_shape = patch_shape
        self.denoise_bm4d = BM4D()

        # Data structures
        self.ids = list()
        self.imgs = dict()
        self.denoised = list()
        self.noise = list()
        self.mn_mxs = list()

    def __len__(self):
        """
        Counts the number of whole-brain images in the dataset.

        Returns
        -------
        int
            Number of whole-brain images in the dataset.
        """
        return len(self.ids)

    def ingest_brain(self, brain_id, img_path):
        self.imgs[brain_id] = img_util.read(img_path)

    def ingest_example(self, brain_id, voxel):
        # Get clean image
        noise, denoised, mn_mx = self.denoise_bm4d(
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


# --- Custom Dataloader ---
class DataLoader:
    """
    DataLoader that uses multithreading to fetch image patches from the cloud
    to form batches.
    """

    def __init__(self, dataset, batch_size=16):
        """
        Instantiates a DataLoader object.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to iterated over.
        batch_size : int
            Number of examples in each batch.
        """
        # Instance attributes
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_shape = dataset.patch_shape

    def __iter__(self):
        """
        Iterates over the dataset and yields batches of examples.

        Returns
        -------
        iterator
            Yields batches of examples.
        """
        for idx in range(0, len(self.dataset), self.batch_size):
            yield self._load_batch(idx)

    def _load_batch(self, start_idx):
        # Compute batch size
        n_remaining_examples = len(self.dataset) - start_idx
        batch_size = min(self.batch_size, n_remaining_examples)

        # Generate batch
        with ProcessPoolExecutor() as executor:
            # Assign processs
            processes = list()
            for idx in range(start_idx, start_idx + batch_size):
                processes.append(
                    executor.submit(self.dataset.__getitem__, idx)
                )

            # Process results
            shape = (batch_size, 1,) + self.patch_shape
            noise_patches = np.zeros(shape)
            denoised_patches = np.zeros(shape)
            mn_mxs = np.zeros((self.batch_size, 2))
            for i, process in enumerate(as_completed(processes)):
                noise, denoised, mn_mx = process.result()
                noise_patches[i, 0, ...] = noise
                denoised_patches[i, 0, ...] = denoised
                mn_mxs[i, :] = mn_mx
        return to_tensor(noise_patches), to_tensor(denoised_patches), mn_mxs


# --- Helpers ---
def init_datasets(
    brain_ids,
    img_paths_json,
    patch_shape,
    foreground_sampling_rate=0.5,
    n_train_examples_per_epoch=100,
    n_validate_examples=0,
    swc_dict=None
):
    # Initializations
    train_dataset = TrainDataset(
        patch_shape,
        foreground_sampling_rate=foreground_sampling_rate,
        n_examples_per_epoch=n_train_examples_per_epoch,
    )
    val_dataset = ValidateDataset(patch_shape)

    # Load data
    for brain_id in tqdm(brain_ids, desc="Load Data"):
        # Set paths
        img_path = get_img_prefix(brain_id, img_paths_json) + str(0)
        if swc_dict:
            swc_pointer = deepcopy(swc_dict)
            swc_pointer["path"] += f"/{brain_id}/world"
        else:
            swc_pointer = None

        # Ingest data
        train_dataset.ingest_brain(brain_id, img_path, swc_pointer)
        val_dataset.ingest_brain(brain_id, img_path)

    # Generate validation examples
    for _ in range(n_validate_examples):
        brain_id = train_dataset.sample_brain()
        voxel = train_dataset.sample_voxel(brain_id)
        val_dataset.ingest_example(brain_id, voxel)
    return train_dataset, val_dataset


def to_tensor(arr):
    """
    Converts the given NumPy array to a torch tensor.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Array converted to a torch tensor.
    """
    return torch.tensor(arr, dtype=torch.float)
