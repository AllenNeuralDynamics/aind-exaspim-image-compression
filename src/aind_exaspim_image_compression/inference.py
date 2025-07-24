"""
Created on Wed April 30 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Denoising routines for 3D microscopy images using patch-based deep learning
inference. Includes functions to extract overlapping patches, normalize and
batch process them through a model on GPU, and stitch denoised patches back
into a full 3D volume.

"""
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from numcodecs import blosc
from tqdm import tqdm

import itertools
import numpy as np
import torch

from aind_exaspim_image_compression.machine_learning.unet3d import UNet
from aind_exaspim_image_compression.utils import img_util


def predict(
    img,
    model,
    denoised=None,
    batch_size=32,
    patch_size=64,
    overlap=12,
    trim=5,
    verbose=True
):
    """
    Denoises a 3D image by processing patches in batches and running deep
    learning model.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image of shape (1, 1, depth, height, width).
    model : torch.nn.Module
        PyTorch model to perform prediction on patches.
    batch_size : int, optional
        Number of patches to process in a batch. Default is 32.
    patch_size : int, optional
        Size of the cubic patch extracted from the image. Default is 64.
    overlap : int, optional
        Number of voxels to overlap between patches. Default is 16.
    verbose : bool, optional
        Whether to show a tqdm progress bar. Default is True.

    Returns
    -------
    numpy.ndarray
        Denoised image.
    """
    # Adjust image dimenions
    while len(img.shape) < 5:
        img = img[np.newaxis, ...]

    # Initializations
    starts_generator = generate_patch_starts(img, patch_size, overlap)
    n_starts = count_patches(img, patch_size, overlap)
    if denoised is None:
        denoised = np.zeros_like(img, dtype=np.uint16)

    # Main
    pbar = tqdm(total=n_starts, desc="Denoise") if verbose else None
    for i in range(0, n_starts, batch_size):
        # Run model
        starts = list(itertools.islice(starts_generator, batch_size))
        patches = _predict_batch(img, model, starts, patch_size, trim)

        # Store result
        for patch, start in zip(patches, starts):
            start = [max(s + trim, 0) for s in start]
            end = [start[i] + patch.shape[i] for i in range(3)]
            end = [min(e, s) for e, s in zip(end, img.shape[2:])]
            denoised[
                0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]
            ] = patch[: end[0] - start[0], : end[1] - start[1], : end[2] - start[2]]
        pbar.update(len(starts)) if verbose else None
    return denoised


def predict_largescale(
    img,
    model,
    output_path,
    compressor,
    batch_size=32,
    patch_size=64,
    overlap=16,
    trim=5,
    verbose=True
):
    # Initializations
    denoised = img_util.init_ome_zarr(img, output_path, compressor=compressor)
    predict(img, model, denoised=denoised)


def predict_patch(patch, model):
    """
    Denoised a single 3D patch using the provided model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model used for prediction.
    patch : numpy.ndarray
        3D input patch to denoise.

    Returns
    -------
    numpy.ndarray
        Denoised 3D patch with the same shape as input patch.
    """
    # Run model
    mn, mx = np.percentile(patch, 5), np.percentile(patch, 99.9)
    patch = to_tensor((patch - mn) / max(mx, 1))
    with torch.no_grad():
        output_tensor = model(patch)

    # Process output
    pred = np.array(output_tensor.cpu())
    return np.maximum(pred[0, 0, ...] * mx + mn, 0).astype(np.uint16)


def _predict_batch(img, model, starts, patch_size, trim=5):
    # Subroutine
    def read_patch(i):
        start = starts[i]
        end = [min(s + patch_size, d) for s, d in zip(start, img.shape[2:])]
        patch = img[0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        mn, mx = np.percentile(patch, 5), np.percentile(patch, 99.9)
        patch = add_padding((patch - mn) / mx, patch_size)
        return i, patch, (mn, mx)

    # Main
    with ThreadPoolExecutor() as executor:
        # Read patches
        threads = list()
        for i in range(len(starts)):
            threads.append(executor.submit(read_patch, i))

        # Compile batch
        inputs = np.zeros((len(starts),) + (patch_size,) * 3)
        mn_mx = len(starts) * [None]
        for thread in as_completed(threads):
            i, patch_i, mn_mx_i = thread.result()
            mn_mx[i] = mn_mx_i
            inputs[i, ...] = patch_i

        # Run model
        inputs = batch_to_tensor(inputs)
        with torch.no_grad():
            outputs = model(inputs)
        outputs = np.array(outputs.cpu()).squeeze(1)

        # Store result
        preds = list()
        start, end = trim, patch_size - trim
        for i in range(outputs.shape[0]):
            mn, mx = mn_mx[i]
            pred = np.maximum(outputs[i] * mx + mn, 0).astype(np.uint16)
            preds.append(pred[start:end, start:end, start:end])
    return preds


# --- Helpers ---
def add_padding(patch, patch_size):
    """
    Pads a 3D patch with zeros to reach the desired patch shape.

    Parameters
    ----------
    patch : numpy.ndarray
        3D array representing the patch to be padded.
    patch_size : int
        Target size for each dimension after padding.

    Returns
    -------
    numpy.ndarray
        Zero-padded patch with shape (patch_size, patch_size, patch_size).
    """
    pad_width = [
        (0, patch_size - patch.shape[0]),
        (0, patch_size - patch.shape[1]),
        (0, patch_size - patch.shape[2]),
    ]
    return np.pad(patch, pad_width, mode="constant", constant_values=0)


def generate_patch_starts(img, patch_size, overlap):
    """
    Generates starting coordinates for 3D patches extracted from an image
    tensor, based on specified patch size and overlap.

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray
        Input image tensor with shape (batch, channels, depth, height, width).
    patch_size : int
        The size of each cubic patch along each spatial dimension.
    overlap : int
        Number of voxels that adjacent patches overlap.

    Returns
    -------
    coords : List[Tuple[int]]
        List of (depth_start, height_start, width_start) coordinates for image
        patches.
    """
    coords = list()
    stride = patch_size - overlap
    for i in range(0, img.shape[2] - patch_size + stride, stride):
        for j in range(0, img.shape[3] - patch_size + stride, stride):
            for k in range(0, img.shape[4] - patch_size + stride, stride):
                yield (i, j, k)


def count_patches(img, patch_size, overlap):
    """
    Counts the number of patches within a 3D image for a given patch size
    and overlap between patches.

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray
        Input image tensor with shape (batch, channels, depth, height, width).
    patch_size : int
        The size of each cubic patch along each spatial dimension.
    overlap : int
        Number of voxels that adjacent patches overlap.

    Returns
    -------
    int
        Number of patches.
    """
    stride = patch_size - overlap
    d_range = range(0, img.shape[2] - patch_size + stride, stride)
    h_range = range(0, img.shape[3] - patch_size + stride, stride)
    w_range = range(0, img.shape[4] - patch_size + stride, stride)
    return len(d_range) * len(h_range) * len(w_range)


def load_model(path, device="cuda"):
    model = UNet()
    model.load_state_dict(torch.load(path))
    model.eval().to(device)
    return model


def to_tensor(arr):
    """
    Converts a NumPy array containing to a PyTorch tensor and moves it to the
    GPU.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Tensor on GPU, with shape (1, 1, depth, height, width).
    """
    while (len(arr.shape)) < 5:
        arr = arr[np.newaxis, ...]
    return torch.tensor(arr).to("cuda", dtype=torch.float)


def batch_to_tensor(arr):
    """
    Converts a NumPy array containing a batch of inputs to a PyTorch tensor
    and moves it to the GPU.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted, with shape (batch_size, depth, height, width).

    Returns
    -------
    torch.Tensor
        Tensor on GPU, with shape (batch_size, 1, depth, height, width).
    """
    return to_tensor(arr[:, np.newaxis, ...])
