"""
Created on Wed April 30 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Denoising routines for 3D microscopy images using patch-based deep learning
inference. Includes functions to extract overlapping patches, normalize and
batch process them through a model on GPU, and stitch denoised patches back
into a full 3D volume.

"""

from tqdm import tqdm

import numpy as np
import torch


def predict(
    img, model, batch_size=32, patch_size=64, overlap=16, verbose=True
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
    coords : List[Tuple[int]]
        List of (i, j, k) starting coordinates of patches processed.
    preds : List[numpy.ndarray]
        List of predicted patches (3D arrays) matching the patch size.
    """
    # Initializations
    batch_coords, batch_inputs, mn_mx = list(), list(), list()
    coords = generate_coords(img, patch_size, overlap)

    # Main
    pbar = tqdm(total=len(coords), desc="Denoise") if verbose else None
    preds = list()
    for idx, (i, j, k) in enumerate(coords):
        # Get end coord
        i_end = min(i + patch_size, img.shape[2])
        j_end = min(j + patch_size, img.shape[3])
        k_end = min(k + patch_size, img.shape[4])

        # Get patch
        patch = img[0, 0, i:i_end, j:j_end, k:k_end]
        mn, mx = np.percentile(patch, 5), np.percentile(patch, 99.9)
        patch = (patch - mn) / mx
        mn_mx.append((mn, mx))

        # Store patch
        patch = add_padding(patch, patch_size)
        batch_inputs.append(patch)
        batch_coords.append((i, j, k))

        # If batch is full or it's the last patch
        if len(batch_inputs) == batch_size or idx == len(coords) - 1:
            # Run model
            input_tensor = batch_to_tensor(np.stack(batch_inputs))
            with torch.no_grad():
                output_tensor = model(input_tensor)

            # Store result
            output_tensor = output_tensor.cpu()
            for cnt in range(output_tensor.shape[0]):
                mn, mx = mn_mx[cnt]
                patch = np.array(output_tensor[cnt, 0, ...])
                preds.append(np.maximum(patch * mx + mn, 0))
                pbar.update(1) if verbose else None

            batch_coords.clear()
            batch_inputs.clear()
            mn_mx.clear()
    return coords, preds


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
    return np.maximum(pred[0, 0, ...] * mx + mn, 0).astype(int)


def stitch(img, coords, preds, patch_size=64, trim=5):
    """
    Stitches overlapping 3D patches back into a full denoised image by
    averaging overlapping regions, with optional trimming of patch borders.

    Parameters
    ----------
    img : numpy.ndarray
        Original image array of shape (batch, channels, depth, height, width).
    coords : List[Tuple[int]]
        List of starting (i, j, k) coordinates for each patch.
    preds : List[numpy.ndarray]
        Predicted patches with shape (patch_size, patch_size, patch_size).
    patch_size : int, optional
        Size of each cubic patch. Default is 64.
    trim : int, optional
        Number of voxels to trim from each side of a patch before stitching.
        Default is 5.

    Returns
    -------
    numpy.ndarray
        Reconstructed image with patches stitched and overlapping areas
        averaged.
    """
    denoised_accum = np.zeros_like(img, dtype=np.float32)
    weight_map = np.zeros_like(img, dtype=np.float32)
    for (i, j, k), pred in zip(coords, preds):
        # Trim prediction
        start, end = trim, patch_size - trim
        pred = pred[start:end, start:end, start:end]

        # Adjust insertion indices
        i_start = i + trim
        j_start = j + trim
        k_start = k + trim

        i_end = i_start + pred.shape[0]
        j_end = j_start + pred.shape[1]
        k_end = k_start + pred.shape[2]

        # Clip to image bounds (for safety)
        i_end = min(i_end, img.shape[2])
        j_end = min(j_end, img.shape[3])
        k_end = min(k_end, img.shape[4])

        i_start = max(i_start, 0)
        j_start = max(j_start, 0)
        k_start = max(k_start, 0)

        denoised_accum[
            0, 0, i_start:i_end, j_start:j_end, k_start:k_end
        ] += pred[: i_end - i_start, : j_end - j_start, : k_end - k_start]
        weight_map[0, 0, i_start:i_end, j_start:j_end, k_start:k_end] += 1

    # Average accumulated
    weight_map[weight_map == 0] = 1
    return denoised_accum / weight_map


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


def generate_coords(img, patch_size, overlap):
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
                coords.append((i, j, k))
    return coords


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
    while(len(arr.shape)) < 5:
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
