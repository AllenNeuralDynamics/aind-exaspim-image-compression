"""
Created on Wed April 30 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for using BM4D-Net to denoise 3D micrscopy images. Includes routines to
extract overlapping patches, normalize and batch process them through a model
on GPU, and stitch denoised patches back into a full 3D volume.

"""

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import itertools
import numpy as np
import torch

from aind_exaspim_image_compression.machine_learning.unet3d import UNet


def predict(
    img,
    model,
    batch_size=32,
    normalization_percentiles=(0.5, 99.9),
    patch_size=64,
    overlap=12,
    trim=5,
    verbose=True
):
    """
    Denoises a 3D image by tiling it into overlapping patches, forming batches
    of patches, and processing each batch through the given model.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image of shape (1, 1, depth, height, width).
    model : torch.nn.Module
        PyTorch model to perform prediction on patches.
    batch_size : int, optional
        Number of patches to process in a batch. Default is 32.
    normalization_percentiles : Tuple[int], optional
        Lower and upper percentiles used for normalization. Default is
        (0.5, 99.9).
    patch_size : int, optional
        Size of the cubic patch extracted from the image. Default is 64.
    overlap : int, optional
        Number of voxels to overlap between patches. Default is 16.
    trim : int, optional
        Number of voxels from the image boundary that are set to zero to
        suppress noisy edge predictions. Default is 5.
    verbose : bool, optional
        Whether to show a progress bar. Default is True.

    Returns
    -------
    denoised : numpy.ndarray
        Denoised image.
    """
    # Preprocess image
    mn, mx = np.percentile(img, normalization_percentiles)
    img = (img - mn) / (mx - mn + 1e-8)
    img = np.clip(img, 0, 5)
    while len(img.shape) < 5:
        img = img[np.newaxis, ...]

    # Initializations
    patch_starts_generator = generate_patch_starts(img, patch_size, overlap)
    n_starts = count_patches(img, patch_size, overlap)
    pbar = tqdm(total=n_starts, desc="Denoise") if verbose else None

    # Main
    accum_pred = np.zeros(img.shape[2:])
    accum_wgt = np.zeros(img.shape[2:])
    for _ in range(0, n_starts, batch_size):
        # Extract batch and run model
        starts = list(itertools.islice(patch_starts_generator, batch_size))
        patches = _predict_batch(img, model, starts, patch_size, trim=trim)

        # Add batch predictions to result
        for patch, start in zip(patches, starts):
            # Compute start and end coordinates
            s = [max(si + trim, 0) for si in start]
            e = [
                min(si + pi, di)
                for si, pi, di in zip(s, patch.shape, img.shape[2:])
            ]

            # Create slices
            pred_slices = tuple(slice(si, ei) for si, ei in zip(s, e))
            patch_slices = tuple(slice(0, ei - si) for si, ei in zip(s, e))

            # Add patch prediction to result
            accum_pred[pred_slices] += patch[patch_slices]
            accum_wgt[pred_slices] += 1

        pbar.update(len(starts)) if verbose else None

    # Postprocess prediction
    denoised = accum_pred[:, ...] / (accum_wgt + 1e-8)
    denoised = np.clip(denoised * (mx - mn) + mn, 0, 2**16 - 1)
    return denoised.astype(np.uint16)


def predict_patch(patch, model, normalization_percentiles=(0.5, 99.9)):
    """
    Denoises a single 3D patch using the provided model.

    Parameters
    ----------
    patch : numpy.ndarray
        3D input patch to denoise.
    model : torch.nn.Module
        PyTorch model used for prediction.
    normalization_percentiles : Tuple[int], optional
        Lower and upper percentiles used for normalization. Default is
        (0.5, 99.9).

    Returns
    -------
    pred : numpy.ndarray
        Denoised 3D patch with the same shape as input patch.
    """
    # Preprocess image
    mn, mx = np.percentile(patch, normalization_percentiles)
    patch = (patch - mn) / (mx - mn + 1e-8)
    patch = np.clip(patch, 0, 5)
    while len(patch.shape) < 5:
        patch = patch[np.newaxis, ...]

    # Run model
    patch = to_tensor(patch)
    with torch.no_grad():
        pred = model(patch)

    # Process output
    pred = np.array(pred.cpu())
    pred = np.clip(pred[0, 0, ...] * (mx - mn) + mn, 0, 2**16 - 1)
    return pred.astype(np.uint16)


def _predict_batch(img, model, starts, patch_size, trim=5):
    # Subroutine
    def get_patch(idx):
        start = starts[idx]
        end = [min(s + patch_size, d) for s, d in zip(start, (D, H, W))]
        patch = img[0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return idx, add_padding(patch, patch_size).astype(np.float32)

    # Parallel patch loading
    D, H, W = img.shape[2:]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(get_patch, range(len(starts))))

    # Reassemble in correct order
    results.sort(key=lambda x: x[0])  # keep consistent ordering
    inputs = np.stack([r[1] for r in results], axis=0)

    # Run model
    inputs = batch_to_tensor(inputs)
    with torch.no_grad():
        outputs = model(inputs).cpu().squeeze(1).numpy()
    return outputs[:, trim:-trim, trim:-trim, trim:-trim]


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
        Image with shape (1, 1, depth, height, width).
    patch_size : int
        Size of each cubic patch along each spatial dimension. This code
        assumes that the patch shape is a cube.
    overlap : int
        Number of voxels that adjacent patches overlap.

    Returns
    -------
    Iterator[Tuple[int]]
        Generates starting coordinates for image patches.
    """
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
        Size of each cubic patch along each spatial dimension.
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
    """
    Loads a pretrained UNet model from a file.

    Parameters
    ----------
    path : str
        Path to the saved model weights (e.g., .pt or .pth file).
    device : str, optional
        Device to load the model onto. Default is "cuda".

    Returns
    -------
    model : torch.nn.Module
        UNet model loaded with weights and set to evaluation mode.
    """
    model = UNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
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
        Tensor on GPU with shape (1, 1, depth, height, width).
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
        Tensor on GPU with shape (batch_size, 1, depth, height, width).
    """
    return to_tensor(arr[:, np.newaxis, ...])
