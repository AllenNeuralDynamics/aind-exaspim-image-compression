"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from bm4d import bm4d
from botocore.config import Config
from numcodecs import Blosc
from ome_zarr.writer import write_image, write_multiscale
from ome_zarr.scale import Scaler
from pathlib import Path
from typing import Union, Any
from xarray_multiscale import multiscale, windowed_mode

import argparse
import boto3
import botocore
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import s3fs
import shutil
import tifffile
import zarr


# --- Image Reader ---
def open_img(prefix):
    """
    Opens an image stored in an S3 bucket as a Zarr array.

    Parameters
    ----------
    prefix : str
        Prefix (or path) within the S3 bucket where the image is stored.

    Returns
    -------
    zarr.core.Array
        A Zarr object representing an image.

    """
    fs = s3fs.S3FileSystem(config_kwargs={"max_pool_connections": 50})
    store = s3fs.S3Map(root=prefix, s3=fs)
    return zarr.open(store, mode="r")


def get_patch(img, voxel, shape, from_center=True):
    """
    Extracts a patch from an image based on the given voxel coordinate and
    patch shape.

    Parameters
    ----------
    img : zarr.core.Array
         A Zarr object representing an image.
    voxel : Tuple[int]
        Voxel coordinate used to extract patch.
    shape : Tuple[int]
        Shape of the image patch to extract.
    from_center : bool, optional
        Indicates whether the given voxel is the center or top, left, front
        corner of the patch to be extracted.

    Returns
    -------
    numpy.ndarray
        Patch extracted from the given image.

    """
    # Get image patch coordiantes
    start, end = get_start_end(voxel, shape, from_center=from_center)
    valid_start = any([s >= 0 for s in start])
    valid_end = any([e < img.shape[i + 2] for i, e in enumerate(end)])

    # Get image patch
    if valid_start and valid_end:
        return img[
            0, 0, start[0]: end[0], start[1]: end[1], start[2]: end[2]
        ]
    else:
        return np.ones(shape)


def calculate_offsets(img, window_shape, overlap):
    """
    Generates a list of 3D coordinates representing the front-top-left corner
    by sliding a window over a 3D image, given a specified window size and
    overlap between adjacent windows.

    Parameters
    ----------
    img : zarr.core.Array
        Input 3D image.
    window_shape : Tuple[int]
        Shape of the sliding window.
    overlap : Tuple[int]
        Overlap between adjacent sliding windows.

    Returns
    -------
    List[Tuple[int]]
        List of 3D voxel coordinates that represent the front-top-left corner.

    """
    # Calculate stride based on the overlap and window size
    stride = tuple(w - o for w, o in zip(window_shape, overlap))
    i_stride, j_stride, k_stride = stride

    # Get dimensions of the window
    _, _, i_dim, j_dim, k_dim = img.shape
    i_win, j_win, k_win = window_shape

    # Loop over the  with the sliding window
    coords = []
    for i in range(0, i_dim - i_win + 1, i_stride):
        for j in range(0, j_dim - j_win + 1, j_stride):
            for k in range(0, k_dim - k_win + 1, k_stride):
                coords.append((i, j, k))
    return coords


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the image patch to be read.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate that specifies either the center or front-top-left
        corner of the patch to be read.
    shape : Tuple[int]
        Shape of the image patch to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the patch or the front-top-left corner. The default is True.

    Return
    ------
    Tuple[List[int]]
        Start and end indices of the image patch to be read.

    """
    if from_center:
        start = [voxel[i] - shape[i] // 2 for i in range(3)]
        end = [voxel[i] + shape[i] // 2 for i in range(3)]
    else:
        start = voxel
        end = [voxel[i] + shape[i] for i in range(3)]
    return start, end


# --- Custom Classes ---
class BM4D:
    def __init__(self, sigma=10):
        self.sigma = sigma

    def __call__(self, noise):
        mn, mx = np.percentile(noise, 5), np.percentile(noise, 99.9)
        denoised = bm4d(noise, self.sigma)
        return (noise - mn) / mx, (denoised - mn) / mx, (mn, mx)


# --- Coordinate Conversions ---
def to_physical(voxel, anisotropy):
    """
    Converts the given coordinate from voxels to physical space.

    Parameters
    ----------
    voxel : ArrayLike
        Voxel coordinate to be converted.
    multiscale
        Level in the image pyramid that the voxel coordinate must index into.


    Returns
    -------
    Tuple[int]
        Physical coordinate of "voxel".

    """
    voxel = voxel[::-1]
    return tuple([voxel[i] * anisotropy[i] for i in range(3)])


def to_voxels(xyz, anisotropy):
    """
    Converts the given coordinate from physical to voxel space.

    Parameters
    ----------
    xyz : ArrayLike
        Physical coordinate to be converted to a voxel coordinate.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.

    Returns
    -------
    numpy.ndarray
        Voxel coordinate.

    """
    voxel = xyz / np.array(anisotropy)
    return np.round(voxel).astype(int)[::-1]


def local_to_physical(local_voxel, offset, multiscale):
    """
    Converts a local voxel coordinate to a physical coordinate in global
    space.

    Parameters
    ----------
    local_voxel : Tuple[int]
        Local voxel coordinate in an image patch.
    offset : Tuple[int]
        Offset from the local coordinate system to the global coordinate
        system.
    multiscale : int
        Level in the image pyramid that the voxel coordinate must index into.

    Returns
    -------
    numpy.ndarray
        Physical coordinate.

    """
    global_voxel = np.array([v + o for v, o in zip(local_voxel, offset)])
    return to_physical(global_voxel, multiscale)


# --- Visualizations ---
def plot_mips(img, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image to generate MIPs from.

    Returns
    -------
    None

    """
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        mip = np.max(img, axis=i)
        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()


# --- Helpers ---
def convert_tiff_ome_zarr(
        in_path,
        out_path,
        chunks: tuple = (1, 1, 64, 128, 128),
        compressor: Any = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE),
        voxel_size: list = (0.748, 0.748, 1.0),
        n_levels: int = 3
):
    """
    Convert a Tiff stack to an N5 dataset.

    Args:
         in_path: the path to the Tiff
         out_path: the path to write the N5
         chunks: the chunk shape of the N5 dataset
         compressor: the numcodecs compressor instance for the N5 dataset
         voxel_size: the voxel spacing of the image, in nanometers
         n_levels: the number of levels in the multiscale pyramid
    """
    im = tifffile.imread(in_path)
    while im.ndim < 5:
        im = im[np.newaxis, ...]
    pyramid = multiscale(im, windowed_mode, scale_factors=[1, 1, 2, 2, 2])[:n_levels]
    pyramid = [l.data for l in pyramid]
    z = zarr.open(store=zarr.DirectoryStore(out_path, dimension_separator='/'), mode='w')
    voxel_size = np.array([1, 1] + list(reversed(voxel_size)))
    scales = [np.concatenate((voxel_size[:2], voxel_size[2:] * 2 ** i)) for i in range(n_levels)]
    coordinate_transformations = [[{"type": "scale", "scale": scale.tolist()}] for scale in scales]
    storage_options = {"compressor": compressor}
    write_multiscale(
        pyramid=pyramid,
        group=z,
        chunks=chunks,
        axes=[{"name": 't', "type": "time", "unit": "millisecond"},
              {"name": 'c', "type": "channel"},
              {"name": 'z', "type": "space", "unit": "micrometer"},
              {"name": 'y', "type": "space", "unit": "micrometer"},
              {"name": 'x', "type": "space", "unit": "micrometer"}],
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options
    )


def compute_cratio(img, codec):
    """
    Computes the compression ratio for a given image.

    Parameters
    ----------
    img : numpy.ndarray
        Image to compute compression ratio of.
    codec : ...
        Object used to compress image.

    Returns
    -------
    float
        Compression ratio for a given image.

    """
    return img.nbytes / len(codec.encode(img))


def fill_boundary(img, depth, value):
    """
    Fill boundary of a 3D image with a given value.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be updated
    depth : int
        Distance to boundary from boundary to be filled.
    value : float
        Fill value.

    Returns
    -------
    numpy.ndarray
        Updated image.

    """
    # Fill along axis 0 (z)
    img[:depth, :, :] = value
    img[-depth:, :, :] = value

    # Fill along axis 1 (y)
    img[:, :depth, :] = value
    img[:, -depth:, :] = value

    # Fill along axis 2 (x)
    img[:, :, :depth] = value
    img[:, :, -depth:] = value
    return img


def get_nbs(voxel, shape):
    """
    Gets the neighbors of a given voxel in a 3D grid with respect to
    26-connectivity.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate for which neighbors are to be found.
    shape : Tuple[int]
        Shape of the 3D grid. This is used to ensure that neighbors are
        within the grid boundaries.

    Returns
    -------
    List[Tuple[int]]
        Voxel coordinates of the neighboring voxels.

    """
    x, y, z = voxel
    nbs = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                # Skip the given voxel
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                # Add neighbor
                nb = (x + dx, y + dy, z + dz)
                if is_inbounds(nb, shape):
                    nbs.append(nb)
    return nbs


def is_inbounds(voxel, shape):
    """
    Checks if a given voxel is within the bounds of a 3D grid.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate to be checked.
    shape : Tuple[int]
        Shape of the 3D grid.

    Returns
    -------
    bool
        Indication of whether the given voxel is within the bounds of the
        grid.

    """
    x, y, z = voxel
    height, width, depth = shape
    if 0 <= x < height and 0 <= y < width and 0 <= z < depth:
        return True
    else:
        return False


def normalize(img_patch):
    """
    Rescales the given image to [0, 1] intensity range.

    Parameters
    ----------
    img_patch : numpy.ndarray
        Image patch to be normalized.

    Returns
    -------
    numpy.ndarray
        Normalized image.

    """
    img_patch -= np.min(img_patch)
    return img_patch / np.max(img_patch)
