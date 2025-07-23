"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from bm4d import bm4d
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from numcodecs import Blosc
from ome_zarr.writer import write_multiscale
from scipy.ndimage import uniform_filter
from typing import Any
from xarray_multiscale import multiscale, windowed_mode

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import os
import s3fs
import tifffile
import zarr

from aind_exaspim_image_compression.utils import util


# --- Image Reader ---
def read(img_path):
    """
    Read an image volume from a supported path based on its extension.

    Supported formats:
    - Zarr ('.zarr') from local, GCS, or S3
    - N5 ('.n5') from local or GCS
    - TIFF ('.tif', '.tiff') from local or GCS

    Parameters
    ----------
    img_path : str
        Path to the image. Can be a local or cloud path (gs:// or s3://).

    Returns
    -------
    np.ndarray
        Loaded image volume as a NumPy array.
    """
    if ".zarr" in img_path:
        return _read_zarr(img_path)
    elif ".n5" in img_path:
        return _read_n5(img_path)
    elif ".tif" in img_path or ".tiff" in img_path:
        return _read_tiff(img_path)
    else:
        raise ValueError(f"Unsupported image format: {img_path}")


def _read_zarr(img_path):
    """
    Read a Zarr volume from local disk, GCS, or S3.

    Parameters
    ----------
    img_path : str
        Path to the Zarr directory.

    Returns
    -------
    np.ndarray
        Loaded image volume.
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.storage.FSStore(img_path, fs=fs)
    elif _is_s3_path(img_path):
        fs = s3fs.S3FileSystem(config_kwargs={"max_pool_connections": 50})
        store = s3fs.S3Map(root=img_path, s3=fs)
    else:
        store = zarr.DirectoryStore(img_path)
    return zarr.open(store, mode="r")


def _read_n5(img_path):
    """
    Read an N5 volume from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to the N5 directory.

    Returns
    -------
    np.ndarray
        N5 group volume stored at key "volume".
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.n5.N5FSStore(img_path, s=fs)
    elif _is_s3_path(img_path):
        fs = s3fs.S3FileSystem(config_kwargs={"max_pool_connections": 50})
        store = s3fs.S3Map(root=img_path, s3=fs)
    else:
        store = zarr.n5.N5Store(img_path)
    return zarr.open(store, mode="r")["volume"]


def _read_tiff(img_path, storage_options=None):
    """
    Read a TIFF file from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to the TIFF file.
    storage_options : dict, optional
        Additional kwargs for GCSFileSystem.

    Returns
    -------
    np.ndarray
        Image data from the TIFF file.
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(**(storage_options or {}))
        with fs.open(img_path, "rb") as f:
            return tifffile.imread(f)
    else:
        return tifffile.imread(img_path)


def _is_gcs_path(path):
    """
    Check if the path is a GCS path.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
    """
    return path.startswith("gs://")


def _is_s3_path(path):
    """
    Check if the path is an S3 path.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
    """
    return path.startswith("s3://")


# --- Read Patches ---
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
        the patch or the front-top-left corner. Default is True.

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


# --- Compression utils ---
class BM4D:
    """
    A simple wrapper for BM4D denoising of 3D volumetric data.
    """
    def __init__(self, sigma=60):
        """
        Initialize the BM4D denoiser.

        Parameters
        ----------
        sigma : float, optional
            Noise standard deviation used for denoising.

        Returns
        -------
        None
        """
        self.sigma = sigma

    def __call__(self, noise):
        """
        Apply BM4D denoising to the input volume.

        Parameters
        ----------
        noise : np.ndarray
            A 3D NumPy array representing the noisy input volume.

        Returns
        -------
        tuple
            A 3-tuple containing:
            - normalized input volume (np.ndarray)
            - normalized denoised volume (np.ndarray)
            - normalization parameters (tuple): (mn, mx)
        """
        mn, mx = np.percentile(noise, 5), np.percentile(noise, 99.9)
        mx = max(1, mx)
        denoised = bm4d(noise, self.sigma)
        return (noise - mn) / mx, (denoised - mn) / mx, (mn, mx)


def compute_cratio(img, codec, patch_shape=(64, 64, 64)):
    """
    Computes a Zarr-style chunked compression ratio for a given image.

    Parameters
    ----------
    img : np.ndarray
        Image to compute compression ratio of.
    codec : blosc.Blosc
        Blosc codec used to compress each chunk.
    patch_shape : Tuple[int]
        Shape of chunks Zarr would use. Default is (64, 64, 64).

    Returns
    -------
    float
        Compression ratio = total uncompressed size / total compressed size.
    """
    # Check image
    if len(img.shape) == 5:
        img = np.ascontiguousarray(img[0, 0], dtype=np.uint16)
    else:
        img = np.ascontiguousarray(img, dtype=np.uint16)

    # Compute chunked cratio
    total_compressed_size = 0
    total_uncompressed_size = 0
    z = [range(0, s, c) for s, c in zip(img.shape, patch_shape)]
    for z0 in z[0]:
        for z1 in z[1]:
            for z2 in z[2] if len(z) > 2 else [0]:
                slice_ = img[
                    z0: z0 + patch_shape[0],
                    z1: z1 + patch_shape[1],
                    z2: z2 + patch_shape[2] if len(z) > 2 else slice(None),
                ]
                chunk = np.ascontiguousarray(slice_)
                compressed = codec.encode(chunk)
                total_compressed_size += len(compressed)
                total_uncompressed_size += chunk.nbytes
    return round(total_uncompressed_size / total_compressed_size, 2)


def compute_cratio_jpegxl(img, codec, patch_shape=(128, 128, 64), max_workers=32):
    # Helper routine
    def compress_patch(idx):
        iterator = zip(idx, patch_shape, img.shape)
        slices = tuple(slice(i, min(i + c, s)) for i, c, s in iterator)
        patch = img[slices]
        compressed_size = 0
        for k in range(patch.shape[-1]):
            slice2d = np.ascontiguousarray(patch[..., k])
            encoded = codec.encode(slice2d)
            compressed_size += len(encoded)
        return patch.nbytes, compressed_size

    # Generate chunk start indices
    img = np.ascontiguousarray(img)
    chunk_ranges = [range(0, s, c) for s, c in zip(img.shape, patch_shape)]
    chunk_coords = list(product(*chunk_ranges))

    # Compute chunked cratio
    total_uncompressed = 0
    total_compressed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ubytes, cbytes in pool.map(compress_patch, chunk_coords):
            total_uncompressed += ubytes
            total_compressed += cbytes
    return round(total_uncompressed / total_compressed, 2)


def compress_and_decompress_jpeg(
    img, codec, patch_shape=(128, 128, 64), max_workers=32
):
    # Helper routine
    def process_patch(idx):
        iterator = zip(idx, patch_shape, img.shape)
        slices = tuple(slice(i, min(i + c, s)) for i, c, s in iterator)
        patch = img[slices]

        compressed_size = 0
        decompressed_slices = []
        for k in range(patch.shape[-1]):
            slice2d = np.ascontiguousarray(patch[..., k])
            encoded = codec.encode(slice2d)
            compressed_size += len(encoded)

            decoded = codec.decode(encoded)
            decompressed_slices.append(decoded)

        decompressed_patch = np.stack(decompressed_slices, axis=-1)
        return slices, patch.nbytes, compressed_size, decompressed_patch

    # Generate chunk start indices
    img = np.ascontiguousarray(img)
    chunk_ranges = [range(0, s, c) for s, c in zip(img.shape, patch_shape)]
    chunk_coords = list(product(*chunk_ranges))

    # Compute chunked ratio
    decompressed_img = np.empty_like(img)
    total_uncompressed = 0
    total_compressed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        iterator = pool.map(process_patch, chunk_coords)
        for slices, ubytes, cbytes, decompressed_patch in iterator:
            decompressed_img[slices] = decompressed_patch
            total_uncompressed += ubytes
            total_compressed += cbytes

    cratio = round(total_uncompressed / total_compressed, 2)
    return decompressed_img, cratio


# --- Image Prefix Search ---
def get_img_prefix(brain_id, img_prefix_path=None):
    # Check prefix path
    if img_prefix_path:
        prefix_lookup = util.read_json(img_prefix_path)
        if brain_id in prefix_lookup:
            return prefix_lookup[brain_id]

    # Search for prefix path
    result = find_img_prefix(brain_id)
    if len(result) == 1:
        prefix = result[0] + "/"
        if img_prefix_path:
            prefix_lookup[brain_id] = prefix
            util.write_json(img_prefix_path, prefix_lookup)
        return prefix

    raise Exception(f"Image Prefixes Found for {brain_id}- {result}")


def find_img_prefix(brain_id):
    # Initializations
    bucket_name = "aind-open-data"
    prefixes = util.list_s3_bucket_prefixes(
        "aind-open-data", keyword="exaspim"
    )

    # Get possible prefixes
    valid_prefixes = list()
    for prefix in prefixes:
        # Check for new naming convention
        if util.exists_in_prefix(bucket_name, prefix, "fusion"):
            prefix = os.path.join(prefix, "fusion")

        # Check if prefix is valid
        if is_valid_prefix(bucket_name, prefix, brain_id):
            valid_prefixes.append(
                os.path.join("s3://aind-open-data", prefix, "fused.zarr")
            )
    return find_functional_img_prefix(valid_prefixes)


def is_valid_prefix(bucket_name, prefix, brain_id):
    # Quick checks
    is_test = "test" in prefix.lower()
    has_correct_id = str(brain_id) in prefix
    if not has_correct_id or is_test:
        return False

    # Check inside prefix - old convention
    if util.exists_in_prefix(bucket_name, prefix, "fused.zarr"):
        img_prefix = os.path.join(prefix, "fused.zarr")
        multiscales = util.list_s3_prefixes(bucket_name, img_prefix)
        multiscales = [s.split("/")[-2] for s in multiscales]
        for s in map(str, range(0, 8)):
            if s not in multiscales:
                return False
    return True


def find_functional_img_prefix(prefixes):
    # Filter img prefixes that fail to open
    functional_prefixes = list()
    for prefix in prefixes:
        try:
            root = os.path.join(prefix, str(0))
            store = s3fs.S3Map(root=root, s3=s3fs.S3FileSystem(anon=True))
            img = zarr.open(store, mode="r")
            if np.max(img.shape) > 25000:
                functional_prefixes.append(prefix)
        except:
            pass
    return functional_prefixes


# --- Helpers ---
def init_ome_zarr(
    img,
    output_path,
    chunks=(1, 1, 64, 128, 128),
    compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
):
    # Setup output store
    store = zarr.DirectoryStore(output_path, dimension_separator="/")
    zgroup = zarr.group(store=store)

    # Create top-level dataset
    output_zarr = zgroup.create_dataset(
        name=0,
        shape=img.shape,
        chunks=chunks,
        dtype=np.uint16,
        compressor=compressor,
        overwrite=True
    )
    return output_zarr


def write_ome_zarr(
    img,
    output_path,
    chunks=(1, 1, 64, 128, 128),
    compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
    n_levels=1,
    scale_factors=(1, 1, 2, 2, 2),
    voxel_size=(748, 748, 1000),
):
    # Ensure 5D image (T, C, Z, Y, X)
    while img.ndim < 5:
        img = img[np.newaxis, ...]

    # Generate multiscale pyramid
    pyramid = multiscale(img, windowed_mode, scale_factors=scale_factors)[:n_levels]
    pyramid = [level.data for level in pyramid]

    # Prepare Zarr store
    store = zarr.DirectoryStore(output_path, dimension_separator="/")
    zgroup = zarr.open(store=store, mode="w")

    # Voxel size scaling for each level
    base_scale = np.array([1, 1, *reversed(voxel_size)])
    scales = [base_scale[:2].tolist() + (base_scale[2:] * 2**i).tolist() for i in range(n_levels)]
    coord_transforms = [[{"type": "scale", "scale": s}] for s in scales]

    # Write to OME-Zarr
    write_multiscale(
        pyramid=pyramid,
        group=zgroup,
        chunks=chunks,
        axes=[
            {"name": "t", "type": "time", "unit": "millisecond"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        coordinate_transformations=coord_transforms,
        storage_options={"compressor": compressor},
    )


def compute_mae(img1, img2):
    """
    Computes the mean absolute difference between two 3D images.

    Parameters
    ----------
    img1 : numpy.ndarray
        3D Image.
    img2 : numpy.ndarray
        3D Image.

    Returns
    -------
    float
        Mean absolute difference between two 3D images.
    """
    return np.mean(abs(img1 - img2))


def compute_lmax(img1, img2, p=99.99):
    """
    Computes the stable l-inf norm between two 3D images.

    Parameters
    ----------
    img1 : numpy.ndarray
        3D Image.
    img2 : numpy.ndarray
        3D Image.
    p : float, optional
        Percentile used to compute stable l-inf norm. Default is 99.99.

    Returns
    -------
    float
        Stable l-inf norm between two 3D images.
    """
    return np.percentile(abs(img1 - img2), p)


def compute_ssim3D(img1, img2, data_range=None, window_size=16):
    """
    Computes the structural similarity (SSIM) between two 3D images.

    Parameters
    ----------
    img1 : numpy.ndarray
        3D Image.
    img2 : numpy.ndarray
        3D Image.
    data_range : float, optional
        Value range of input images. If None, computed from "img1". Default
        is None.
    window_size : int, optional
        Size of the 3D filter window. Default is 16.

    Returns
    -------
    float
        SSIM between the two input images.
    """
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")

    if data_range is None:
        data_range1 = np.max(img1) - np.min(img1)
        data_range2 = np.max(img2) - np.min(img2)
        data_range = max(data_range1, data_range2)

    # Mean filter
    mu1 = uniform_filter(img1, window_size)
    mu2 = uniform_filter(img2, window_size)

    # Variance and covariance
    sigma1_sq = uniform_filter(img1**2, window_size) - mu1**2
    sigma2_sq = uniform_filter(img2**2, window_size) - mu2**2
    sigma12 = uniform_filter(img1 * img2, window_size) - mu1 * mu2

    # SSIM map
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (np.maximum(denominator, 1e-8) + 1e-6)
    return np.mean(ssim_map)


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


def plot_mips(img, output_path=None, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input image to generate MIPs from.

    Returns
    -------
    None
    """
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        if len(img.shape) == 5:
            mip = np.max(img[0, 0, ...], axis=i)
        else:
            mip = np.max(img, axis=i)

        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    plt.show()
    plt.close(fig)
