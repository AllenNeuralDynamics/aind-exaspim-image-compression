"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from concurrent.futures import ThreadPoolExecutor
from imagecodecs.numcodecs import Jpegxl
from itertools import product
from numcodecs import Blosc, register_codec
from ome_zarr.writer import write_multiscale
from scipy.ndimage import uniform_filter
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mode

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import tifffile
import zarr


# --- Readers ---
def read(img_path):
    """
    Reads an image volume from a supported path based on its extension.

    Supported formats:
        - Zarr ('.zarr') from local, GCS, or S3
        - N5 ('.n5') from local or GCS
        - TIFF ('.tif', '.tiff') from local or GCS

    Parameters
    ----------
    img_path : str
        Path to image. Can be a local or cloud path (gs:// or s3://).

    Returns
    -------
    ArrayLike
        Image volume.
    """
    # Read image
    if ".zarr" in img_path:
        img = _read_zarr(img_path)
    elif ".n5" in img_path:
        img = _read_n5(img_path)
    elif ".tif" in img_path or ".tiff" in img_path:
        img = _read_tiff(img_path)
    else:
        raise ValueError(f"Unsupported image format: {img_path}")

    # Ensure shape is (1, 1, h, w, d)
    while img.ndim < 5:
        img = img[np.newaxis, ...]
    return img


def _read_zarr(img_path):
    """
    Reads a Zarr volume from local disk, GCS, or S3.

    Parameters
    ----------
    img_path : str
        Path to Zarr directory.

    Returns
    -------
    zarr.ndarray
        Image volume.
    """
    register_codec(Jpegxl)
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.storage.FSStore(img_path, fs=fs)
    elif _is_s3_path(img_path):
        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=img_path, s3=fs)
    else:
        store = zarr.DirectoryStore(img_path)
    return zarr.open(store, mode="r")


def _read_n5(img_path):
    """
    Reads an N5 volume from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to N5 directory.

    Returns
    -------
    zarr.core.Array
        Image volume.
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
    Reads a TIFF file from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to TIFF file.
    storage_options : dict, optional
        Additional kwargs for GCSFileSystem.

    Returns
    -------
    np.ndarray
        Image volume.
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(**(storage_options or {}))
        with fs.open(img_path, "rb") as f:
            return tifffile.imread(f)
    else:
        return tifffile.imread(img_path)


def _is_gcs_path(path):
    """
    Checks if the path is a GCS path.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
        Indication of whether the path is a GCS path.
    """
    return path.startswith("gs://")


def _is_s3_path(path):
    """
    Checks if the path is an S3 path.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
        Indication of whether the path is an S3 path.
    """
    return path.startswith("s3://")


# --- Read Patches ---
def get_patch(img, voxel, shape, is_center=True):
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
    is_center : bool, optional
        Indicates whether the given voxel is the center or top, left, front
        corner of the patch to be extracted.

    Returns
    -------
    numpy.ndarray
        Patch extracted from the given image.
    """
    # Get patch coordinates
    start, end = get_start_end(voxel, shape, is_center=is_center)
    valid_start = any([s >= 0 for s in start])
    valid_end = any([e < img.shape[i + 2] for i, e in enumerate(end)])

    # Read patch
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
        3D voxel coordinates that represent the front-top-left corner.
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


def get_start_end(voxel, shape, is_center=True):
    """
    Gets the start and end indices of the image patch to be read.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate that specifies either the center or front-top-left
        corner of the patch to be read.
    shape : Tuple[int]
        Shape of the image patch to be read.
    is_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the patch or the front-top-left corner. Default is True.

    Return
    ------
    Tuple[List[int]]
        Start and end indices of the image patch to be read.
    """
    start = [v - d // 2 for v, d in zip(voxel, shape)] if is_center else voxel
    end = [voxel[i] + shape[i] // 2 for i in range(3)]
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
def compute_cratio(img, codec, patch_shape=(64, 64, 64)):
    """
    Computes a Zarr-style chunked compression ratio for a given image.

    Parameters
    ----------
    img : np.ndarray
        Image to compute the compression ratio of.
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
                    z2: z2 + patch_shape[2],
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
    img, codec, patch_shape=(32, 256, 256), max_workers=32
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
    img_decompressed = np.empty_like(img)
    total_uncompressed = 0
    total_compressed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        iterator = pool.map(process_patch, chunk_coords)
        for slices, ubytes, cbytes, decompressed_patch in iterator:
            img_decompressed[slices] = decompressed_patch
            total_uncompressed += ubytes
            total_compressed += cbytes

    cratio = round(total_uncompressed / total_compressed, 2)
    return img_decompressed, cratio


# --- Plotting ---
def plot_histogram(img, bins=256, max_value=np.inf, output_path=None):
    """
    Plots a histogram of voxel intensities for a 3D image.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image array.
    bins : int, optional
        Number of histogram bins. Default is 256.
    max_value : float, optional
        Threshold for filtering image intensities in the histogram. Default is
        np.inf.
    output_path : str, optional
        If provided, saves the histogram figure. Default is None.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(img[img < max_value].ravel(), bins=bins, alpha=0.7)
    plt.title("Intensity Histogram", fontsize=14)
    plt.xlabel("Intensity")
    plt.ylabel("Log Frequency")
    plt.yscale("log")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


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


def plot_slices(img, output_path=None, vmax=None):
    """
    Plots the middle slice of a 3D image along the XY, XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Image to generate MIPs from.
    """
    # Get middle slice
    shape = img.shape[2:] if len(img.shape) == 5 else img.shape
    zc, yc, xc = (s // 2 for s in shape)
    slices = [
        img[zc, :, :],   # XY plane
        img[:, yc, :],   # XZ plane
        img[:, :, xc]    # YZ plane
    ]

    # Plot
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    for i in range(3):
        axs[i].imshow(slices[i], vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)

    plt.show()
    plt.close(fig)


# --- Helpers ---
def get_slices(center, shape):
    """
    Gets the start and end indices of the chunk to be read.

    Parameters
    ----------
    center : tuple
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    start = [c - d // 2 for c, d in zip(center, shape)]
    return tuple(slice(s, s + d) for s, d in zip(start, shape))


def init_ome_zarr(
    output_path,
    shape,
    chunks=(1, 1, 64, 128, 128),
    compressor=Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE),
):
    """
    Initializes an OME-Zarr dataset on disk for a given image.

    Parameters
    ----------
    output_path : str or Path
        Path to the directory where the OME-Zarr dataset will be created.
    shape : Tuple[int]
        Shape of OME-Zarr dataset.
    chunks : Tuple[int], optional
        Chunk sizes for the dataset. Default is (1, 1, 64, 128, 128).
    compressor : numcodecs.Blosc, optional
        Compression codec used for chunk compression. Default is Blosc with
        "zstd" compression, compression level 5, and shuffle enabled.

    Returns
    -------
    zarr.core.Array
        Zarr dataset object corresponding to the initialized OME-Zarr dataset.
    """
    # Setup output store
    store = zarr.DirectoryStore(output_path, dimension_separator="/")
    zgroup = zarr.group(store=store)

    # Create top-level dataset
    print("Creating OME-ZARR Image with Shape:", shape)
    output_zarr = zgroup.create_dataset(
        name=0,
        shape=shape,
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
