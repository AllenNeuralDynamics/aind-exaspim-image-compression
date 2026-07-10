"""
Created on Fri Nov 22 12:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from cloudvolume import CloudVolume
from concurrent.futures import ThreadPoolExecutor
from imagecodecs.numcodecs import Jpegxl
from itertools import product
from matplotlib.colors import ListedColormap
from numcodecs import register_codec
from ome_zarr.writer import write_multiscale
from scipy.ndimage import uniform_filter
from xarray_multiscale import multiscale
from xarray_multiscale.reducers import windowed_mode

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts
import tifffile
import zarr

from aind_exaspim_image_compression.utils import util


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
    img : ArrayLike
        Image volume.
    """
    # Read image
    if ".n5" in img_path:
        img = _read_n5(img_path)
    elif ".tif" in img_path or ".tiff" in img_path:
        img = _read_tiff(img_path)
    elif ".zarr" in img_path:
        img = _read_zarr(img_path)
    elif is_neuroglancer_precomputed(img_path):
        img = _read_neuroglancer_precompted(img_path)
    else:
        raise ValueError(f"Unsupported image format: {img_path}")

    # Ensure shape is (1, 1, h, w, d)
    while img.ndim < 5:
        img = img[np.newaxis, ...]
    return img


def _read_n5(img_path):
    """
    Reads the "volume" dataset of an N5 container from local disk, GCS, or S3.

    zarr v3 dropped built-in N5 support, so this reads through tensorstore (the
    same backend used for neuroglancer volumes). NOTE: migrated for zarr>=3
    compatibility but not exercised against live N5 data -- no current dataset
    uses N5.

    Parameters
    ----------
    img_path : str
        Path to N5 directory.

    Returns
    -------
    tensorstore.TensorStore
        Lazy, sliceable image volume.
    """
    if is_s3_path(img_path) or is_gcs_path(img_path):
        bucket, prefix = util.parse_cloud_path(img_path)
        kvstore = {
            "driver": get_storage_driver(img_path),
            "bucket": bucket,
            "path": prefix.rstrip("/") + "/volume/",
        }
    else:
        kvstore = {"driver": "file", "path": img_path.rstrip("/") + "/volume"}
    arr = ts.open({"driver": "n5", "kvstore": kvstore}).result()
    return arr


def _read_neuroglancer_precompted(img_path):
    # Extract metadata
    bucket, path = util.parse_cloud_path(img_path)
    driver = get_storage_driver(img_path)

    # Read image
    img = ts.open(
        {
            "driver": "neuroglancer_precomputed",
            "kvstore": {
                "driver": driver,
                "bucket": bucket,
                "path": path,
                },
            "context": {
                "cache_pool": {"total_bytes_limit": 0},
                "cache_pool#remote": {"total_bytes_limit": 0},
                "data_copy_concurrency": {"limit": 8},
            },
            }
    ).result()

    # Check whether to permute axes
    if bucket == "allen-nd-goog":
        img = img[ts.d["channel"][0]]
        img = img[ts.d[0].transpose[2]]
        img = img[ts.d[0].transpose[1]]
    return img[:].read().result()


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
    numpy.ndarray
        Image volume.
    """
    if is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(**(storage_options or {}))
        with fs.open(img_path, "rb") as f:
            return tifffile.imread(f)
    else:
        return tifffile.imread(img_path)


def _read_zarr(img_path):
    """
    Reads a Zarr volume from local disk, GCS, or S3.

    Parameters
    ----------
    img_path : str
        Path to Zarr directory.

    Returns
    -------
    zarr.Array
        Image volume.
    """
    register_codec(Jpegxl)
    # zarr v3 builds the store from the path: a LocalStore for a filesystem
    # path, an FsspecStore for s3:// / gs://. The S3 buckets read here are
    # public (read anonymously); GCS uses the default credential chain.
    storage_options = {"anon": True} if is_s3_path(img_path) else None
    return zarr.open(img_path, mode="r", storage_options=storage_options)


def get_ome_zarr_level_transform(img_path):
    """Read the coordinate transform for an OME-Zarr array level.

    Parameters
    ----------
    img_path : str
        Path to a level array, for example ``/data/image.ome.zarr/0``.

    Returns
    -------
    dict
        Five-dimensional ``scale`` and ``translation`` tuples in axis order,
        plus the common ``spatial_unit`` declared by the spatial axes.
    """
    level_path = img_path.rstrip("/")
    if "/" not in level_path:
        raise ValueError(f"Expected a Zarr level path, got: {img_path}")
    group_path, dataset_path = level_path.rsplit("/", 1)
    storage_options = {"anon": True} if is_s3_path(img_path) else None
    group = zarr.open_group(
        group_path, mode="r", storage_options=storage_options
    )

    ome = group.attrs.get("ome", {})
    multiscales = group.attrs.get("multiscales") or ome.get("multiscales")
    if not multiscales:
        raise ValueError(f"No OME multiscales metadata found at {group_path}")

    for multiscale_metadata in multiscales:
        datasets = multiscale_metadata.get("datasets", [])
        dataset = next(
            (item for item in datasets if item.get("path") == dataset_path),
            None,
        )
        if dataset is None:
            continue

        axes = multiscale_metadata.get("axes", [])
        if [axis.get("name") for axis in axes] != ["t", "c", "z", "y", "x"]:
            raise ValueError("Expected OME-Zarr axes in (t, c, z, y, x) order")

        spatial_units = {
            axis.get("unit") for axis in axes if axis.get("type") == "space"
        }
        if len(spatial_units) != 1 or None in spatial_units:
            raise ValueError("Expected one common unit for all spatial axes")

        scale = np.ones(5, dtype=float)
        translation = np.zeros(5, dtype=float)
        for transform in dataset.get("coordinateTransformations", []):
            if transform.get("type") == "scale":
                scale *= np.asarray(transform["scale"], dtype=float)
            elif transform.get("type") == "translation":
                translation += np.asarray(
                    transform["translation"], dtype=float
                )

        return {
            "scale": tuple(scale.tolist()),
            "translation": tuple(translation.tolist()),
            "spatial_unit": spatial_units.pop(),
        }

    raise ValueError(
        f"Dataset {dataset_path!r} is not listed in OME metadata at "
        f"{group_path}"
    )


def ome_zarr_coordinate_to_voxel(xyz, level_transform):
    """Convert Neuroglancer ``(x, y, z)`` coordinates to ``(z, y, x)``.

    Neuroglancer displays each coordinate in units of that dimension's scale,
    shown separately in its position toolbar. Therefore, the physical OME
    translation is normalized by the scale before it is subtracted. The
    returned indices identify the nearest voxel center in the source array.
    """
    coordinate_xyz = np.asarray(xyz, dtype=float)
    scale = np.asarray(level_transform["scale"], dtype=float)
    translation = np.asarray(level_transform["translation"], dtype=float)
    if coordinate_xyz.shape != (3,):
        raise ValueError("xyz must contain exactly three coordinates")
    if scale.shape != (5,) or translation.shape != (5,):
        raise ValueError("OME scale and translation must each have five values")
    if np.any(scale[2:] == 0):
        raise ValueError("OME spatial scale values must be nonzero")

    coordinate_zyx = coordinate_xyz[::-1]
    voxel_zyx = coordinate_zyx - translation[2:] / scale[2:]
    return tuple(np.rint(voxel_zyx).astype(int).tolist())


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
        Indicates whether the given voxel is the center or front-top-left
        corner of the patch to be extracted.

    Returns
    -------
    numpy.ndarray
        Patch extracted from the given image.
    """
    # Get patch coordinates
    assert len(img.shape) == 5, "Error: Image must have shape TxCxDxHxW!"
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
    anisotropy : Tuple[float]
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.

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
    anisotropy : Tuple[float]
        Image to physical coordinates scaling factors to account for the
        anisotropy of the microscope.

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


def compute_cratio_jpegxl(
    img, codec, patch_shape=(128, 128, 64), max_workers=32
):
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


# --- Visualization ---
def make_segmentation_colormap(mask, seed=42):
    """
    Creates a matplotlib ListedColormap for a segmentation mask. Ensures label
    0 maps to black and all other labels get distinct random colors.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask with integer labels. Assumes label 0 is background.
    seed : int, optional
        Random seed for color reproducibility. Default is 42.

    Returns
    -------
    ListedColormap
        Colormap with black for background and unique colors for other labels.
    """
    n_labels = int(mask.max()) + 1
    rng = np.random.default_rng(seed)
    colors = [(0, 0, 0)]
    colors += list(rng.uniform(0.2, 1.0, size=(n_labels - 1, 3)))

    return ListedColormap(colors)


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
    output_path : None or str, optional
        Path that plot is saved to if provided. Default is None.
    vmax : None or float, optional
        Brightness intensity used as upper limit of the colormap. Default is
        None.
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


def plot_segmentation_mips(mask):
    """
    Plots maximum intensity projections (MIPs) of a segmentation mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask. Can be either:
        - 3D array (Z, Y, X), or
        - 5D array (N, C, Z, Y, X), in which case the first sample
          and first channel are used.
    """
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    cmap = make_segmentation_colormap(mask)

    for i in range(3):
        if len(mask.shape) == 5:
            mip = np.max(mask[0, 0, ...], axis=i)
        else:
            mip = np.max(mask, axis=i)

        axs[i].imshow(mip, cmap=cmap, interpolation="none")
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_slices(img, output_path=None, vmax=None):
    """
    Plots the middle slice of a 3D image along the XY, XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Image to generate MIPs from.
    output_path : None or str, optional
        Path that plot is saved to if provided. Default is None.
    vmax : None or float, optional
        Brightness intensity used as upper limit of the colormap. Default is
        None.
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
def get_storage_driver(img_path):
    """
    Gets the storage driver needed to read the image.

    Parameters
    ----------
    img_path : str
        Image path to be checked.

    Returns
    -------
    str
        Storage driver needed to read the image.
    """
    if is_s3_path(img_path):
        return "s3"
    elif is_gcs_path(img_path):
        return "gcs"
    else:
        raise ValueError(f"Unsupported path type: {img_path}")


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


def is_gcs_path(path):
    """
    Checks if the path is a GCS path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is a GCS path.
    """
    return path.startswith("gs://")


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


def is_neuroglancer_precomputed(path):
    """
    Checks if the path points to a neuroglancer precomputed dataset.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path points to a neuroglancer precomputed
        dataset.
    """
    try:
        vol = CloudVolume(path)
        return all(k in vol.info for k in ["data_type", "scales", "type"])
    except Exception:
        return False


def is_s3_path(path):
    """
    Checks if the path is an S3 path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is an S3 path.
    """
    return path.startswith("s3://")


def write_ome_zarr(
    img,
    output_path,
    chunks=(1, 1, 64, 128, 128),
    compressor=None,
    n_levels=1,
    scale_factors=(1, 1, 2, 2, 2),
    voxel_size=(748, 748, 1000),
    scale=None,
    translation=None,
    spatial_unit="nanometer",
    storage_options=None,
):
    # Zarr v3 codec; default matches the cratio codec (zstd, level 5, shuffle).
    from zarr.codecs import BloscCodec

    if compressor is None:
        compressor = BloscCodec(
            cname="zstd", clevel=5, shuffle="shuffle"
        )

    # Ensure 5D image (T, C, Z, Y, X)
    while img.ndim < 5:
        img = img[np.newaxis, ...]

    # Generate multiscale pyramid
    pyramid = multiscale(
        img, windowed_mode, scale_factors=scale_factors
    )[:n_levels]
    pyramid = [level.data for level in pyramid]

    # Zarr v3 builds a LocalStore or FsspecStore from the path/URL.
    zgroup = zarr.open_group(
        store=output_path, mode="w", storage_options=storage_options
    )

    # Voxel size scaling for each level. ``voxel_size`` is in (x, y, z)
    # order; an explicit scale uses the stored (t, c, z, y, x) axis order.
    base_scale = np.asarray(
        scale if scale is not None else [1, 1, *reversed(voxel_size)],
        dtype=float,
    )
    if base_scale.shape != (5,):
        raise ValueError(
            "scale must have five values in (t, c, z, y, x) order"
        )
    base_translation = np.asarray(
        translation if translation is not None else np.zeros(5), dtype=float
    )
    if base_translation.shape != (5,):
        raise ValueError(
            "translation must have five values in (t, c, z, y, x) order"
        )
    level_factors = np.asarray(scale_factors, dtype=float)
    if level_factors.shape != (5,):
        raise ValueError("scale_factors must have one value for each axis")
    scales = [
        (base_scale * level_factors**i).tolist() for i in range(n_levels)
    ]
    coord_transforms = []
    for level_scale_values in scales:
        level_scale = np.asarray(level_scale_values)
        # Downsampling groups neighboring voxel centers, moving each coarser
        # level's first voxel center by half the increase in voxel size.
        level_translation = base_translation + (level_scale - base_scale) / 2
        coord_transforms.append(
            [
                {"type": "scale", "scale": level_scale.tolist()},
                {
                    "type": "translation",
                    "translation": level_translation.tolist(),
                },
            ]
        )

    # Write to OME-Zarr
    write_multiscale(
        pyramid=pyramid,
        group=zgroup,
        axes=[
            {"name": "t", "type": "time", "unit": "millisecond"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": spatial_unit},
            {"name": "y", "type": "space", "unit": spatial_unit},
            {"name": "x", "type": "space", "unit": spatial_unit},
        ],
        coordinate_transformations=coord_transforms,
        storage_options={
            "chunks": chunks,
            "compressors": [compressor],
        },
    )


def write_zarr(
    img,
    output_path,
    chunks=(1, 1, 64, 64, 64),
    cname="zstd",
    clevel=5,
    shuffle="shuffle",
    storage_options=None,
):
    """
    Writes an image volume to a single Zarr array (local or cloud).

    Uses the Zarr v3 API, so ``output_path`` may be a local path or a cloud URL
    (``s3://...``, ``gs://...``); Zarr builds the store from the URL. Cloud
    credentials are resolved from the standard chain unless ``storage_options``
    overrides them. The array is stored 5D (t, c, z, y, x) so ``read`` reads it
    back unchanged.

    Parameters
    ----------
    img : numpy.ndarray
        Image volume to write. Promoted to 5D if it has fewer dimensions.
    output_path : str
        Destination array path/URL (e.g. "s3://bucket/denoised.zarr").
    chunks : Tuple[int], optional
        Chunk shape. Default is (1, 1, 64, 64, 64), matching the cratio chunk.
    cname : str, optional
        Blosc compressor name. Default is "zstd".
    clevel : int, optional
        Blosc compression level. Default is 5.
    shuffle : str, optional
        Blosc shuffle mode ("shuffle", "bitshuffle", or "noshuffle"). Default
        is "shuffle".
    storage_options : dict or None, optional
        fsspec storage options for cloud stores (e.g. {"anon": True}). Default
        is None (default credential chain).
    """
    from zarr.codecs import BloscCodec

    while img.ndim < 5:
        img = img[np.newaxis, ...]
    z = zarr.create_array(
        store=output_path,
        shape=img.shape,
        chunks=chunks,
        dtype=img.dtype,
        compressors=BloscCodec(
            cname=cname, clevel=clevel, shuffle=shuffle
        ),
        overwrite=True,
        storage_options=storage_options,
    )
    z[:] = img


def ssim3D(img1, img2, data_range=None, window_size=16):
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

    # Integer powers and products overflow before scipy sees them (notably for
    # normal uint16 microscopy inputs), corrupting the local moments. Convert
    # once up front so all subsequent arithmetic is floating point.
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

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


def compute_mae(img1, img2):
    """
    Computes the mean absolute error between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        Image.
    img2 : numpy.ndarray
        Image with the same shape as "img1".

    Returns
    -------
    float
        Mean absolute error between the two images.
    """
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    return float(np.mean(np.abs(a - b)))


def compute_lmax(img1, img2):
    """
    Computes the maximum absolute error between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        Image.
    img2 : numpy.ndarray
        Image with the same shape as "img1".

    Returns
    -------
    float
        Maximum absolute error between the two images.
    """
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    return float(np.max(np.abs(a - b)))
