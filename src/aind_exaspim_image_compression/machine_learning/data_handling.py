"""
Created on Jan 3 12:30:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training and inference.

"""

from aind_exaspim_dataset_utils.s3_util import get_img_prefix
from bm4d import bm4d
from collections.abc import Iterable
from concurrent.futures import (
    as_completed, ProcessPoolExecutor, ThreadPoolExecutor,
)
from copy import deepcopy
from torch.utils.data import Dataset
from tqdm import tqdm

import logging
import fastremap
import math
import numpy as np
import os
import queue
import random
import tensorstore as ts
import threading
import torch

from aind_exaspim_image_compression.machine_learning.metrics import (
    make_foreground_mask,
    make_segmentation_mask,
    make_skeleton_mask,
    patch_has_incoherent_segment,
)
from aind_exaspim_image_compression.machine_learning.transforms import (
    build_transform,
    calibrate_transform,
)
from aind_exaspim_image_compression.utils import img_util, util


logger = logging.getLogger(__name__)


def build_training_example(
    transform, preserve_foreground, raw, teacher, fg_mask
):
    """
    Assembles a training example from count-space arrays.

    Applies the foreground-preserving target construction and the transform,
    shared by the live TrainDataset and the CachedPatchDataset.

    Parameters
    ----------
    transform : IntensityTransform
        Transform mapping counts to the normalized domain.
    preserve_foreground : bool
        Whether the target keeps raw counts on the foreground.
    raw : numpy.ndarray
        Offset-subtracted raw counts.
    teacher : numpy.ndarray
        Clipped BM4D denoising in counts.
    fg_mask : numpy.ndarray
        Foreground mask.

    Returns
    -------
    Tuple[numpy.ndarray]
        (x, y, fg_mask) — model input, target, and mask (float 0/1).
    """
    fg = np.asarray(fg_mask).astype(bool)
    target = np.where(fg, raw, teacher) if preserve_foreground else teacher
    return (
        transform.forward(raw),
        transform.forward(target),
        fg.astype(np.float32),
    )


from aind_exaspim_image_compression.utils.swc_util import Reader


# --- Custom Datasets ---
class TrainDataset(Dataset):
    """
    A PyTorch Dataset for sampling 3D patches from whole-brain images and
    applying the BM4D denoising algorithm. The dataset's __getitem__ method
    returns both the original and denoised patches. Optionally, the patch
    sampling maybe biased toward foreground regions.

    Attributes
    ----------
    """

    def __init__(
        self,
        patch_shape,
        anisotropy=(0.748, 0.748, 1.0),
        boundary_buffer=5000,
        foreground_sampling_rate=0.5,
        min_foreground_voxels=50,
        min_segmentation_volume=200,
        n_examples_per_epoch=300,
        offsets=None,
        prefetch_foreground_sampling=16,
        preserve_foreground=True,
        reject_incoherent_patches=False,
        coherence_min_autocorr=0.4,
        coherence_max_highfreq_frac=0.35,
        coherence_min_segment_voxels=50,
        coherence_smooth_sigma=1.0,
        coherence_lag=2,
        max_resample_attempts=50,
        segmentation_dilate=0,
        sigma_bm4d=16,
        skeleton_radius=2,
        transform=None,
    ):
        # Call parent class
        super(TrainDataset, self).__init__()

        # Class attributes
        self.anisotropy = anisotropy
        self.boundary_buffer = boundary_buffer
        self.foreground_sampling_rate = foreground_sampling_rate
        self.min_foreground_voxels = min_foreground_voxels
        self.min_segmentation_volume = min_segmentation_volume
        self.n_examples_per_epoch = n_examples_per_epoch
        self.offsets = offsets or dict()
        self.patch_shape = patch_shape
        self.preserve_foreground = preserve_foreground
        self.prefetch_foreground_sampling = prefetch_foreground_sampling
        self.reject_incoherent_patches = reject_incoherent_patches
        self.coherence_min_autocorr = coherence_min_autocorr
        self.coherence_max_highfreq_frac = coherence_max_highfreq_frac
        self.coherence_min_segment_voxels = coherence_min_segment_voxels
        self.coherence_smooth_sigma = coherence_smooth_sigma
        self.coherence_lag = coherence_lag
        self.max_resample_attempts = max_resample_attempts
        self.segmentation_dilate = segmentation_dilate
        self.sigma_bm4d = sigma_bm4d
        self.skeleton_radius = skeleton_radius
        self.transform = transform or build_transform({"kind": "asinh"})
        self.swc_reader = Reader()

        # Data structures
        self.segmentations = dict()
        self.skeletons = dict()
        self.imgs = dict()

    # --- Ingest data ---
    def ingest_brain(self, brain_id, img_path, segmentation_path, swc_pointer):
        """
        Loads a brain image, label mask, and skeletons, then stores each in
        internal dictionaries.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the brain corresponding to the image.
        img_path : str or Path
            Path to whole-brain image to be read.
        segmentation_path : str
            Path to segmentation.
        swc_path : str
            Path to SWC files.
        """
        self.imgs[brain_id] = img_util.read(img_path)
        self._load_segmentation(brain_id, segmentation_path)
        self._load_swcs(brain_id, swc_pointer)

    def _load_segmentation(self, brain_id, segmentation_path):
        """
        Reads a segmentation mask generated by Google Applied Sciences (GAS).

        Parameters
        ----------
        brain_id : str
            Unique identifier for the brain corresponding to the given path.
        segmentation_path : str
            Path to segmentation.
        """
        if segmentation_path:
            # Parse into (bucket, key), tolerating a full gs://bucket/key URL
            # or a bucket-relative key. The kvstore path must not include the
            # gs://bucket/ prefix, or tensorstore doubles it.
            bucket, key = self._parse_gcs_path(segmentation_path)

            # Load image
            label_mask = ts.open(
                {
                    "driver": "neuroglancer_precomputed",
                    "kvstore": {
                        "driver": "gcs",
                        "bucket": bucket,
                        "path": key,
                    },
                    "context": {
                        "cache_pool": {"total_bytes_limit": 0},
                        "cache_pool#remote": {"total_bytes_limit": 0},
                        "data_copy_concurrency": {"limit": 8},
                    },
                }
            ).result()

            # Permute axes to be consistent with raw image.
            label_mask = label_mask[ts.d["channel"][0]]
            label_mask = label_mask[ts.d[0].transpose[2]]
            label_mask = label_mask[ts.d[0].transpose[1]]
            self.segmentations[brain_id] = label_mask

    @staticmethod
    def _parse_gcs_path(path, default_bucket="allen-nd-goog"):
        """
        Splits a GCS path into a (bucket, key) pair.

        Accepts either a full ``gs://bucket/key`` URL or a bucket-relative
        key. The kvstore "path" must be relative to the bucket, so the
        ``gs://bucket/`` prefix is stripped when present.

        Parameters
        ----------
        path : str
            Full gs:// URL or bucket-relative key.
        default_bucket : str, optional
            Bucket used when "path" has no gs:// scheme. Default is
            "allen-nd-goog".

        Returns
        -------
        Tuple[str]
            The (bucket, key) pair.
        """
        if path.startswith("gs://"):
            bucket, _, key = path[len("gs://"):].partition("/")
            return bucket, key
        return default_bucket, path

    def _load_swcs(self, brain_id, swc_pointer):
        if swc_pointer:
            # Initializations
            swc_dicts = self.swc_reader.read(swc_pointer)
            point_sets = list()

            # SWCs are expected to be dense in voxel space. Validate parent
            # links with Chebyshev distance so one-step 3D diagonals count as
            # adjacent, but do not rasterize edges into the mask.
            for swc_dict in swc_dicts:
                points = self.to_voxels(
                    np.asarray(swc_dict["xyz"], dtype=np.float32).copy()
                )
                if not len(points):
                    continue
                point_sets.append(points)
                id_to_index = {
                    int(node_id): i
                    for i, node_id in enumerate(swc_dict["id"])
                }
                edge_lengths = list()
                for child_index, parent_id in enumerate(swc_dict["pid"]):
                    parent_index = id_to_index.get(int(parent_id))
                    if parent_index is not None:
                        edge_lengths.append(
                            int(
                                np.max(
                                    np.abs(
                                        points[child_index]
                                        - points[parent_index]
                                    )
                                )
                            )
                        )
                long_edges = [length for length in edge_lengths if length > 1]
                if long_edges:
                    logger.warning(
                        "SWC for brain %s has %d parent-child edges longer "
                        "than one voxel (maximum Chebyshev length: %d)",
                        brain_id,
                        len(long_edges),
                        max(long_edges),
                    )

            if point_sets:
                self.skeletons[brain_id] = np.concatenate(point_sets, axis=0)

    # --- Sample Image Patches ---
    def __getitem__(self, dummy_input):
        """
        Returns a pair of transformed noisy and BM4D-denoised image patches.

        Parameters
        ----------
        dummy_input : Any
            Dummy argument required by PyTorch's "Dataset" interface for
            indexing. Not used in the patch sampling procedure.

        Returns
        -------
        x : numpy.ndarray
            Noisy image patch in the normalized transform domain.
        y : numpy.ndarray
            Target image patch in the normalized transform domain.
        fg_mask : numpy.ndarray
            Foreground mask (float 0/1) for the signal-preserving loss.
        """
        raw, teacher, fg_mask = self._sample_counts()
        return build_training_example(
            self.transform, self.preserve_foreground, raw, teacher, fg_mask
        )

    def _sample_counts(self):
        """
        Samples one patch and its BM4D target in offset-subtracted counts.

        This is the expensive step (cloud read + BM4D + foreground mask) and
        is exactly what the patch cache stores; the cheap transform + target
        construction is applied by build_training_example. The patch is drawn
        by sample_clean, which resamples past patches contaminated by an
        incoherent processing artifact before the (expensive) BM4D runs.

        Returns
        -------
        Tuple[numpy.ndarray]
            (raw, teacher, fg_mask) in count space. raw has the per-brain
            offset subtracted; teacher is the clipped BM4D denoising.
        """
        brain_id, voxel, raw, labels = self.sample_clean(self.read_counts)
        teacher = bm4d(raw, self.sigma_bm4d)
        teacher = np.clip(teacher, 0, self.transform.max_count)
        fg_mask = self.foreground_mask(brain_id, voxel, raw, labels=labels)
        return raw, teacher, fg_mask

    def read_counts(self, brain_id, center):
        """
        Reads a patch and subtracts the per-brain offset (count space).

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled brain.
        center : Tuple[int]
            Center voxel of the patch.

        Returns
        -------
        numpy.ndarray
            Offset-subtracted raw counts.
        """
        raw = np.asarray(self.read_patch(brain_id, center)).astype(np.float32)
        return raw - self.offsets.get(brain_id, 0.0)

    def sample_clean(self, read_counts):
        """
        Samples a patch, resampling past incoherent-artifact contamination.

        Draws a (brain, voxel), reads its raw counts and (if the brain is
        segmented) its label patch, and -- when reject_incoherent_patches is
        set -- checks whether the segmentation contains a bright, spatially
        incoherent processing artifact (see
        metrics.patch_has_incoherent_segment). Such a patch is discarded and a
        new one drawn, up to max_resample_attempts, because the artifact
        corrupts the raw input itself, so the patch is a poor training example
        even with the offending label removed. The check runs before BM4D, so
        rejected patches cost only a label read, not the denoising. Returning
        the labels lets the caller build the mask without a second read.

        When rejection is disabled this draws a single patch. On exhausting the
        attempt budget it returns the last patch drawn (rare; keeps the caller,
        e.g. a fixed-size cache build, from stalling).

        Parameters
        ----------
        read_counts : Callable[[str, Tuple[int]], numpy.ndarray]
            Returns the offset-subtracted raw counts for a (brain, voxel). Lets
            the validation cache read its own image while this TrainDataset
            supplies the segmentation and skeletons.

        Returns
        -------
        Tuple[str, Tuple[int], numpy.ndarray, numpy.ndarray or None]
            (brain_id, voxel, raw, labels); labels is None when the brain has
            no segmentation.
        """
        attempts = self.max_resample_attempts \
            if self.reject_incoherent_patches else 1
        brain_id = voxel = raw = labels = None
        for _ in range(max(1, attempts)):
            brain_id = self.sample_brain()
            voxel = self.sample_voxel(brain_id)
            raw = read_counts(brain_id, voxel)
            labels = None
            if brain_id in self.segmentations:
                labels = np.asarray(self.read_precomputed_patch(brain_id, voxel))
                if self.reject_incoherent_patches and patch_has_incoherent_segment(
                    labels,
                    raw,
                    min_autocorr=self.coherence_min_autocorr,
                    max_highfreq_frac=self.coherence_max_highfreq_frac,
                    min_segment_voxels=self.coherence_min_segment_voxels,
                    smooth_sigma=self.coherence_smooth_sigma,
                    coherence_lag=self.coherence_lag,
                ):
                    continue
            return brain_id, voxel, raw, labels
        return brain_id, voxel, raw, labels

    def foreground_mask(self, brain_id, center, raw, labels=None):
        """
        Builds a foreground mask for a patch from ground-truth annotations.

        Foreground is the union of the segmentation labels (used as-is unless
        segmentation_dilate > 0) and the traced skeleton (dilated to a neurite
        radius), so both segmented and traced neurites are preserved from the
        BM4D teacher while bright non-neuronal structures -- noise, off-target
        label -- are not. The skeleton union matters because the segmentation
        can miss neurites the ground-truth skeletons trace, and those patches
        are sampled deliberately. Brains with neither annotation fall back to
        the robust intensity threshold (should not occur when every brain is
        segmented).

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled brain.
        center : Tuple[int]
            Center voxel of the patch.
        raw : numpy.ndarray
            Raw image patch in counts, used only for the no-annotation
            fallback.
        labels : numpy.ndarray, optional
            Pre-read segmentation label patch, when the caller already read it
            (e.g. sample_clean). Avoids a second cloud read. Default is None.

        Returns
        -------
        numpy.ndarray
            Boolean foreground mask with the shape of "raw".
        """
        mask = self.annotation_mask(brain_id, center, labels=labels)
        return make_foreground_mask(raw) if mask is None else mask

    def annotation_mask(self, brain_id, center, labels=None):
        """
        Builds the ground-truth foreground mask (segmentation and skeleton).

        Unions the segmentation labels (dilated only when segmentation_dilate >
        0) with the rasterized, dilated skeleton for the patch. Returns None
        when the brain has neither annotation, so callers can fall back to an
        intensity mask. Incoherent-artifact segments are handled by rejecting
        the whole patch at sampling time (see sample_clean), not here, so this
        mask trusts the labels as given.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled brain.
        center : Tuple[int]
            Center voxel of the patch.
        labels : numpy.ndarray, optional
            Pre-read segmentation label patch. When None it is read here.
            Default is None.

        Returns
        -------
        numpy.ndarray or None
            Boolean foreground mask with shape "self.patch_shape", or None if
            the brain has no segmentation and no skeleton.
        """
        mask = None
        if brain_id in self.segmentations:
            if labels is None:
                labels = self.read_precomputed_patch(brain_id, center)
            mask = make_segmentation_mask(labels, dilate=self.segmentation_dilate)
        if brain_id in self.skeletons:
            skel = self.skeleton_mask(brain_id, center)
            mask = skel if mask is None else (mask | skel)
        return mask

    def skeleton_mask(self, brain_id, center):
        """
        Rasterizes the ground-truth skeleton points falling within a patch.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled brain.
        center : Tuple[int]
            Center voxel of the patch.

        Returns
        -------
        numpy.ndarray
            Boolean foreground mask with shape "self.patch_shape".
        """
        start = [c - d // 2 for c, d in zip(center, self.patch_shape)]
        return make_skeleton_mask(
            self.skeletons[brain_id],
            start,
            self.patch_shape,
            dilate=self.skeleton_radius,
        )

    def sample_brain(self):
        """
        Samples a brain ID from the loaded images.

        Returns
        -------
        brain_id : str
            Unique identifier of the sampled whole-brain.
        """
        return util.sample_once(self.imgs.keys())

    def sample_voxel(self, brain_id):
        """
        Samples a voxel from a brain volume, either foreground or interior.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate chosen according to the foreground or interior
            sampling strategy.
        """
        if random.random() < self.foreground_sampling_rate:
            return self.sample_foreground_voxel(brain_id)
        else:
            return self.sample_interior_voxel(brain_id)

    def sample_foreground_voxel(self, brain_id):
        """
        Samples a voxel likely to be part of the foreground of a neuron.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate representing a likely foreground location.
        """
        if brain_id in self.skeletons and np.random.random() > 0.5:
            return self.sample_skeleton_voxel(brain_id)
        elif brain_id in self.segmentations:
            return self.sample_segmentation_voxel(brain_id)
        else:
            return self.sample_bright_voxel(brain_id)

    def sample_interior_voxel(self, brain_id):
        """
        Samples a random voxel coordinate from the interior of a 3D image
        volume, avoiding boundary regions.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate sampled uniformly at random within the valid
            interior region of the image volume.
        """
        voxel = list()
        for s in self.imgs[brain_id].shape[2::]:
            upper = s - self.boundary_buffer
            voxel.append(random.randint(self.boundary_buffer, upper))
        return tuple(voxel)

    def sample_skeleton_voxel(self, brain_id):
        """
        Samples a voxel coordinate near a skeleton point.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        Tuple[int]
            Voxel coordinate near a skeleton point.
        """
        idx = random.randint(0, len(self.skeletons[brain_id]) - 1)
        radius = np.array(self.patch_shape) // 4
        shift = np.array([np.random.randint(-r, r + 1) for r in radius])
        return tuple(self.skeletons[brain_id][idx] + shift)

    def sample_segmentation_voxel(self, brain_id):
        """
        Sample a voxel coordinate whose corresponding segmentation patch
        contains a sufficiently large object.

        Parameters
        ----------
        brain_id : str
            Identifier for the image volume which must be a key in
            "self.segmentations".

        Returns
        -------
        best_voxel : Tuple[int]
            Voxel coordinate whose patch contains a sufficiently large object
            or had the largest object after 5 * self.prefetch attempts.
        """
        best_volume = 0
        best_voxel = self.sample_interior_voxel(brain_id)
        cnt = 0
        with ThreadPoolExecutor() as executor:
            while best_volume < self.min_segmentation_volume:
                # Read random image patches
                pending = dict()
                for _ in range(self.prefetch_foreground_sampling):
                    voxel = self.sample_interior_voxel(brain_id)
                    thread = executor.submit(
                        self.read_precomputed_patch, brain_id, voxel
                    )
                    pending[thread] = voxel

                # Check if labels patch has large enough object. Reads run
                # concurrently, but results are consumed in submission order
                # (not completion order) so ties break deterministically and
                # a seeded run is reproducible.
                for thread, voxel in pending.items():
                    labels_patch = thread.result()
                    vals, cnts = fastremap.unique(
                        labels_patch, return_counts=True
                    )

                    if len(cnts) > 1:
                        volume = np.max(cnts[1:])
                        if volume > best_volume:
                            best_voxel = voxel
                            best_volume = volume

                # Check number of tries
                cnt += 1
                if cnt > 5:
                    break
        return best_voxel

    def sample_bright_voxel(self, brain_id):
        """
        Samples a voxel whose patch has enough foreground voxels.

        Foreground is counted with the same robust mask used for targets and
        metrics (median + k * sigma), so the threshold adapts to each patch
        instead of using a fixed intensity cutoff. The occupancy requirement
        is low enough (min_foreground_voxels) to accept thin fibers.

        Parameters
        ----------
        brain_id : str
            Unique identifier of a whole-brain.

        Returns
        -------
        best_voxel : Tuple[int]
            Voxel coordinate whose patch has the most foreground voxels found,
            stopping once min_foreground_voxels is reached.
        """
        best_brightness = 0
        best_voxel = self.sample_interior_voxel(brain_id)
        cnt = 0
        with ThreadPoolExecutor() as executor:
            while best_brightness < self.min_foreground_voxels:
                # Read random image patches
                pending = dict()
                for _ in range(self.prefetch_foreground_sampling):
                    voxel = self.sample_interior_voxel(brain_id)
                    thread = executor.submit(
                        self.read_patch, brain_id, voxel
                    )
                    pending[thread] = voxel

                # Check if image patch has enough foreground. Reads run
                # concurrently, but results are consumed in submission order
                # (not completion order) so ties break deterministically and
                # a seeded run is reproducible.
                for thread, voxel in pending.items():
                    img_patch = thread.result()
                    brightness = int(make_foreground_mask(img_patch).sum())
                    if brightness > best_brightness:
                        best_voxel = voxel
                        best_brightness = brightness

                # Check number of tries
                cnt += 1
                if cnt > 5:
                    break
        return best_voxel

    # --- Helpers ---
    def __len__(self):
        """
        Gets the number of training examples used in each epoch.

        Returns
        -------
        int
            Number of training examples used in each epoch.
        """
        return self.n_examples_per_epoch

    def sample_intensity_values(self, n_patches=8):
        """
        Reads a few interior patches and returns their flattened counts.

        Used to calibrate a transform's background offset from a global
        sample of the training data.

        Parameters
        ----------
        n_patches : int, optional
            Number of interior patches to sample. Default is 8.

        Returns
        -------
        numpy.ndarray
            Flattened raw counts from the sampled patches.
        """
        values = list()
        for _ in range(n_patches):
            brain_id = self.sample_brain()
            voxel = self.sample_interior_voxel(brain_id)
            patch = np.asarray(self.read_patch(brain_id, voxel))
            values.append(patch.ravel())
        return np.concatenate(values)

    def read_patch(self, brain_id, center):
        """
        Reads an image patch from a Zarr array.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled brain.
        center : Tuple[int]
            Center of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        s = img_util.get_slices(center, self.patch_shape)
        return self.imgs[brain_id][(0, 0, *s)]

    def read_precomputed_patch(self, brain_id, center):
        """
        Reads an image patch from a Precomputed array.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled brain.
        center : Tuple[int]
            Center of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        s = img_util.get_slices(center, self.patch_shape)
        return self.segmentations[brain_id][s].read().result()

    def to_voxels(self, xyz_arr):
        """
        Converts 3D points from physical to voxel coordinates.

        Parameters
        ----------
        xyz_arr : numpy.ndarray
            Array with shape (n, 3) that contains 3D points.

        Returns
        -------
        numpy.ndarray
            3D Points converted to voxel coordinates.
        """
        for i in range(3):
            xyz_arr[:, i] = xyz_arr[:, i] / self.anisotropy[i]
        return np.flip(xyz_arr, axis=1).astype(int)


class ValidateDataset(Dataset):

    def __init__(
        self,
        patch_shape,
        sigma_bm4d=16,
        transform=None,
        preserve_foreground=True,
        offsets=None,
    ):
        """
        Instantiates a ValidateDataset object.

        Parameters
        ----------
        patch_shape : Tuple[int]
            Shape of image patches to be extracted.
        sigma_bm4d : float, optional
            Smoothing parameter used in the BM4D denoising algorithm. Default
            is 16.
        transform : IntensityTransform, optional
            Transform mapping raw counts to the normalized domain. Default is
            an asinh transform.
        preserve_foreground : bool, optional
            Whether targets keep raw counts on the foreground. Default is
            True.
        offsets : dict, optional
            Per-brain background offsets subtracted from raw patches before
            the transform. Default is None (no subtraction).
        """
        # Call parent class
        super(ValidateDataset, self).__init__()

        # Instance attributes
        self.patch_shape = patch_shape
        self.sigma_bm4d = sigma_bm4d
        self.transform = transform or build_transform({"kind": "asinh"})
        self.preserve_foreground = preserve_foreground
        self.offsets = offsets or dict()

        # Data structures
        self.example_ids = list()
        self.imgs = dict()
        self.denoised = list()
        self.noise = list()
        self.raws = list()
        self.fg_masks = list()

    def __len__(self):
        """
        Counts the number of examples in the dataset.

        Returns
        -------
        int
            Number of examples in the dataset.
        """
        return len(self.example_ids)

    def ingest_brain(self, brain_id, img_path):
        """
        Loads a brain image and stores it in the internal image dictionary.

        Parameters
        ----------
        brain_id : str
            Unique identifier for the brain corresponding to the image.
        img_path : str or Path
            Path to whole-brain image to be read.
        """
        self.imgs[brain_id] = img_util.read(img_path)

    def read_counts(self, brain_id, voxel):
        """
        Reads a patch and subtracts the per-brain offset (count space).

        Exposed so a caller that also needs the raw patch to build the
        foreground mask (e.g. the coherence gate) can read it once and hand it
        to "sample_counts", avoiding a second cloud read.

        Parameters
        ----------
        brain_id : hashable
            Unique identifier of the brain from which to extract the patch.
        voxel : Tuple[int]
            Voxel coordinates of the patch center in the brain volume.

        Returns
        -------
        numpy.ndarray
            Offset-subtracted raw counts.
        """
        raw = np.asarray(self.read_patch(brain_id, voxel)).astype(np.float32)
        return raw - self.offsets.get(brain_id, 0.0)

    def sample_counts(self, brain_id, voxel, fg_mask=None, raw=None):
        """
        Samples one validation patch and its BM4D target in count space.

        This is the expensive step (cloud read + BM4D + foreground mask) and
        is exactly what the validation cache stores; the cheap transform +
        target construction is applied by build_training_example. The mask is
        supplied by the caller -- the TrainDataset's annotation mask
        (segmentation unioned with skeleton), matching the training mask -- and
        falls back to the intensity threshold only when none is supplied.

        Parameters
        ----------
        brain_id : hashable
            Unique identifier of the brain from which to extract the patch.
        voxel : Tuple[int]
            Voxel coordinates of the patch center in the brain volume.
        fg_mask : numpy.ndarray, optional
            Precomputed foreground mask aligned with the sampled voxel. When
            None, the mask falls back to the robust intensity threshold.
        raw : numpy.ndarray, optional
            Offset-subtracted raw counts for this voxel, when already read by
            the caller (e.g. to build the coherence-gated mask). Avoids a
            redundant cloud read. Default is None (read here).

        Returns
        -------
        Tuple[numpy.ndarray]
            (raw, teacher, fg_mask) in count space. raw has the per-brain
            offset subtracted; teacher is the clipped BM4D denoising.
        """
        if raw is None:
            raw = self.read_counts(brain_id, voxel)
        teacher = bm4d(raw, self.sigma_bm4d)
        teacher = np.clip(teacher, 0, self.transform.max_count)
        if fg_mask is None:
            fg_mask = make_foreground_mask(raw)
        return raw, teacher, fg_mask

    def ingest_example(self, brain_id, voxel, fg_mask=None, raw=None):
        """
        Extracts, denoises, transforms, and stores an image patch.

        Parameters
        ----------
        brain_id : hashable
            Unique identifier of the brain from which to extract the patch.
        voxel : Tuple[int]
            Voxel coordinates of the patch center in the brain volume.
        fg_mask : numpy.ndarray, optional
            Precomputed foreground mask aligned with the sampled voxel. When
            None, the mask falls back to the intensity threshold.
        raw : numpy.ndarray, optional
            Offset-subtracted raw counts for this voxel, when already read by
            the caller. Avoids a redundant cloud read. Default is None.
        """
        # Sample image patch and its BM4D-denoised target
        raw, teacher, fg_mask = self.sample_counts(
            brain_id, voxel, fg_mask=fg_mask, raw=raw
        )

        # Preserve raw counts on the ground-truth neurite foreground
        if self.preserve_foreground:
            target = np.where(fg_mask, raw, teacher)
        else:
            target = teacher

        # Store transformed patches plus count-space metadata for metrics
        self.example_ids.append((brain_id, voxel))
        self.noise.append(self.transform.forward(raw))
        self.denoised.append(self.transform.forward(target))
        self.raws.append(raw)
        self.fg_masks.append(fg_mask.astype(np.float32))

    def __getitem__(self, idx):
        """
        Retrieves a single example from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        x : numpy.ndarray
            Noisy image patch in the normalized transform domain.
        y : numpy.ndarray
            BM4D-denoised image patch in the normalized transform domain.
        raw : numpy.ndarray
            Raw noisy image patch in counts (for count-space metrics).
        fg_mask : numpy.ndarray
            Foreground mask (float 0/1) for the metric split.
        """
        return (
            self.noise[idx],
            self.denoised[idx],
            self.raws[idx],
            self.fg_masks[idx],
        )

    # --- Helpers ---
    def read_patch(self, brain_id, center):
        """
        Reads an image patch from a Zarr array.

        Parameters
        ----------
        brain_id : str
            Unique identifier of the sampled brain.
        center : Tuple[int]
            Center of image patch to be read.

        Returns
        -------
        numpy.ndarray
            Image patch.
        """
        slices = img_util.get_slices(center, self.patch_shape)
        return self.imgs[brain_id][(0, 0, *slices)]


class CachedPatchDataset(Dataset):
    """
    Dataset that reads precomputed count-space patches from disk.

    The expensive cloud reads + BM4D + foreground masks are precomputed once
    (see scripts/precompute.py --split train) into memory-mapped arrays; this
    dataset applies only the cheap transform + target construction, so
    training becomes GPU-bound instead of BM4D-bound. Each cache entry is
    addressable by index so the DataLoader controls epoch ordering.

    Attributes
    ----------
    patch_shape : Tuple[int]
        Shape of the cached patches.
    transform : IntensityTransform
        Transform mapping counts to the normalized domain.
    """

    def __init__(self, cache_dir, transform=None, preserve_foreground=True):
        """
        Instantiates a CachedPatchDataset.

        Parameters
        ----------
        cache_dir : str or Iterable[str]
            Directory or list of directories containing raw.npy, teacher.npy,
            and fg.npy.
        transform : IntensityTransform, optional
            Transform mapping counts to the normalized domain. Default is an
            asinh transform.
        preserve_foreground : bool, optional
            Whether the target keeps raw counts on the foreground. Default is
            True.
        """
        # Call parent class
        super().__init__()

        # Create cache directory list
        if isinstance(cache_dir, (str, os.PathLike)):
            cache_dirs = [cache_dir]
        elif isinstance(cache_dir, Iterable):
            cache_dirs = list(cache_dir)
        else:
            raise TypeError(
                "cache_dir must be a path or an iterable of paths"
            )

        # Load arrays with mmap mode
        self.raw = self._load_cached_arrs(cache_dirs, "raw")
        self.teacher = self._load_cached_arrs(cache_dirs, "teacher")
        self.fg = self._load_cached_arrs(cache_dirs, "fg")
        self._validate_cache()

        self.lengths = [len(x) for x in self.raw]
        self.cumulative_lengths = np.cumsum(self.lengths)

        # Instance attributes
        self.transform = transform or build_transform({"kind": "asinh"})
        self.preserve_foreground = preserve_foreground
        self.patch_shape = tuple(self.raw[0].shape[1:])

    def __len__(self):
        """Number of cached training examples."""
        return int(self.cumulative_lengths[-1])

    def __getitem__(self, idx):
        """
        Returns a cached example as (x, y, fg_mask).

        Parameters
        ----------
        idx : int
            Index of the example to retrieve.

        Returns
        -------
        Tuple[numpy.ndarray]
            (x, y, fg_mask) for the model.
        """
        raw, teacher, fg_mask = self._get_arrays(idx)
        return build_training_example(
            self.transform, self.preserve_foreground, raw, teacher, fg_mask
        )

    def _get_arrays(self, idx):
        """
        Retrieves raw cached arrays for a global sample index.
    
        Parameters
        ----------
        idx : int
            Global sample index.

        Returns
        -------
        Tuple[np.ndarray]
            Raw counts, teacher target, and foreground mask arrays.
        """
        cache_idx, local = self._locate(idx)
        raw = np.asarray(self.raw[cache_idx][local], dtype=np.float32)
        teacher = np.asarray(self.teacher[cache_idx][local], dtype=np.float32)
        fg_mask = np.asarray(self.fg[cache_idx][local], dtype=np.float32)
        return raw, teacher, fg_mask

    def _locate(self, idx):
        """
        Maps a global sample index to a cache index and local offset.

        Parameters
        ----------
        idx : int
            Global sample index.

        Returns
        -------
        cache_idx : int
            Index of the cache containing the sample.
        offset : int
            Local index within the selected cache.
        """
        # Check for valid index
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # Find dataset and local indices
        cache_idx = np.searchsorted(
            self.cumulative_lengths, idx, side="right"
        )
        offset = (
            idx
            if cache_idx == 0
            else idx - self.cumulative_lengths[cache_idx - 1]
        )
        return cache_idx, offset

    @staticmethod
    def _load_cached_arrs(cache_dirs, name):
        """
        Loads memory-mapped arrays from multiple cache directories.

        Parameters
        ----------
        cache_dirs : List[str]
            Cache directories containing the array files.
        name : str
            Array name (e.g., "raw", "teacher", or "fg").

        Returns
        -------
        List[np.ndarray]
            Memory-mapped arrays, one per cache directory.
        """
        return [
            np.load(os.path.join(d, f"{name}.npy"), mmap_mode="r")
            for d in cache_dirs
        ]

    def _validate_cache(self):
        # Check same number of cached datasets
        assert len(self.raw) == len(self.teacher) == len(self.fg)

        # Check cache is nonempty
        if len(self.raw) == 0:
            raise ValueError("No cached arrays found")

        # Check patch shapes are the same within dataset
        for raw, teacher, fg in zip(self.raw, self.teacher, self.fg):
            assert len(raw) == len(teacher) == len(fg)
            assert raw.shape[1:] == teacher.shape[1:] == fg.shape[1:]

        # Check patch shapes are same across datasets
        shapes = {r.shape[1:] for r in self.raw}
        if len(shapes) > 1:
            raise ValueError(
                f"Inconsistent patch shapes across cache_dirs: {shapes}"
            )


class CachedValidateDataset(CachedPatchDataset):
    """
    Cached validation dataset with raw counts returned for metrics.
    """

    def __getitem__(self, idx):
        """
        Returns a cached validation example as (x, y, raw, fg_mask).

        Parameters
        ----------
        idx : int
            Index of the example to retrieve.

        Returns
        -------
        Tuple[numpy.ndarray]
            (x, y, raw, fg_mask) matching ValidateDataset.__getitem__: model
            input, target, raw counts (for count-space metrics), and the
            foreground mask (float 0/1).
        """
        raw, teacher, fg_mask = self._get_arrays(idx)
        x, y, fg = build_training_example(
            self.transform, self.preserve_foreground, raw, teacher, fg_mask
        )
        return x, y, raw, fg


# --- Custom Dataloader ---
_WORKER_DATASET = None


def _worker_init(dataset):
    """Stores the dataset in a per-worker global (avoids per-task pickling)."""
    global _WORKER_DATASET
    _WORKER_DATASET = dataset


def _worker_getitem(idx):
    """Fetches one example from the per-worker dataset global."""
    return _WORKER_DATASET[idx]


class DataLoader:
    """
    Prefetching DataLoader that overlaps batch preparation with training.

    A background thread fills a bounded queue of prepared batches while the
    training loop consumes them, so the GPU is not starved. Per-example work
    runs either in-thread (num_workers=0, best for the cheap cached dataset)
    or in a persistent process pool (num_workers>0 or None, for the cloud
    dataset where BM4D dominates). The pool is created once per epoch, and the
    dataset is pickled to workers once at pool startup rather than per task.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to iterate over.
    batch_size : int
        Number of examples in each batch.
    patch_shape : Tuple[int]
        Shape of image patch expected by the model.
    """

    def __init__(
        self,
        dataset,
        batch_size=16,
        num_workers=None,
        prefetch=2,
        shuffle=False,
        seed=0,
    ):
        """
        Instantiates a DataLoader object.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to iterate over.
        batch_size : int, optional
            Number of examples in each batch. Default is 16.
        num_workers : int, optional
            Process-pool workers for per-example work. None uses the CPU count;
            0 runs in-thread (best for the cheap cached dataset). Default is
            None.
        prefetch : int, optional
            Number of batches prepared ahead of the consumer. Default is 2.
        shuffle : bool, optional
            Whether to deterministically shuffle indices each epoch. Default
            is False.
        seed : int, optional
            Base seed used with the epoch to generate shuffled indices.
            Default is 0.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.patch_shape = dataset.patch_shape
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Sets the epoch used to generate a reproducible shuffled order."""
        self.epoch = epoch

    def __iter__(self):
        """
        Yields batches, preparing them ahead of the consumer in a thread.

        Returns
        -------
        iterator
            Yields batches of tensors.
        """
        if self.shuffle:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch])
            )
            indices = rng.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))
        batches = [
            indices[start:start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        if not batches:
            return

        executor = None
        if self.num_workers != 0:
            executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_worker_init,
                initargs=(self.dataset,),
            )

        result_queue = queue.Queue(maxsize=max(1, self.prefetch))
        done = object()

        def produce():
            try:
                for batch_indices in batches:
                    result_queue.put(
                        (None, self._load_batch(executor, batch_indices))
                    )
            except Exception as exc:  # surface loader errors to the consumer
                result_queue.put((exc, None))
            else:
                result_queue.put((done, None))

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        try:
            while True:
                flag, batch = result_queue.get()
                if flag is done:
                    break
                if flag is not None:
                    raise flag
                yield batch
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            thread.join(timeout=1)

    def _load_batch(self, executor, indices):
        batch_size = len(indices)

        # Per-example work: in-thread when there is no pool, else in parallel.
        if executor is None:
            results = [self.dataset[idx] for idx in indices]
        else:
            futures = [executor.submit(_worker_getitem, idx) for idx in indices]
            results = [f.result() for f in as_completed(futures)]

        # Stack each field of the example tuple into its own batch tensor.
        # Handles the (x, y, fg_mask) train shape and the
        # (x, y, raw, fg_mask) validation shape alike.
        shape = (batch_size, 1,) + self.patch_shape
        n_fields = len(results[0])
        batched = [np.zeros(shape) for _ in range(n_fields)]
        for i, fields in enumerate(results):
            for j, field in enumerate(fields):
                batched[j][i, 0, ...] = field
        return tuple(to_tensor(arr) for arr in batched)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


# --- Helpers ---
def init_datasets(
    brain_ids,
    img_paths_json,
    patch_shape,
    foreground_sampling_rate=0.5,
    min_foreground_voxels=50,
    min_segmentation_volume=200,
    n_train_examples_per_epoch=100,
    n_validate_examples=0,
    reject_incoherent_patches=False,
    coherence_min_autocorr=0.4,
    coherence_max_highfreq_frac=0.35,
    coherence_min_segment_voxels=50,
    coherence_smooth_sigma=1.0,
    coherence_lag=2,
    max_resample_attempts=50,
    segmentation_prefixes_path=None,
    segmentation_dilate=0,
    sigma_bm4d=16,
    skeleton_radius=2,
    swc_pointers=None,
    transform_cfg=None,
    preserve_foreground=True,
    offsets=None,
):
    # Initializations
    if transform_cfg is None:
        transform_cfg = {"kind": "asinh"}
    train_dataset = TrainDataset(
        patch_shape,
        foreground_sampling_rate=foreground_sampling_rate,
        min_foreground_voxels=min_foreground_voxels,
        min_segmentation_volume=min_segmentation_volume,
        n_examples_per_epoch=n_train_examples_per_epoch,
        offsets=offsets,
        preserve_foreground=preserve_foreground,
        reject_incoherent_patches=reject_incoherent_patches,
        coherence_min_autocorr=coherence_min_autocorr,
        coherence_max_highfreq_frac=coherence_max_highfreq_frac,
        coherence_min_segment_voxels=coherence_min_segment_voxels,
        coherence_smooth_sigma=coherence_smooth_sigma,
        coherence_lag=coherence_lag,
        max_resample_attempts=max_resample_attempts,
        segmentation_dilate=segmentation_dilate,
        sigma_bm4d=sigma_bm4d,
        skeleton_radius=skeleton_radius,
    )
    val_dataset = ValidateDataset(
        patch_shape,
        sigma_bm4d=sigma_bm4d,
        preserve_foreground=preserve_foreground,
        offsets=offsets,
    )

    # Read segmentation path lookup (if applicable)
    if segmentation_prefixes_path:
        segmentation_paths = util.read_json(segmentation_prefixes_path)
    else:
        segmentation_paths = dict()

    # Load data
    for brain_id in tqdm(brain_ids, desc="Load Data"):
        # Set image path
        img_path = get_img_prefix(brain_id, img_paths_json) + str(0)

        # Set segmentation path
        if brain_id in segmentation_paths:
            segmentation_path = segmentation_paths[brain_id]
        else:
            segmentation_path = None

        # Set SWC pointer
        if swc_pointers:
            swc_pointer = deepcopy(swc_pointers)
            swc_pointer["path"] += f"/{brain_id}/world"
        else:
            swc_pointer = None

        # Ingest data
        val_dataset.ingest_brain(brain_id, img_path)
        train_dataset.ingest_brain(
            brain_id, img_path, segmentation_path, swc_pointer
        )

    # Resolve one frozen transform, optionally calibrating the offset from a
    # global training sample, and share it across train and validation.
    if transform_cfg.get("calibrate", {}).get("offset", False):
        sample = train_dataset.sample_intensity_values()
        transform_cfg = calibrate_transform(transform_cfg, sample)
    transform = build_transform(transform_cfg)
    train_dataset.transform = transform
    val_dataset.transform = transform

    # Generate validation examples. The voxel is drawn from the train dataset,
    # which owns the segmentations and skeletons, so build the annotation mask
    # (segmentation unioned with skeleton) here and hand it to the validation
    # dataset (which loads neither) for a foreground mask consistent with
    # training.
    for _ in range(n_validate_examples):
        brain_id, voxel, raw, labels = train_dataset.sample_clean(
            val_dataset.read_counts
        )
        fg_mask = train_dataset.annotation_mask(brain_id, voxel, labels=labels)
        val_dataset.ingest_example(brain_id, voxel, fg_mask=fg_mask, raw=raw)
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
