"""
Estimate per-brain background offsets from the level-5 multiscale zarr.

For each brain in the training list, this reads the coarse (level-5) image,
computes a low percentile as the background / black-point, and writes a
{brain_id: offset} JSON. It also prints the distribution of offsets across
brains so the fixed-vs-per-brain decision falls out of the numbers:

    * spread << scale  -> a single fixed offset (the median) is fine
    * spread >= scale  -> prefer a per-brain offset

The percentile is computed over NONZERO voxels so that zero-padding outside
the imaged volume does not drag the estimate down to 0. The per-brain zero
fraction is reported so padding is visible. Level 5 is ~32x downsampled, so
its voxels are local averages: the estimate is a smoothed black point, not
the raw-resolution noise floor.

Note: this reads each whole level-5 volume into memory (tens to a few hundred
MB per brain) and processes brains sequentially.

"""

import numpy as np
from tqdm import tqdm

from aind_exaspim_dataset_utils.s3_util import get_img_prefix

from aind_exaspim_image_compression.utils import img_util, util


def estimate_offset(brain_id, img_prefixes_path, level, percentile):
    """
    Estimates the background offset for a single brain.

    Parameters
    ----------
    brain_id : str
        Unique identifier of the brain.
    img_prefixes_path : str
        Path to the JSON mapping brain IDs to image prefixes.
    level : int
        Multiscale level to read (e.g., 5).
    percentile : float
        Low percentile used as the background estimate (e.g., 0.1).

    Returns
    -------
    dict
        Offset (nonzero), offset over all voxels, nonzero median, and the
        fraction of zero voxels.
    """
    prefix = get_img_prefix(brain_id, img_prefixes_path)
    arr = img_util.read(prefix + str(level))
    vol = np.asarray(arr[0, 0]).reshape(-1)  # channel 0, timepoint 0
    nonzero = vol[vol > 0]
    zero_fraction = 1.0 - nonzero.size / vol.size
    return {
        "offset": (
            float(np.percentile(nonzero, percentile))
            if nonzero.size else float("nan")
        ),
        "offset_all_voxels": float(np.percentile(vol, percentile)),
        "median": (
            float(np.median(nonzero)) if nonzero.size else float("nan")
        ),
        "zero_fraction": zero_fraction,
    }


def main():
    # Estimate an offset per brain
    brain_ids = util.read_txt(brain_ids_path)
    offsets = dict()
    for brain_id in tqdm(brain_ids, desc="Estimate offsets"):
        try:
            result = estimate_offset(
                brain_id, img_prefixes_path, level, percentile
            )
            offsets[brain_id] = result["offset"]
            print(
                f"{brain_id}: offset={result['offset']:.1f}  "
                f"(all={result['offset_all_voxels']:.1f}, "
                f"median={result['median']:.1f}, "
                f"zeros={100 * result['zero_fraction']:.1f}%)"
            )
        except Exception as e:
            print(f"{brain_id}: FAILED ({e})")

    # Write per-brain offsets
    util.write_json(output_path, offsets)
    print(f"\nWrote {len(offsets)} offsets to {output_path}")

    # Summarize the spread to inform fixed-vs-per-brain
    values = np.array([v for v in offsets.values() if np.isfinite(v)])
    if values.size:
        lo, med, hi = float(values.min()), float(np.median(values)), \
            float(values.max())
        spread = hi - lo
        print("\n--- Background offset distribution ---")
        print(f"  brains:  {values.size}")
        print(f"  min:     {lo:.1f}")
        print(f"  median:  {med:.1f}")
        print(f"  max:     {hi:.1f}")
        print(
            f"  spread:  {spread:.1f} counts "
            f"({spread / scale_hint:.2f} x scale={scale_hint:g})"
        )
        if spread < scale_hint:
            print(f"  => spread < scale: a FIXED offset ~{med:.0f} is fine.")
        else:
            print("  => spread >= scale: prefer a PER-BRAIN offset.")


if __name__ == "__main__":
    # Paths
    brain_ids_path = "/data/train_brain_ids.txt"
    img_prefixes_path = "/data/exaspim_image_prefixes.json"
    output_path = "/data/exaspim_background_offsets.json"

    # Parameters
    level = 5
    percentile = 0.1
    scale_hint = 32.0  # asinh knee; only used to judge whether spread matters

    main()
