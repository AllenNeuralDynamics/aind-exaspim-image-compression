"""
Created on Wed April 30 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from tqdm import tqdm

import numpy as np
import torch

from aind_exaspim_image_compression.utils import img_util, util


def predict(img, model, batch_size=32, patch_size=64, overlap=16):
    # Initializations
    batch_coords, batch_inputs, mn_mx = list(), list(), list()
    coords = generate_coords(img, patch_size, overlap)

    # Main
    pbar = tqdm(total=len(coords), desc="Denoise")
    preds = list()
    for idx, (i, j, k) in enumerate(coords):
        # Get end coord
        i_end = min(i + patch_size, img.shape[0])
        j_end = min(j + patch_size, img.shape[1])
        k_end = min(k + patch_size, img.shape[2])

        # Get patch
        patch = img[i:i_end, j:j_end, k:k_end]
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
            input_tensor =to_tensor(np.stack(batch_inputs))
            with torch.no_grad():
                output_tensor = model(input_tensor)

            # Store result
            output_tensor = output_tensor.cpu()
            for cnt in range(output_tensor.shape[0]):
                mn, mx = mn_mx[cnt]
                patch = np.array(output_tensor[cnt, 0, ...])
                preds.append(patch * mx + mn)
                pbar.update(1)

            batch_coords.clear()
            batch_inputs.clear()
            mn_mx.clear()
    return coords, preds


def stitch(img, coords, preds, patch_size=64, trim=5):
    denoised_accum = np.zeros_like(img, dtype=np.float32)
    weight_map = np.zeros_like(img, dtype=np.float32)

    for (i, j, k), pred in zip(coords, preds):
        # Determine how much to trim
        trim_start = trim
        trim_end = patch_size - trim

        # Trim prediction
        pred_trimmed = pred[trim_start:trim_end, trim_start:trim_end, trim_start:trim_end]

        # Adjust insertion indices
        i_start = i + trim
        j_start = j + trim
        k_start = k + trim

        i_end = i_start + pred_trimmed.shape[0]
        j_end = j_start + pred_trimmed.shape[1]
        k_end = k_start + pred_trimmed.shape[2]

        # Clip to image bounds (for safety)
        i_end = min(i_end, img.shape[0])
        j_end = min(j_end, img.shape[1])
        k_end = min(k_end, img.shape[2])

        i_start = max(i_start, 0)
        j_start = max(j_start, 0)
        k_start = max(k_start, 0)

        denoised_accum[i_start:i_end, j_start:j_end, k_start:k_end] += pred_trimmed[:i_end - i_start, :j_end - j_start, :k_end - k_start]
        weight_map[i_start:i_end, j_start:j_end, k_start:k_end] += 1

    # Average accumulated
    weight_map[weight_map == 0] = 1
    denoised = denoised_accum / weight_map

    # Fill boundary trim
    fill_value = np.percentile(denoised[trim:-trim, trim:-trim, trim:-trim], 10)
    return img_util.fill_boundary(denoised, trim, fill_value)


# --- Helpers ---
def add_padding(patch, patch_size):
    pad_width = [
        (0, patch_size - patch.shape[0]),
        (0, patch_size - patch.shape[1]),
        (0, patch_size - patch.shape[2]),
    ]
    return np.pad(patch, pad_width, mode='constant', constant_values=0)


def generate_coords(img, patch_size, overlap):
    coords = list()
    stride = patch_size - overlap
    for i in range(0, img.shape[0] - patch_size + stride, stride):
        for j in range(0, img.shape[1] - patch_size + stride, stride):
            for k in range(0, img.shape[2] - patch_size + stride, stride):
                coords.append((i, j, k))
    return coords


def to_tensor(arr):
    return torch.tensor(arr[:, np.newaxis, ...]).to("cuda")
