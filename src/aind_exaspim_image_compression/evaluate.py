"""
Created on Tue June 3 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for evaluating a denoising model.

"""

from bm4d import bm4d
from numcodecs import blosc
from tqdm import tqdm

import ast
import numpy as np
import os
import pandas as pd
import torch

from aind_exaspim_image_compression.machine_learning import data_handling
from aind_exaspim_image_compression.inference import (
    predict, predict_patch, stitch
)
from aind_exaspim_image_compression.utils import img_util, util
from aind_exaspim_image_compression.utils.img_util import (
    compute_cratio, compute_ssim3D, compute_mae, compute_lmax
)


class SupervisedEvaluator:
    def __init__(self, img_paths, model, output_dir):
        # Instance attributes
        self.codec = blosc.Blosc(cname="zstd", clevel=6, shuffle=blosc.SHUFFLE)
        self.img_paths = img_paths
        self.model = model
        self.model.eval().to("cuda")

        # Initialize output directory
        self.output_dir = output_dir
        util.mkdir(output_dir, delete=True)

        # Load images
        self.load_images()

    def load_images(self):
        # Initialize MIPs directory
        noise_dir = os.path.join(self.output_dir, "noise_mips")
        util.mkdir(noise_dir)

        # Read images
        self.noise_imgs = dict()
        self.noise_cratios = dict()
        for img_path in self.img_paths:
            block_id = self.find_img_name(img_path)
            img = img_util.read(img_path)
            self.noise_cratios[block_id] = compute_cratio(img, self.codec)
            self.noise_imgs[block_id] = img

            output_path = os.path.join(noise_dir, block_id)
            img_util.plot_mips(img, output_path=output_path)

    # --- Main ---
    def run(self, model_path):
        # Initializations
        self.model.load_state_dict(torch.load(model_path))
        model_name = os.path.basename(model_path)
        results_dir = os.path.join(self.output_dir, model_name)
        util.mkdir(results_dir)

        # Generate prediction
        rows = sorted(list(self.noise_imgs.keys()))
        df = pd.DataFrame(index=rows, columns=["cratio", "ssim"])
        desc = "Denoise Blocks"
        for block_id, noise in tqdm(self.noise_imgs.items(), desc=desc):
            # Run model
            coords, preds = predict(noise, self.model, verbose=False)
            denoised = stitch(noise, coords, preds)

            # Compute metrics
            df.loc[block_id, "cratio"] = compute_cratio(denoised, self.codec)
            df.loc[block_id, "ssim"] = compute_ssim3D(
                noise[0, 0, ...],
                denoised[0, 0, ...],
                data_range=np.max(noise),
            )

            # Save MIPs
            output_path = os.path.join(results_dir, block_id)
            img_util.plot_mips(denoised, output_path=output_path)

        # Save metrics
        path = os.path.join(results_dir, "results.csv")
        df.to_csv(path, index=False)
        return df

    # --- Helpers ---
    def find_img_name(self, img_path):
        for part in img_path.split("/"):
            if "block_" in part:
                return part
        raise Exception(f"Block ID not found in {img_path}")


class UnsupervisedEvaluator:
    def __init__(self, root_dir, model, img_paths_json, patch_shape):
        # Class attributes
        self.codec = blosc.Blosc(cname="zstd", clevel=6, shuffle=blosc.SHUFFLE)
        self.img_paths_json = img_paths_json
        self.model = model
        self.patch_shape = patch_shape
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.result_dir = os.path.join(root_dir, "models")

        # Initialize directories
        util.mkdir(self.result_dir)

    # --- Main ---
    def run(self, model_path, is_test=True):
        # Initializations
        brain_ids_dict = self.read_brain_ids(is_test)
        dataset = self.init_dataset(brain_ids_dict)
        self.ingest_model(model_path)

        # Evaluation
        exp_name = "test" if is_test else "train"
        model_name = os.path.basename(model_path)
        for brain_id in brain_ids_dict["alpha"] + brain_ids_dict["beta"]:
            for is_foreground in [True, False]:
                # Generate result
                result = self.compute_metrics(
                    brain_id,
                    dataset,
                    is_foreground=is_foreground,
                    is_test=is_test
                )

                # Save result
                voxel_type = "foreground" if is_foreground else "background"
                filename = f"{exp_name}-{voxel_type}-{brain_id}.csv"
                path = os.path.join(self.result_dir, model_name, filename)
                pd.DataFrame(result).to_csv(path, index=False)

    def compute_metrics(
        self, brain_id, dataset, is_foreground=True, is_test=True
    ):
        # Initializations
        voxels = self.read_voxels(brain_id, is_foreground)
        metrics = {
            "cratio_noise": list(),
            "cratio_gt": list(),
            "cratio": list(),
            "ssim_noise": list(),
            "ssim_gt": list(),
            "l1_gt": list(),
            "lmax_gt": list()
        }

        # Run evaluation
        for voxel in tqdm(voxels, desc=brain_id):
            # Get images
            input_noise = dataset.get_patch(brain_id, voxel)
            noise = input_noise[5:-5, 5:-5, 5:-5]
            denoised_gt = np.maximum(bm4d(noise, 10), 0).astype(int)
            denoised = predict_patch(input_noise, self.model)[5:-5, 5:-5, 5:-5]
 
            # Compute metrics
            metrics["cratio"].append(compute_cratio(denoised, self.codec))
            metrics["cratio_noise"].append(compute_cratio(noise, self.codec))
            metrics["cratio_gt"].append(compute_cratio(denoised_gt, self.codec))

            metrics["ssim_noise"].append(compute_ssim3D(noise, denoised))
            metrics["ssim_gt"].append(compute_ssim3D(denoised_gt, denoised))

            metrics["l1_gt"].append(compute_mae(denoised_gt, denoised))
            metrics["lmax_gt"].append(compute_lmax(denoised_gt, denoised))
        return metrics

    # --- Helpers ---
    def init_dataset(self, brain_ids_dict):
        dataset, _ = data_handling.init_datasets(
            brain_ids_dict["alpha"] + brain_ids_dict["beta"],
            self.img_paths_json,
            self.patch_shape,
            0,
        )
        return dataset

    def ingest_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        model_name = os.path.basename(model_path)
        util.mkdir(os.path.join(self.result_dir, model_name))

    def read_brain_ids(self, is_test=True):
        exp_name = "test" if is_test else "train"
        path = os.path.join(self.data_dir, f"{exp_name}_brain_ids.json")
        return util.read_json(path)

    def read_voxels(self, brain_id, is_foreground=True):
        try:
            voxel_type = "foreground" if is_foreground else "background"
            path = os.path.join(self.data_dir, f"{voxel_type}-{brain_id}.csv")
            return pd.read_csv(path)["voxels"].apply(ast.literal_eval)
        except FileNotFoundError:
            return list()
