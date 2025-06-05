"""
Created on Tue June 3 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for evaluating a denoising model on an image dataset.

"""

from numcodecs import blosc
from tqdm import tqdm

import numpy as np
import os
import pandas as pd
import torch

from aind_exaspim_image_compression.inference import predict, stitch
from aind_exaspim_image_compression.utils import img_util, util


class Evaluator:
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
            self.noise_cratios[block_id] = img_util.compute_cratio(
                img, self.codec
            )
            self.noise_imgs[block_id] = img

            output_path = os.path.join(noise_dir, block_id)
            img_util.plot_mips(img[0, 0, ...], output_path=output_path)

    # --- Main ---
    def run(self, model_path):
        # Initializations
        self.model.load_state_dict(torch.load(model_path))
        model_name = os.path.basename(model_path)
        results_dir = os.path.join(self.output_dir, model_name)
        util.mkdir(results_dir)

        # Generate prediction
        rows = list(self.noise_imgs.keys())
        df = pd.DataFrame(index=rows, columns=["cratio", "ssim"])
        desc = "Denoise Blocks"
        for block_id, noise in tqdm(self.noise_imgs.items(), desc=desc):
            # Run model
            coords, preds = predict(noise, self.model, verbose=False)
            denoised = stitch(noise, coords, preds)
            denoised = denoised.astype(np.uint16)

            # Compute metrics
            df.loc[block_id, "cratio"] = img_util.compute_cratio(
                denoised, self.codec
            )
            df.loc[block_id, "ssim"] = img_util.compute_ssim3D(
                noise[0, 0, ...],
                denoised[0, 0, ...],
                data_range=np.percentile(noise, 99.9),
            )

            # Save MIPs
            output_path = os.path.join(results_dir, block_id)
            img_util.plot_mips(denoised[0, 0, ...], output_path=output_path)

        # Save metrics
        path = os.path.join(results_dir, "results.csv")
        df.to_csv(path, index=True)
        return model_name, df

    # --- Helpers ---
    def find_img_name(self, img_path):
        for part in img_path.split("/"):
            if "block_" in part:
                return part
        raise Exception(f"Block ID not found in {img_path}")
