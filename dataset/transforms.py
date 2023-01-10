# Transform module
# Converts custom scans (or other sources) to expected kitti value ranges and distributions (optional)

import torch
import numpy as np
from scipy.stats import norm

from visualization import img_vis


class DataAnalyzer:
    def __init__(self, proj_in):
        self.proj_in = proj_in
        self.proj_x_np = np.copy(self.proj_in[0][0].numpy()).reshape(-1)
        self.proj_y_np = np.copy(self.proj_in[0][1].numpy()).reshape(-1)
        self.proj_z_np = np.copy(self.proj_in[0][2].numpy()).reshape(-1)
        self.proj_range_np = np.copy(self.proj_in[0][3].numpy()).reshape(-1)
        self.proj_remission_np = np.copy(self.proj_in[0][4].numpy()).reshape(-1)

    def print_values(self):
        print(f"proj_x range: [{min(self.proj_x_np)}, {max(self.proj_x_np)}]")
        print(f"proj_y range: [{min(self.proj_y_np)}, {max(self.proj_y_np)}]")
        print(f"proj_z range: [{min(self.proj_z_np)}, {max(self.proj_z_np)}]")
        print(f"proj_range range: [{min(self.proj_range_np)}, {max(self.proj_range_np)}]")
        print(f"proj_remission range: [{min(self.proj_remission_np)}, {max(self.proj_remission_np)}]")

        mu, std = norm.fit(self.proj_x_np)
        print(f"proj_x stats: mu: {mu}, std: {std}")
        mu, std = norm.fit(self.proj_y_np)
        print(f"proj_y stats: mu: {mu}, std: {std}")
        mu, std = norm.fit(self.proj_z_np)
        print(f"proj_z stats: mu: {mu}, std: {std}")
        mu, std = norm.fit(self.proj_range_np)
        print(f"proj_range stats: mu: {mu}, std: {std}")
        mu, std = norm.fit(self.proj_remission_np)
        print(f"proj_remission stats: mu: {mu}, std: {std}")

    def visualize_histograms(self):
        img_vis.visualize_histogram(self.proj_x_np, title='X', bins=255)
        img_vis.visualize_histogram(self.proj_y_np, title='Y', bins=255)
        img_vis.visualize_histogram(self.proj_z_np, title='Z', bins=255)
        img_vis.visualize_histogram(self.proj_range_np, title='Range', bins=255)
        img_vis.visualize_histogram(self.proj_remission_np, title='Remission', bins=255, plot_dist=True)


class Transforms:
    def __init__(self):
        return

    @staticmethod
    def transform(proj: torch.Tensor, params: dict):
        assert (isinstance(params, dict))
        assert (isinstance(proj, torch.Tensor))

        if proj.shape[0] != 5:
            print(f"Shape of projection ({proj.shape}) is not valid, expected [5, 64, 2048]")
            raise "Error!"

        proj_cp = torch.clone(proj)

        intensity_max = params["intensity"]["max"]
        intensity_min = params["intensity"]["min"]

        rem_max = torch.amax(proj)
        rem_min = torch.amin(proj)

        print(f"rem_max: {rem_max}, rem_min: {rem_min}")
        print(f"intensity_max: {intensity_max}, intensity_min: {intensity_min}")

        m = (intensity_max-intensity_min) / (rem_max-rem_min)
        n = rem_max-intensity_max*m

        proj_cp[4] = proj_cp[4]*m+n

        return proj_cp