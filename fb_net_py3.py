"""
fb_net_py3.py - StarNet's FBNet ported to Python 3 / yt

Phase 1A: predictions only.



Deposition into BaryonField requires Enzo
runtime (libyt.grid_data writes) and is wired in separately during the
in-situ integration phase.







Key changes from Azton's Py2 version:
- removed `from enzo import grid_data, conversion_factors`
- removed `from inline_routines import *` (deposition helpers, snapshot-irrelevant)
- removed `from mpi4py import` (single-process)
- Py2 print statements -> print()
- apply_spherical_feedback / deposit_or_find_volume omitted entirely
- forward() returns predictions; NO EFFECTS on BaryonField


"""

import os
import torch
import numpy as np

from IMF_Sampler import IMF_sampler


class FBNet:
    def __init__(self, cfg):
        self.FBmethod = cfg["fbnet"]["method"]
        if self.FBmethod.lower() == "uniform_sphere":
            self.final_neutral_frac = cfg["fbnet"].getfloat("neutral_fraction")
            self.final_ionize_frac = 1.0 - self.final_neutral_frac
            self.model_time = cfg["fbnet"].getint("model_time")
            self.model_dt = cfg["fbnet"].getint("model_dt")
            
            # IMF sampler
            self.imf_sampler = IMF_sampler(
                maxtime=self.model_time,
                dt=self.model_dt,
                nstar_mean=1.311,  # these are values hard-coded from Azton's FBNet
                nstar_std=0.352,
            )

            # regression weights for radius prediction
            candidates = [
                f"./StarNetRuntime/resources/regression_weights_{self.model_time}.txt",
                f"./resources/regression_weights_{self.model_time}.txt",
            ]
            weights_path = next((p for p in candidates if os.path.exists(p)), None)
            if weights_path is None:
                raise FileNotFoundError(
                    f"regression weights not found in {candidates}"
                )
            self.model_weights = torch.from_numpy(np.loadtxt(weights_path)).float()

            self.idx_edges = np.zeros(3)
            self.idx_edges[0] = 10 ** (self.imf_sampler.nstar_mean - self.imf_sampler.nstar_std)
            self.idx_edges[1] = 10 ** (self.imf_sampler.nstar_mean)
            self.idx_edges[2] = 10 ** (self.imf_sampler.nstar_mean + self.imf_sampler.nstar_std)

        self.levels = cfg["fbnet"].getint("simulation_max_level")
        self.top_grid_dims = cfg["starnet"].getfloat("simulation_top_dimensions")  # not used anywhere
        self.radius_modifier = cfg["fbnet"].getfloat("radius_modifier")




    def find_feedback_center(self, labels):
        """Mean voxel index of identified positive star-forming cells."""
        d = labels.size(-1)
        screen = torch.from_numpy(np.mgrid[0:d, 0:d, 0:d]).cpu()
        screen = (screen * labels.cpu()).numpy()
        mean_cells = np.array([
            int(screen[i].sum() / max((screen[i] != 0).sum(), 1)) for i in [0, 1, 2]
        ])
        return mean_cells



    def feedback_uniform_sphere(self, ds, labels, dx, left_edge):
        """IMF samples -> metal mass + radius regression -> center."""
        x, metals, stellar_mass = self.imf_sampler.generate_samples(ds, 1)
        x = np.append(np.array([1]), x)

        idx = 0
        if x[:8].sum() > self.idx_edges[0]: idx += 1
        if x[:8].sum() > self.idx_edges[1]: idx += 1
        if x[:8].sum() > self.idx_edges[2]: idx += 1

        # predict radius from linear model
        radius_log10 = max(
            float(torch.matmul(torch.from_numpy(x).float(), self.model_weights[idx])),
            np.log10(0.25),
        )

        center_ind = self.find_feedback_center(labels)
        center_pos = left_edge + ds.arr((center_ind + 0.5) * dx, "unitary")
        return center_pos, radius_log10, metals, stellar_mass

    def forward(self, ds, labels, sample_dx, left_edge):
        """
        Predict feedback for a positive segmentation region.
        Returns: center (YTArray), log10(radius_kpccm), M_Z_Msun, M_star_Msun

        Deposition is NOT performed. For snapshot mode the prediction is
        logged. For in-situ via libyt, a separate deposition routine reads
        the prediction list and writes into libyt.grid_data[gid][field].
        """
        if self.FBmethod.lower() == "uniform_sphere":
            return self.feedback_uniform_sphere(ds, labels, sample_dx, left_edge)

        # deposition is NOT done here
        raise NotImplementedError(f"{self.FBmethod} not a valid feedback method")
