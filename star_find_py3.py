"""
star_find_py3.py - StarNet's StarFind ported to Python 3 / yt

Phase 1A snapshot mode: no Enzo runtime dependency. Reads from a yt dataset
loaded from a saved snapshot. The covering grid construction (the original
get_grid_volume() with manual AMR hierarchy walking) is replaced with yt's
ds.smoothed_covering_grid, which handles AMR boundaries correctly.

Key changes from Azton's Py2 version:
- removed `from enzo import` and `from mpi4py import`
- Py2 print statements -> print(), or log passed from inline function
- get_grid_volume -> build_covering_grid (uses yt)
- load .jtpt JIT checkpoints directly; no class hierarchy needed at runtime
- conversion factors via yt unit system, matching Azton's CGS convention
- single-process; MPI parallelism deferred to in-situ phase
"""

import os
import torch
import numpy as np
from copy import deepcopy

from StarNetDataLoader import StarnetDataLoader







class StarFind:
    def __init__(self, cfg, logger=None):
        # modded so that starfind takes in logger, if not print statement
        self._log = logger if logger is not None else print

        self.Loader = StarnetDataLoader(cfg)
        field_list = cfg["starfind"]["field_list"].split(",")
        self.field_list = [str(f).strip() for f in field_list]
        self.max_refinement_level = cfg["starfind"].getint("maximum_refinement_level")

        # map derived field names -> enzo BaryonField names
        self.sim_fields = deepcopy(self.field_list)
        tmap = {
            "density": "Density",
            "H2_p0_density": "H2I_Density",
            "total_energy": "TotalEnergy",
        }
        for i, f in enumerate(self.sim_fields):
            if f in tmap:
                self.sim_fields[i] = tmap[f]
        if "velocity_divergence" in self.sim_fields:
            self.sim_fields.remove("velocity_divergence")
            self.sim_fields += ["x-velocity", "y-velocity", "z-velocity"]
        if "metals" in self.sim_fields:
            self.sim_fields.remove("metals")
            self.sim_fields += ["SN_Colour", "Metal_Density"]

        self.stage1, self.stage2 = self.load_models(cfg)


        # =====
        # Azton's get_grid_volume hardcodes the sample width to
        # 10 kpccm (line: width = ds.quan(10, 'kpccm')). The config
        # 'region_width' value is read into his object but never actually
        # used for the sample geometry -- the hardcode supersedes it. We
        # match the operative behavior: hardcode 10 kpccm. This is the
        # physical size of the 64^3 sample volume (10 kpccm / 64 = 156 pc
        # per cell, matching region_target_dx = 160 pc).
        # self.sample_width_kpccm = 10.0
        # config value kept only for reference/secondary use; not the
        # operative sample size
        self.cfg_region_width = cfg["starfind"].getfloat("region_width")
        self.sample_width_kpccm = self.cfg_region_width


        self.region_dx_pccm = cfg["starfind"].getfloat("region_target_dx")
        self.region_dim = cfg["starfind"].getint("region_dims")
        self.top_grid_dims = cfg["starnet"].getfloat("simulation_top_dimensions")  # not used anywhere
        self._warned_missing = set()






    def load_models(self, cfg):
        """Load .jtpt JIT-compiled classifier and segmenter."""
        stage1_name = cfg["starfind"]["stage1_model"]
        stage2_name = cfg["starfind"]["stage2_model"]
        cp_dir = cfg["s1_config"]["checkpoint_path"]

        # try a few likely locations
        candidates_s1 = [
            os.path.join(cp_dir, f"{stage1_name}.jtpt"),
            os.path.join(os.path.dirname(cp_dir), "model_checkpoints", f"{stage1_name}.jtpt"),
            f"./model_checkpoints/{stage1_name}.jtpt",
        ]
        candidates_s2 = [
            os.path.join(cp_dir, f"{stage2_name}.jtpt"),
            os.path.join(os.path.dirname(cp_dir), "model_checkpoints", f"{stage2_name}.jtpt"),
            f"./model_checkpoints/{stage2_name}.jtpt",
        ]
        jtpt_s1 = next((p for p in candidates_s1 if os.path.exists(p)), None)
        jtpt_s2 = next((p for p in candidates_s2 if os.path.exists(p)), None)
        if jtpt_s1 is None or jtpt_s2 is None:
            raise FileNotFoundError(
                f"JIT checkpoints not found.\n  s1: {candidates_s1}\n  s2: {candidates_s2}\n"
                f"Edit checkpoint_path in config, or run convert_torch_model.ipynb."
            )



        # load the stage 1 and 2 models
        self._log(f"  loading classifier: {jtpt_s1}")
        self._log(f"  loading segmenter:  {jtpt_s2}")
        class_model = torch.jit.load(jtpt_s1, map_location="cpu")
        seg_model = torch.jit.load(jtpt_s2, map_location="cpu")
        class_model.eval()
        seg_model.eval()
        return class_model, seg_model
        # skips the GPU parallel loading on starnet



    def build_covering_grid(self, ds, region_left_edge, level=None):
        """
        Construct a 64^3 covering grid at the StarFind target resolution.

        Replaces Azton's manual hierarchy walking with yt's smoothed_covering_grid.
        Returns (dict_of_torch_tensors, dx_quantity) or None if invalid region.

        Field conventions follow Azton's get_grid_volume:
        - densities (Density, *_Density, SN_Colour) in g/cm^3 (CGS)
        - velocities in cm/s
        - TotalEnergy in code units (no conversion)
        """
        if level is None:
            level = self.max_refinement_level

        # sample width is 10 kpccm (see __init__ note), in code units
        sample_width = ds.quan(self.sample_width_kpccm, "kpccm").to("unitary").d
        rle = np.asarray(region_left_edge)
        rre = rle + sample_width
        if np.any(rle < 0) or np.any(rre > 1):
            return None

        try:
            cg = ds.smoothed_covering_grid(
                level=level,
                left_edge=ds.arr(rle, "unitary"),
                dims=[self.region_dim] * 3,
            )
        except Exception as e:
            # if covering grid fails on local rank
            self._log(f"  covering_grid failed at le={rle}: {e}")
            return None

        out = {}
        for field in self.sim_fields:
            try:
                if field == "TotalEnergy":
                    arr = cg[("enzo", "TotalEnergy")].d  # code units (unchanged)
                elif field == "SN_Colour":
                    try:
                        # try to build the covering grid for SN_color
                        arr = cg[("enzo", "SN_Colour")].in_units("g/cm**3").d
                    except Exception:
                        # if this doesn't exist just use metal density
                        arr = cg[("enzo", "Metal_Density")].in_units("g/cm**3").d
                        if "SN_Colour" not in self._warned_missing:  # warn once only
                            self._log("  SN_Colour absent, substituting Metal_Density for inference")
                            self._warned_missing.add("SN_Colour")
                elif field == "Density" or field.endswith("_Density"):
                    arr = cg[("enzo", field)].in_units("g/cm**3").d
                elif "velocity" in field:
                    arr = cg[("enzo", field)].in_units("cm/s").d
                else:
                    arr = cg[("enzo", field)].d
            
            # if the field doesn't exist
            except Exception as e:
                # field missing — typical for sims without chemistry (MultiSpecies=0)
                # or Pop III tracking (no SN_Colour). For Phase 1A pipeline validation,
                # fill optional fields with zeros so inference still runs. Density is
                # required and its absence is fatal.
                if field == "Density":
                    self._log(f"  REQUIRED field Density unavailable: {e}")
                    return None
                if field not in self._warned_missing:
                    self._log(f"  field {field} unavailable, filling with zeros ")
                    self._log(f"(suppressing further warnings for this field)")
                    self._warned_missing.add(field)
                arr = np.zeros((self.region_dim,) * 3, dtype=np.float32)
                
            out[field] = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))

        dx_unitary = sample_width / self.region_dim
        return out, ds.quan(dx_unitary, "unitary")





    def false_weak_positive_check(self, seg_classes):
        if seg_classes.sum() == 0:
            return "nothing"
        if seg_classes.sum() <= 4:
            return "weak"
        if (seg_classes[0, :3, :3, :3].sum() > 0
                or seg_classes[0, -2:, -2:, -2:].sum() > 0):
            return "edge"
        return "awesome"




    def forward(self, ds, region):
        """Classifier first, then segmenter. Returns pvox (1,D,D,D) or None."""
        sample = self.Loader(ds, region)

        with torch.no_grad():
            pred = self.stage1(sample)
            _, classes = torch.max(torch.softmax(pred, 1), 1)
        if classes.sum() == 0:
            return None

        with torch.no_grad():
            pvox = torch.softmax(self.stage2(sample), 1)
            _, pvox = torch.max(pvox, 1)

        quality = self.false_weak_positive_check(pvox)
        if quality in ("nothing", "weak", "edge"):
            return None
        return pvox