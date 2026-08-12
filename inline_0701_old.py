"""
inline.py - libyt in-situ entry point for StarNet Pop III feedback
            (distributed-inference version)
 
Called by Enzo every CycleSkipLibytCall cycles, on EVERY MPI rank
simultaneously.
 
CRITICAL DESIGN CONSTRAINT (learned the hard way via deadlock):
  ds.covering_grid() under libyt is a COLLECTIVE operation. Every rank
  MUST participate in every covering_grid call, in the same order. You
  CANNOT have rank 0 alone build covering grids while other ranks wait
  at a barrier -- that deadlocks (rank 0 waits for a collective the
  others never join).
 
So the architecture is:
  1. ALL ranks: build ds = yt_libyt.libytDataset()
  2. ALL ranks: walk the SAME list of level-min_level grids, build the
     SAME sequence of tile covering_grids together (collective, synced).
     For each tile, ONLY the rank assigned to it (tile_index % nranks ==
     rank) runs the torch StarFind inference. Other ranks build the
     covering grid (required for the collective) but skip inference.
  3. ALL ranks: allgather the per-rank event lists into a global list.
  4. ALL ranks: each rank deposits the portion of each event sphere that
     falls in its locally-owned grids (libyt.grid_data writes).
  5. RANK 0: log events + save projection.
 
This distributes the expensive torch inference across all ranks while
keeping the collective covering_grid calls synchronized. The covering
grid cost is shared (unavoidable -- it's collective); the inference cost
is divided by nranks.
 
Time cadence: a guard ensures StarNet only fires every
MIN_TIME_MYR_BETWEEN_CALLS of sim time.
"""
 
import os
import sys
import time
import traceback
 
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*align should be passed.*")
# numpy 2.4 VisibleDeprecationWarning from torch's unpickler
# suppress warnings from pytorch model loading
try:
    warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
except AttributeError:
    pass
 
import yt
import yt_libyt
import libyt
from mpi4py import MPI
 
yt.enable_parallelism()
yt.set_log_level(40)
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
NRANKS = COMM.Get_size()
 
 
# =====
# configuration
# =====
CONFIG_PATH = "./ex_Pop3Net.conf"
 
RUN_DIR = "/work2/11056/zwu/frontera/enzo_gh/data/251110/ic512_b_starnetlibyt"
LOG_DIR = os.path.join(RUN_DIR, "starnet_insitu_logs")
FIG_DIR = os.path.join(RUN_DIR, "starnet_insitu_figs")
EVENT_LOG = os.path.join(LOG_DIR, "starnet_events_insitu.log")
RUN_LOG = os.path.join(LOG_DIR, "starnet_insitu_run.log")
 
MIN_TIME_MYR_BETWEEN_CALLS = 5.0
MAX_REDSHIFT_FOR_STARNET = 35.0
RADIUS_MODIFIER_FALLBACK = 0.4
 
# verbose per-rank logging for debugging. set False once stable to cut IO.
DEBUG_PERRANK = True
 
 
# =====
# module-level state, persists across libyt calls
# =====
_call_count = 0
_initialized = False
_sf = None              # StarFind instance -- now loaded on ALL ranks
_fb = None              # FBNet instance -- now loaded on ALL ranks
_cfg = None
_min_level = None
_last_run_time_myr = None
_radius_modifier = RADIUS_MODIFIER_FALLBACK
 
_last_events = []       # most recent deposition's events, for plotting
_last_event_call = -1   # call number when those events were deposited
 
 
def printlog(text):
    """Run log, root only."""
    if RANK != 0:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(RUN_LOG, "a") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
 
 
def dbg(text):
    """Per-rank debug log with timestamp. Off when DEBUG_PERRANK=False."""
    if not DEBUG_PERRANK:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"rank_{RANK:04d}.log")
    with open(path, "a") as f:
        f.write(f"[{time.time():.3f}] {text}\n")
        f.flush()
 
 
def log_event(call_num, z, t_myr, grid_idx, center, log_r, m_z, m_star):
    """
    Append event in the snapshot-runner format so plot_events_on_projection.py works on it.
    deposits to starnet_insitu_logs/starnet_events_insitu.log by default
    Ran on ROOT ONLY.
    """
    if RANK != 0:  # rank protection
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(EVENT_LOG):
        with open(EVENT_LOG, "w") as f:
            f.write("# z t_Myr grid_idx cx cy cz log10_r_kpccm M_Z_Msun M_star_Msun\n")
    with open(EVENT_LOG, "a") as f:
        f.write(
            f"{z:.4f} {t_myr:.3f} {grid_idx} "
            f"{float(center[0]):.8f} {float(center[1]):.8f} {float(center[2]):.8f} "
            f"{log_r:.4f} {m_z:.3f} {m_star:.3f}\n"
        )
        
 
 
# =====
# one-time init: ALL ranks load the models now (every rank does inference
# on its assigned tiles, so every rank needs the torch models)
# =====
def _initialize():
    global _initialized, _sf, _fb, _cfg, _min_level, _radius_modifier
    import configparser as cp
 
    _cfg = cp.ConfigParser()
    _cfg.read(CONFIG_PATH)
    _min_level = _cfg["starnet"].getint("minimum_level_for_tiling")
    try:
        # load the radius modifier, if none than use 0.4
        _radius_modifier = _cfg["fbnet"].getfloat("radius_modifier")
    except Exception:
        _radius_modifier = RADIUS_MODIFIER_FALLBACK
 
 
    # =====
    # import STARNET libraries
    # fb_net_py3, IMF_Sampler, star_find_py3, and StarNetDataLoader have to be present in the directory
    from star_find_py3 import StarFind
    from fb_net_py3 import FBNet
 
    dbg("init: loading models")
    t0 = time.time()
 
    # starfind and fbnet objects
    _sf = StarFind(_cfg, logger=dbg)  # give starfind the log function
    _fb = FBNet(_cfg)
    dbg(f"init: models loaded in {time.time()-t0:.1f}s")
 
    _initialized = True
 
 
# =====
# distributed inference: ALL ranks walk the same grid/tile list together.
# Every rank builds every tile's covering grid (collective requirement),
# but only the assigned rank runs torch inference on each tile.
# returns this rank's local event list and the total tile count.
# =====
def _run_starnet_inference_distributed(ds):
    # ==
    # note here that this region_width is different from the one in StarFind, which uses the 10kpc from the param file as the SIZE of the volume
    # this "width" is the separation / stride for tile-stepping
    # ==
    # region_width = 1.0 / float(ds.domain_dimensions[0])  # region_width = 1 / top grid dimensions
    
    
    # tile stride = the sample width (10 kpccm), in code units. This makes
    # tiling gap-free and box-size-independent. Azton used 1/TopGridDimensions
    # as the stride, which equals 10 kpccm ONLY in the Phoenix box where one
    # root cell was chosen to be 10 kpc; that coincidence does not hold for
    # the Hicks box, so we use the physical sample width directly (matching
    # the hardcoded 10 kpccm sample size in build_covering_grid).
    tile_stride = float(
        ds.quan(_sf.sample_width_kpccm, "kpccm").to("unitary").d
    )
    n_grids = len(ds.index.grids)
 
    local_events = []
    my_tile_log = []        # (tile_index, le, segpos_flag) for tiles this rank owns
    tile_index = 0          # global tile counter, identical on all ranks
    n_my_tiles = 0
    n_my_segpos = 0
    t0 = time.time()
 
    for grid_idx in range(n_grids):
        level = int(ds.index.grid_levels[grid_idx][0])
        if level != _min_level:
            continue
 
        # split up the grid
        le = np.asarray(ds.index.grid_left_edge[grid_idx])
        re = np.asarray(ds.index.grid_right_edge[grid_idx])
        gw = re - le
 
        # split up the box by tile_stride
        n_x = max(int(gw[0] / tile_stride), 1)
        n_y = max(int(gw[1] / tile_stride), 1)
        n_z = max(int(gw[2] / tile_stride), 1)
 
        x_e = np.linspace(le[0], re[0] - tile_stride, n_x) if n_x > 1 else np.array([le[0]])
        y_e = np.linspace(le[1], re[1] - tile_stride, n_y) if n_y > 1 else np.array([le[1]])
        z_e = np.linspace(le[2], re[2] - tile_stride, n_z) if n_z > 1 else np.array([le[2]])
 
        for xi in x_e:
            for yi in y_e:
                for zi in z_e:
                    # which rank owns inference for this tile
                    owner = tile_index % NRANKS
                    tile_le = np.array([xi, yi, zi])
 
                    # =====
                    # ALL ranks build the covering grid (collective!).
                    # build_covering_grid internally calls
                    # ds.smoothed_covering_grid which is collective; every
                    # rank must reach it in lockstep or we deadlock.
                    # =====
                    result = _sf.build_covering_grid(ds, tile_le)
                    if tile_index % 10 == 0:
                        dbg(f"    walk: tile {tile_index}, t={time.time()-t0:.1f}s")
 
 
                    # only the owning rank runs torch inference
                    if owner != RANK:
                        tile_index += 1
                        continue
 
                    if result is None:
                        my_tile_log.append((tile_index, xi, yi, zi, "none"))
                        tile_index += 1
                        continue
                    cg, dx = result
                    n_my_tiles += 1
 
 
                    # =====
                    # forward pass I
                    # whether or not there is pop III star
 
                    pvox = _sf.forward(ds, cg)
                    if pvox is None:
                        my_tile_log.append((tile_index, xi, yi, zi, "neg"))
                        tile_index += 1
                        continue
                    n_my_segpos += 1
                    my_tile_log.append((tile_index, xi, yi, zi, "POS"))
 
 
                    # =====
                    # forward pass II
                    # which voxel contains pop III bubble
 
                    center, log_r, m_z, m_star = _fb.forward(
                        ds, pvox, dx, ds.arr(tile_le, "unitary")
                    )
 
 
                    # =====
                    # enrichment suppression (ported from azton's
                    # deposit_or_find_volume): reject if the predicted
                    # center is already above critical metallicity. Pop III
                    # does not form in enriched gas. Uses Metal_Density
                    # since SN_Colour is absent; 0.01295 = solar metal mass
                    # fraction, 3.1e-6 = critical metallicity in Zsun.
                    # sample the metals/density from the covering grid at
                    # the segmenter's center-of-mass voxel.
                    # =====
                    cvox = _fb.find_feedback_center(pvox)  # (i,j,k) in 64^3
                    ci, cj, ck = [int(np.clip(c, 0, _sf.region_dim - 1)) for c in cvox]
                    metal_at_c = float(cg["Metal_Density"][ci, cj, ck])  # uses metal density instead of both Metal_Density and SN_colour in FBNet.deposit_or_find_volume()
                    dens_at_c = float(cg["Density"][ci, cj, ck])
                    Z_over_Zsun = metal_at_c / dens_at_c / 0.01295 if dens_at_c > 0 else 0.0
                    if Z_over_Zsun > 3.1e-6:
                        my_tile_log.append((tile_index, xi, yi, zi, "enriched_skip"))
                        tile_index += 1
                        continue
 
 
                    # append the grid ID, position, radius, metal, and stellar masses
                    local_events.append({
                        "grid_idx": grid_idx,
                        "center": np.array([
                            float(center[0]), float(center[1]), float(center[2])
                        ]),
                        "log_r": float(log_r),
                        "m_z": float(m_z),
                        "m_star": float(m_star),
                    })
 
                    tile_index += 1
 
    dt = time.time() - t0
 
    # tally up number of tiles
    # should be the same number for all MPI ranks
    dbg(
        f"inference: total_tiles={tile_index}, my_tiles={n_my_tiles}, "
        f"my_segpos={n_my_segpos}, my_events={len(local_events)}, "
        f"in {dt:.1f}s"
    )
 
 
    # print to log file the results
    # detail of which global tile indices this rank handled and outcome
    for ti, xi, yi, zi, outcome in my_tile_log:
        dbg(
            f"    tile #{ti}: le=({xi:.4f},{yi:.4f},{zi:.4f}) -> {outcome}"
        )
    return local_events, tile_index
 
 
# =====
# target gas energy (code units) for T=1e4 K
# specific energy: T = (gamma-1)*mu*GasEnergy_code*TempUnits
# TempUnits = m_H * VelocityUnits^2 / k_B; VelocityUnits = LengthU/TimeU
# =====
def _target_gas_energy_code(ds):
    # v_units = float((ds.length_unit / ds.time_unit).in_units("cm/s").d)
    v_units = float(ds.velocity_unit.in_units("cm/s").d)  # using velocity units here instead
    m_h = 1.673e-24
    k_b = 1.381e-16
    temp_units = (m_h * v_units ** 2) / k_b
    
    # the desired teperature is 1E4 Kelvin
    desired_temp = 1.0e4
    desired_energy = desired_temp / 0.6667 / 1.22 / temp_units
    return desired_energy
 
 
# =====
# deposition: validated deposit_sphere, per-event metal mass + radius
# returns n_cells, n_grids, cells_per_level
# =====
def deposit_sphere(ds, center_code, radius_code, metal_mass_msun,
                   target_gas_energy_code):
    n_cells = 0
    n_grids = 0
    cells_per_level = {}
 
    # initialize before the loop so they always exist even when this rank
    # owns no cells in this event's sphere (loop body never runs)
    sample_before = None
    sample_after = None
 
    v_sphere_code = (4.0 / 3.0) * np.pi * radius_code ** 3
    mass_unit_msun = float(ds.mass_unit.in_units("Msun").d)
    metal_mass_code = metal_mass_msun / mass_unit_msun
    metal_density_code = metal_mass_code / v_sphere_code
 
    for gid in libyt.grid_data.keys():
        grid_idx = gid - 1
 
        # grab the edges of the boundary owned by the rank
        level = int(ds.index.grid_levels[grid_idx][0])
        le = np.asarray(ds.index.grid_left_edge[grid_idx])
        re = np.asarray(ds.index.grid_right_edge[grid_idx])
 
 
        # deposit IF THE CELL BELONGS TO THE RANK
        if any(center_code[d] - radius_code > re[d] for d in range(3)):
            continue
        if any(center_code[d] + radius_code < le[d] for d in range(3)):
            continue
 
        metal = libyt.grid_data[gid]["Metal_Density"]
        gas_energy = libyt.grid_data[gid]["GasEnergy"]
        total_energy = libyt.grid_data[gid]["TotalEnergy"]
 
        shape = metal.shape
        NG = libyt.param_user["NumberOfGhostZones"]
        active_zyx = np.array(shape) - 2 * NG
        cw_x = (re[0] - le[0]) / active_zyx[2]
        cw_y = (re[1] - le[1]) / active_zyx[1]
        cw_z = (re[2] - le[2]) / active_zyx[0]
 
        ii, jj, kk = np.indices(shape)
        x_centers = le[0] + (kk + 0.5 - NG) * cw_x
        y_centers = le[1] + (jj + 0.5 - NG) * cw_y
        z_centers = le[2] + (ii + 0.5 - NG) * cw_z
 
        dist_sq = (
            (x_centers - center_code[0]) ** 2
            + (y_centers - center_code[1]) ** 2
            + (z_centers - center_code[2]) ** 2
        )
        inside = dist_sq < radius_code ** 2
        n_inside = int(inside.sum())
        if n_inside == 0:
            continue
 
 
 
 
        # ==========
        # The deposit block
 
 
        # =====
        # capture before/after for the first cell inside the sphere
        # proves the write landed in BaryonField for this event
        # =====
        inside_idx = np.argwhere(inside)
        if len(inside_idx) > 0:
            si, sj, sk = inside_idx[0]
            metal_before = float(metal[si, sj, sk])
            ge_before = float(gas_energy[si, sj, sk])
 
 
        ge_old = gas_energy[inside].copy()
        metal[inside] += metal_density_code
        gas_energy[inside] = target_gas_energy_code
        total_energy[inside] = (
            total_energy[inside] - ge_old + target_gas_energy_code
        )
 
        if len(inside_idx) > 0:
            metal_after = float(metal[si, sj, sk])
            ge_after = float(gas_energy[si, sj, sk])
            sample_before = (metal_before, ge_before)
            sample_after = (metal_after, ge_after)
 
        n_cells += n_inside
        n_grids += 1
        cells_per_level[level] = cells_per_level.get(level, 0) + n_inside
 
    # return n_cells, n_grids, cells_per_level  # the old return statement that does not include before / after
    return n_cells, n_grids, cells_per_level, sample_before, sample_after
 
 
 
 
# function to save a projection plot
# runs EVERY time libyt is called
def _save_projection(ds, z, call_num, events_to_mark, fired_this_call):
    os.makedirs(FIG_DIR, exist_ok=True)
    prj = yt.ProjectionPlot(
        ds, "z", ("gas", "metal_density"),
        center=ds.domain_center,
        width=ds.domain_width.in_units("Mpccm/h")[0],
    )
    prj.set_log(("gas", "metal_density"), True)
    prj.set_axes_unit("Mpccm/h")
    tag = "DEPOSIT" if fired_this_call else "evolve"
    prj.annotate_title(
        f"metal density | call #{call_num} | z={z:.2f} | {tag}"
    )
 
 
 
    # if STARNET has been called THIS CYCLE, print a star
    # only mark stars on the call where they were freshly deposited
    if fired_this_call:
        for ev in events_to_mark:
            c = ds.arr(ev["center"], "unitary")
            prj.annotate_marker(
                c, marker="*",
                plot_args={"color": "red", "s": 200,
                           "edgecolors": "white", "linewidths": 0.6},
            )
    if RANK == 0:
        prj.save(os.path.join(FIG_DIR, f"metal_call{call_num:04d}"))
 
 
 
 
 
 
 
 
 
 
# =========================
# =========================
# =========================
# =========================
# =========================
 
# =====
# the function Enzo/libyt calls
# =====
 
 
 
 
 
def yt_inline_func():
    # global variables
    # these are persistent from the first time libyt is called in the simulation
    global _call_count, _last_run_time_myr, _last_events, _last_event_call
    _call_count += 1
 
 
    # everything in a try block for debugging
    try:
        dbg(f"call #{_call_count} ENTER")
        if not _initialized:
            _initialize()  # load model weights, etc. takes a while but only ran once
 
 
        # EVERY rank loads libyt dataset
        ds = yt_libyt.libytDataset()  # build the dataset
 
 
        # # ==========
        # # DEBUG
        # # ==========
        # # in case ds doesn't have those attributes
        # for nm in ("length_unit","time_unit","velocity_unit","mass_unit","density_unit"):
        #     try:
        #         dbg(f"UNITCHK {nm} = {getattr(ds, nm).in_cgs()!r}")
        #     except:
        #         dbg(f"UNITCHK {nm} **FAILED**, no such attributes")
        # # velocity units
        # vu = (ds.length_unit/ds.time_unit).in_units("cm/s")
        # dbg(f"UNITCHK z={float(ds.current_redshift):.4f} temp_units={1.673e-24*float(vu.d)**2/1.381e-16:.6e}")
 
 
        z = float(ds.current_redshift)
        t_myr = float(ds.current_time.in_units("Myr").d)
        dbg(f"call #{_call_count} ds ready z={z:.3f} t={t_myr:.2f}")
        printlog(
            f"\n=== libyt call #{_call_count} | z={z:.3f} | "
            f"t={t_myr:.2f} Myr | n_grids={len(ds.index.grids)} | "
            f"max_level={ds.index.max_level} | nranks={NRANKS} ==="
        )
 
 
 
        # =====
        # decide whether to run StarNet (rank 0 decides, bcast)
        # =====
 
        run_starnet = False
        if RANK == 0:  # ONLY rank 0
            if z > MAX_REDSHIFT_FOR_STARNET:  # only if lower than redshift set in .conf file
                run_starnet = False
            elif _last_run_time_myr is None:  # run every 5 Myr & run on the first time called
                run_starnet = True
            elif (t_myr - _last_run_time_myr) >= MIN_TIME_MYR_BETWEEN_CALLS:
                run_starnet = True
 
        # broadcast this to EVERY single MPI rank
        run_starnet = COMM.bcast(run_starnet, root=0)
        dbg(f"call #{_call_count} run_starnet={run_starnet}")
 
 
        # "fired" tracks whether THIS call did a deposition,
        # so the projection at the end knows whether to draw star markers on the plot, see _save_projection()
        # regardless, one frame is saved every libyt call for the animation!
        fired = False
 
 
 
 
        # ==========
        # ==========
        if run_starnet:
            # =====
            # STEP 1
            # ALL ranks participate in the collective covering_grid calls
            # Tiles are assigned to every rank in a crude modulo method
            #
            # STEP 2
            # Each runs torch inference only on its assigned tiles
            # (some may end up with 0 or more than 1, but roughly load-balanced)
            # =====
            dbg(f"call #{_call_count} entering distributed inference")
            local_events, total_tiles = _run_starnet_inference_distributed(ds)
            dbg(f"call #{_call_count} done inference, my_events={len(local_events)}")
            if RANK == 0:
                _last_run_time_myr = t_myr
 
 
 
            # =====
            # STEP 3
            # Gather all event lists from ALL ranks into a single list
            # =====
            dbg(f"call #{_call_count} before allgather")
            gathered = COMM.allgather(local_events)  # gathered event list from ALL ranks
            dbg(f"call #{_call_count} after allgather")
            
            events = [ev for sublist in gathered for ev in sublist]  # unwrap
 
            printlog(
                f"  inference: {total_tiles} total tiles walked, "
                f"{len(events)} events found across all ranks"
            )
 
 
 
 
 
            # =====
            # only deposit if we actually found events. all ranks evaluate
            # this identically from the same shared events list
            # =====
            if len(events) > 0:
 
 
 
                # =====
                # STEP 4: every rank deposits each event's local portion
                # print to both rank- and all-rank- log files
                # =====
 
                local_cells = 0
                local_grids = 0
                local_levels = {}
 
 
                # calculate energy
                tge = _target_gas_energy_code(ds)
 
                # for each event
                for ev in events:
                    # radius multiplied by Azton's multiplier, defaults to 0.4
                    # **unclear, needs to check with him
                    r_kpccm = (10 ** ev["log_r"]) * _radius_modifier
                    radius_code = float(ds.quan(r_kpccm, "kpccm").to("unitary").d)
 
 
                    # deposit the sphere
                    # grav the before / after
                    n_c, n_g, lvls, s_before, s_after = deposit_sphere(
                        ds, ev["center"], radius_code, ev["m_z"], tge
                    )
                    local_cells += n_c  # append the number cells & grids changed locally
                    local_grids += n_g
                    
                    for lvl, n in lvls.items():
                        local_levels[lvl] = local_levels.get(lvl, 0) + n
                    # log first deposited cell's before/after on whichever rank wrote it
 
                    # =====
                    # write verification to the individual log
                    if s_before is not None:
                        dbg(
                            f"  DEPOSIT VERIFY ev@({ev['center'][0]:.3f},"
                            f"{ev['center'][1]:.3f},{ev['center'][2]:.3f}): "
                            f"Metal {s_before[0]:.3e}->{s_after[0]:.3e}, "
                            f"GasE {s_before[1]:.3e}->{s_after[1]:.3e}"
                        )
 
 
                # print to rank log
                dbg(f"call #{_call_count} deposition done, local_cells={local_cells}")
                
                # print to all rank log
                global_cells = COMM.allreduce(local_cells, op=MPI.SUM)
                per_rank = COMM.gather(local_cells, root=0)
                all_levels = COMM.gather(local_levels, root=0)
                printlog(f"  DEPOSITED {len(events)} events: global_cells={global_cells}")
                # print which ranks had deposition
                if per_rank is not None:
                    nonzero = [c for c in per_rank if c > 0]
                    printlog(
                        f"  per-rank cells (nonzero ranks): {nonzero} "
                        f"(sum={global_cells})"
                    )
                # print how many cells per level
                if all_levels is not None:
                    merged = {}
                    for d in all_levels:
                        for lvl, n in d.items():
                            merged[lvl] = merged.get(lvl, 0) + n
                    breakdown = ", ".join(f"L{lvl}={merged[lvl]}" for lvl in sorted(merged))
                    printlog(f"  per-level cells: {breakdown}")
 
 
 
 
                # =====
                # STEP 5: rank 0 logs events. cache events for the marker
                # plot and flag that THIS call fired a deposition
                # This logging is ONLY DONE ON RANK 0
                # =====
 
                if RANK == 0:
                    for ev in events:
                        log_event(
                            _call_count, z, t_myr, ev["grid_idx"],
                            ev["center"], ev["log_r"], ev["m_z"], ev["m_star"]
                        )
                
                # set the local to last global variable
                _last_events = events
                _last_event_call = _call_count
                fired = True  # indicate to plotting that there was STARNET deposition this cycle
 
            else:  # if STARNET ran but produced no events
                printlog("  no events this call")
 
        else:  # if STARNET did not fullfill run condition
            printlog(
                f"  skipping StarNet (z={z:.2f}, "
                f"last_run={_last_run_time_myr})"
            )
 
        
        
        
        
        
        # =====
        # ALWAYS save a projection frame, EVERY libyt call
        # Star markers are drawn only when fired=True (a fresh deposition this call)
        # All ranks reach this together; the projection is a collective operation,
        # so it must NOT be inside a rank-gated branch
        # =====
 
        _save_projection(ds, z, _call_count, _last_events, fired)
        dbg(f"call #{_call_count} projection saved (fired={fired}), COMPLETE")
 
    except Exception as e:
        dbg(f"call #{_call_count} EXCEPTION {type(e).__name__}: {e}")
        printlog(
            f"!!! EXCEPTION call #{_call_count}: {type(e).__name__}: {e}"
        )
        printlog(traceback.format_exc())
        raise