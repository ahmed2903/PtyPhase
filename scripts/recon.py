"""
Phase retrieval of real data (a saved ptyxplore Bragg ROI) with the per-streak 2D engine
(ptyphase.runner.FourierPtychoEngine): one 2D object spectrum per streak, flat q_z slices,
no 3D support. The 3D object is the centred IFFT of the slice stack along q_z.

The pupil is shared by all streaks by default. With --independent_pupils every streak
gets its own pupil and is reconstructed as a separate problem; --independent_from n
runs the first n iterations with the shared pupil and then copies it to every streak.

The input must already be split into streaks (e.g. ptyxplore make_qz_streaks). Streaks
are placed on the uniform q_z grid as in recon_ewald.py (same helpers). Grid points without
a streak become empty slices, kept at 0 (nothing constrains them here).

    python recon.py --input roi.h5 --steps dm:300:0.9 er:50
    python recon.py --input roi.h5 --steps dm:300 er:50 --independent_pupils --independent_from 100

Kernels: dm, raar, raar_standard, er (ptyphase.algorithms). Zernike projection is off by
default (--zernike_steps 0). See `python recon.py --help` for everything else.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import h5py
import hdf5plugin

import ptyxplore.loader as db
import ptyxplore.roi_ops as dp
from ptyxplore.kvectors import fit_global_G, kins_from_G, qz_from_G

from ptyphase.runner import FourierPtychoEngine
from ptyphase.algorithms import DM, RAAR, RAAR_standard, ER
from ptyphase.utils_pr import split_psize

# shared with the Ewald script, so both place streaks and initialise the same way
from recon_ewald import (optional_float, flat, estimate_dqz_from_pixels, place_streaks,
                         initial_pupil, initial_slices)

time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

KERNELS = ("dm", "raar", "raar_standard", "er")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Per-streak Fourier ptychography with shared or independent pupils (ptyphase.runner).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    g = parser.add_argument_group("data")
    g.add_argument("--input", type=str, required=True,
                   help="Saved Bragg ROI (ptyxplore ROI.save_data), already split into streaks.")
    g.add_argument("--pupil_file", type=str, default=None,
                   help="Saved pupil ROI (its kins set the pupil grid). "
                        "Default: pupil_scan-<scan>.h5 next to --input.")
    g.add_argument("--axis_flip", type=str, default="xy",
                   choices=["xy", "x_neg_y", "neg_x_y", "neg_x_neg_y"])

    # algorithm
    g = parser.add_argument_group("algorithm")
    g.add_argument("--steps", type=str, nargs="+", default=["dm:300", "er:50"],
                   help=f"Kernel stages run in order, each 'name:iterations[:beta]'. Names: {', '.join(KERNELS)}.")
    g.add_argument("--beta", type=float, default=0.9,
                   help="Default beta for dm / raar / raar_standard (a stage's ':beta' overrides it).")
    g.add_argument("--beta_decay", type=optional_float, default=None,
                   help="Multiply beta by this every iteration (dm, raar, raar_standard).")
    g.add_argument("--beta_min", type=float, default=0.5,
                   help="Floor for beta when --beta_decay is used (dm, raar_standard).")
    g.add_argument("--dm_estimate", type=str, default="data", choices=["data", "model"],
                   help="DM: fit object/pupil to the shadow P_D(2P_M - 1) ('data') or the iterate ('model').")

    # pupil
    g = parser.add_argument_group("pupil")
    g.add_argument("--independent_pupils", action="store_true",
                   help="One pupil per streak (each streak a separate problem). Default: one shared pupil.")
    g.add_argument("--independent_from", type=int, default=0,
                   help="With --independent_pupils: run this many iterations (over all stages) with the "
                        "shared pupil first, then copy it to every streak.")
    g.add_argument("--meta_pupil", type=str, default="random",
                   help="Initial pupil phase: 'random' (random square-Zernike coefficients), 'flat', "
                        "or a path to a phase .npy or .h5 (fitted with square Zernikes, e.g. speckle tracking).")
    g.add_argument("--meta_pupil_dset", type=str, default="speckle_tracking/phase",
                   help="Dataset inside the --meta_pupil .h5 file that holds the pupil phase.")
    g.add_argument("--pupil_delay", type=int, default=0,
                   help="Keep the pupil fixed for the first n iterations (counted over all stages).")
    g.add_argument("--object_steps", type=int, default=0,
                   help="0 = update object and pupil every iteration; n > 0 = n object-only iterations "
                        "per --pupil_steps pupil iterations.")
    g.add_argument("--pupil_steps", type=int, default=1,
                   help="Pupil iterations per cycle when --object_steps > 0.")
    g.add_argument("--pupil_iters", type=int, default=3,
                   help="Rounds of pupil <-> object LSQ inside the model projection, so P_M projects onto "
                        "{P*O} with the pupil free. 0 = pupil fixed in P_M and updated once after each step.")
    g.add_argument("--zernike_steps", type=int, default=0,
                   help="Project the pupil phase onto square Zernikes every n-th iteration (0 = off).")
    g.add_argument("--no_constrain_pupil", action="store_true",
                   help="Do not force the pupil to zero outside the aperture.")
    g.add_argument("--no_normalise_pupil", action="store_true",
                   help="Do not rescale rms|P| = 1 on the covered pupil after each pupil update.")
    g.add_argument("--extend", type=str, default="double",
                   choices=["double", "triple", "quadruple", "quintuple", "sixtuple"])

    # geometry
    g = parser.add_argument_group("geometry")
    g.add_argument("--dqz", type=str, default="streaks",
                   help="q_z spacing between streaks (rad/nm): a number, 'streaks' (median spacing of "
                        "the streaks' mean q_z), 'pixels' (median q_z step between neighbouring "
                        "detector pixels) or 'roi' (roi.delta_qz).")
    g.add_argument("--streak_order", type=str, default="qz", choices=["qz", "as_given"],
                   help="'qz': place each streak on the uniform grid by its mean q_z (q_z = 0 at Nz//2, "
                        "gaps become empty streaks); 'as_given': use the file's order and assume it "
                        "already is that grid.")
    g.add_argument("--nz_pad", type=int, default=0,
                   help="Extra empty streaks on each side of the measured q_z range (--streak_order qz).")
    g.add_argument("--ttheta_deg", type=optional_float, default=None,
                   help="2θ (deg) for --refit_G. None = estimate from the intensity.")
    g.add_argument("--refit_G", action="store_true",
                   help="Refit the global G (and kins, q_z) even if the input stores one.")

    # initialisation / output
    g = parser.add_argument_group("initialisation / output")
    g.add_argument("--init", type=str, default="tilt_corrected",
                   choices=["ones", "amplitude", "amplitude_random", "tilt_corrected", "tilt_corrected_random"],
                   help="Initial slices: unit spectrum, or the FFT of sqrt(mean coherent image) of each "
                        "streak ('amplitude') or sqrt(mean of the streak's images after alignment by cross-"
                        "correlation, ptyxplore tilt_correction) ('tilt_corrected'), with zero or random phase.")
    g.add_argument("--tc_iters", type=int, default=3,
                   help="Alignment passes for --init tilt_corrected*.")
    g.add_argument("--seed", type=int, default=None, help="Random seed (pupil and init phases).")
    g.add_argument("--outdir", type=str, default="recon", help="Output folder (created if missing).")
    g.add_argument("--tag", type=str, default=None, help="Tag added to the output file names (e.g. the job id).")

    args = parser.parse_args()
    args.stages = [parse_stage(s, args.beta) for s in args.steps]
    if not args.independent_pupils:
        args.independent_from = None
    return args


def parse_stage(spec, default_beta):
    parts = spec.split(":")
    if len(parts) not in (2, 3) or parts[0] not in KERNELS:
        raise SystemExit(f"bad --steps entry '{spec}': use name:iterations[:beta] with name in {KERNELS}")
    beta = float(parts[2]) if len(parts) == 3 else default_beta
    return parts[0], int(parts[1]), beta


def make_kernel(name, beta, args):
    if name == "dm":
        return DM(beta=beta, beta_decay=args.beta_decay, beta_min=args.beta_min, estimate=args.dm_estimate)
    if name == "raar":
        return RAAR(beta=beta, beta_decay=args.beta_decay)
    if name == "raar_standard":
        return RAAR_standard(beta=beta, beta_decay=args.beta_decay, beta_min=args.beta_min)
    return ER()


# _______________ Running the stages _______________

def run_stages(engine, args):
    """Run every stage; split them where the pupil delay ends and the pupils become independent."""
    obj_steps, pup_steps = (args.object_steps, args.pupil_steps) if args.object_steps > 0 else (None, None)
    zern = args.zernike_steps if args.zernike_steps > 0 else None
    common = dict(constrain_pupil=not args.no_constrain_pupil,
                  normalise_pupil=not args.no_normalise_pupil,
                  pupil_iters=args.pupil_iters, live_plot=False)
    split_at = args.independent_from          # None = shared pupil throughout

    it0 = 0
    for name, n, beta in args.stages:
        kernel = make_kernel(name, beta, args)
        end = it0 + n
        cuts = sorted({it0, end} | {b for b in (args.pupil_delay, split_at)
                                    if b is not None and it0 < b < end})
        for a, b in zip(cuts[:-1], cuts[1:]):
            fixed = a < args.pupil_delay
            indep = split_at is not None and a >= split_at
            print(f"Running {kernel.__class__.__name__} (beta {beta}) for {b - a} iters "
                  f"[{'pupil fixed' if fixed else 'pupil free'}, "
                  f"{'independent pupils' if indep else 'shared pupil'}]")
            engine.solve(kernel, b - a,
                         object_steps=1 if fixed else obj_steps,
                         pupil_steps=0 if fixed else pup_steps,
                         zernike_steps=None if fixed else zern,
                         independent_pupils=indep, **common)
        it0 = end

    if split_at is not None and not engine.independent_pupils:
        # --independent_from at or beyond the last iteration: still save one pupil per streak
        engine.split_pupil()
    return it0


def save_3d(engine, path, dqz, pad=1):
    """3D object (centred IFFT of the slice stack along q_z), in the same layout as
    recon_ewald's Reconstructed_3D / Axes, plus the pupil coverage."""
    Nz, n0, n1 = engine.rec_fourier_images.shape
    dz = 2 * np.pi / (Nz * dqz)
    dx, dy = split_psize(engine.lr_psize)
    obj = np.pad(engine.obj_3D, pad)
    with h5py.File(path, "a") as h5f:
        h5f["Reconstructed_data"].create_dataset("Pupil_coverage", data=engine.pupil_cover_mask(),
                                                 compression="gzip")
        rec = h5f.create_group("Reconstructed_3D")
        rec.create_dataset("Object_amplitude", data=np.abs(obj), compression="gzip")
        rec.create_dataset("Object_phase", data=np.angle(obj), compression="gzip")
        rec.create_dataset("voxel_size", data=np.array([dz, dx, dy]))
        rec.attrs["pad"] = pad
        rec.attrs["layout"] = (f"[z, x, y], zero-padded by {pad} voxel(s) on every side (Axes/ hold the "
                               f"unpadded axes); voxel_size (dz, dx, dy) in nm, x = image rows (scan_y)")
        ax = h5f.create_group("Axes")
        ax.create_dataset("qz_rad_nm_inv", data=(np.arange(Nz) - Nz // 2) * dqz)
        ax.create_dataset("z_nm", data=(np.arange(Nz) - Nz // 2) * dz)
        ax.create_dataset("x_nm", data=(np.arange(n0) - n0 // 2) * dx)
        ax.create_dataset("y_nm", data=(np.arange(n1) - n1 // 2) * dy)
        ax.attrs["curvature"] = False
        ax.attrs["dqz_rad_nm_inv"] = dqz
        ax.attrs["layout"] = "object [z, x, y]; spectrum [qz, Kx, Ky]"


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # _______________ Data _______________
    data_path = Path(args.input)
    print(f"Input  file: {data_path}")
    with h5py.File(data_path, "r") as f:
        print(dict(f["processing_params"].attrs))

    roi = db.ROI.load_data(data_path)
    pupil_path = Path(args.pupil_file) if args.pupil_file else data_path.parent / f"pupil_scan-{roi.scan_num}.h5"
    pupil_kins = db.ROI.load_data(pupil_path).kins
    print(f"Pupil  file: {pupil_path}")

    if not isinstance(roi.coherent_imgs, list):
        raise SystemExit("--input is not split into streaks (single image stack); "
                         "run ptyxplore make_qz_streaks first")
    images = [np.asarray(im, dtype=np.float64) for im in dp.flip_images(roi, args.axis_flip)]
    kouts = [np.asarray(ko) for ko in roi.kouts]
    k = float(np.median(np.linalg.norm(flat(kouts), axis=1)))
    wavelength = 2 * np.pi / k
    # (row, col) = (scan_y, scan_x): coherent-image rows are scan_y
    scan_step_size = np.abs([roi.step_size_y, roi.step_size_x])

    # ______________ Global G, kins, q_z (per streak) ______________
    G = roi.G
    if G is None or args.refit_G:
        weights = flat([im.mean(axis=(1, 2)) for im in images])
        ttheta = None if args.ttheta_deg is None else np.deg2rad(args.ttheta_deg)
        G, info = fit_global_G(flat(kouts), pupil_kins, wavelength, weights=weights, ttheta=ttheta)
        print(f"Global G fitted: {G}, 2θ = {np.rad2deg(info['ttheta']):.4f} deg")
        kins = [kins_from_G(ko, G, k) for ko in kouts]
        qz0 = [qz_from_G(ko, G, k) for ko in kouts]
    else:
        print(f"Using global G stored in the input file: {G}")
        kins = [np.asarray(ki) for ki in roi.kins]
        qz0 = [np.asarray(q) for q in roi.qz0] if roi.qz0 is not None else [qz_from_G(ko, G, k) for ko in kouts]
    G = np.asarray(G)
    qz_c = np.array([q.mean() for q in qz0])

    if args.dqz == "streaks":
        dqz = float(np.median(np.diff(np.sort(qz_c))))
    elif args.dqz == "pixels":
        dqz = estimate_dqz_from_pixels(flat(roi.kout_coords), flat(qz0))
    elif args.dqz == "roi":
        dqz = float(roi.delta_qz)
    else:
        dqz = float(args.dqz)
    print(f"dqz ({args.dqz}) = {dqz:.4e} rad/nm, wavelength = {wavelength:.5f} nm, "
          f"{len(images)} streaks in the file")

    imgs, ks, counts, qz_offset = place_streaks(images, kins, qz_c, dqz, args.nz_pad, args.streak_order)
    Nz = len(imgs)

    # _______________ Initialisation ____________
    pupil_guess_init = initial_pupil(args.meta_pupil, rng, args.meta_pupil_dset)
    init_fourier = initial_slices(imgs, args.init, rng, args.tc_iters)
    init_fourier[counts == 0] = 0          # empty q_z slices: no data, no support -> stay 0

    # _______________ Recon _______________
    engine = FourierPtychoEngine(images=imgs,
                                 pupil_func=pupil_guess_init,
                                 kins=ks,
                                 ks_pupil=pupil_kins,
                                 lr_psize=scan_step_size,
                                 rec_fourier_images=init_fourier,
                                 num_jobs=64)
    engine.prepare(extend=args.extend)
    pupil_init = engine.pupil_func.copy()

    total = run_stages(engine, args)

    # per-streak change for independent pupils (empty streaks keep the initial pupil)
    change = (np.linalg.norm(engine.pupil_func - pupil_init, axis=(-2, -1))
              / np.linalg.norm(pupil_init))
    if np.ndim(change) == 0:
        print(f"Pupil change ||P - P_init|| / ||P_init|| = {change:.3e}")
    else:
        c = change[counts > 0]
        print(f"Pupil change ||P_s - P_init|| / ||P_init|| over streaks with images: "
              f"min {c.min():.3e}, median {np.median(c):.3e}, max {c.max():.3e}")

    # ______________ Saving __________________
    meta = {
        'engine': 'FourierPtychoEngine',
        'data_path': str(data_path),
        'pupil_path': str(pupil_path),
        'axis_flip': args.axis_flip,
        'steps': " ".join(args.steps),
        'beta_default': args.beta,
        'stage_betas': " ".join(f"{name}:{beta}" for name, _, beta in args.stages),
        'beta_decay': str(args.beta_decay),
        'beta_min': args.beta_min,
        'dm_estimate': args.dm_estimate,
        'independent_from': str(args.independent_from),
        'pupil': args.meta_pupil,
        'pupil_dset': args.meta_pupil_dset,
        'pupil_delay': args.pupil_delay,
        'object_steps': args.object_steps,
        'pupil_steps': args.pupil_steps,
        'pupil_iters': args.pupil_iters,
        'zernike_steps': args.zernike_steps,
        'constrain_pupil': not args.no_constrain_pupil,
        'normalise_pupil': not args.no_normalise_pupil,
        'extend': args.extend,
        'dqz': dqz,
        'dqz_source': args.dqz,
        'streak_order': args.streak_order,
        'qz_offset': qz_offset,
        'Nz': Nz,
        'nz_pad': args.nz_pad,
        'init': args.init,
        'tc_iters': args.tc_iters,
        'seed': str(args.seed),
        'G': G,
        'wavelength_nm': wavelength,
        'det_distance_nm': roi.det_distance,
        'delta_q': roi.delta_q,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    mode = "indep" if args.independent_pupils else "shared"
    save_path = str(out_dir / f"recon_scan{roi.scan_num}_{mode}_iters{total}{tag}_{time_str}")

    engine._post_process()
    engine.save_reconsturction(save_path + ".h5", metadata=meta)   # also sets 'independent_pupils'
    save_3d(engine, save_path + ".h5", dqz)
    with h5py.File(save_path + ".h5", "a") as f:
        f["Recon_Params"].create_dataset("images_per_streak", data=counts)

    print(f"Saved {save_path}.h5")


if __name__ == '__main__':

    main()
