"""
Phase retrieval of real data (a saved ptyxplore Bragg ROI) with the curved-Ewald-slice
engine (ptyphase.runner_ewald) and a 3D real-space support.

The input must already be split into streaks (e.g. ptyxplore make_qz_streaks). The engine
needs streak index <-> q_z on a uniform grid with q_z = 0 at index Nz//2, so by default every
given streak is placed at index round(<q_z> / dqz) + Nz//2 from its mean q_z (roi.qz0), and
missing q_z values become empty streaks (the support fills them in). The pixels inside a
streak are never regrouped. --streak_order as_given uses the file's order unchanged.

Kernels run in the order given by --steps, e.g.

    python recon_ewald.py --input roi.h5 --steps dnc:300 er:50 --z_extent_nm 300
    python recon_ewald.py --input roi.h5 --steps dm:200:0.9 er:50 --support_file sup.npy \
                          --shrinkwrap_steps 20 --sw_start 100 --pupil_delay 50

Kernels: dm, raar, raar_standard, er, dnc (all the *Ewald versions in algorithms_ewald).
See `python recon_ewald.py --help` for everything else.
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

from ptyphase import zernike
from ptyphase.runner_ewald import FourierPtychoEwaldEngine
from ptyphase.algorithms_ewald import DMEwald, RAAREwald, RAAR_standardEwald, EREwald, DnCEwald

time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

KERNELS = ("dm", "raar", "raar_standard", "er", "dnc")


def optional_float(s):
    return None if s in (None, "None", "none", "") else float(s)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Curved-Ewald-slice Fourier ptychography with a 3D support (ptyphase.runner_ewald).",
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
    g.add_argument("--steps", type=str, nargs="+", default=["dnc:300", "er:50"],
                   help=f"Kernel stages run in order, each 'name:iterations[:beta]'. Names: {', '.join(KERNELS)}.")
    g.add_argument("--beta", type=float, default=1.0,
                   help="Default beta for dm / raar / raar_standard / dnc (a stage's ':beta' overrides it).")
    g.add_argument("--beta_decay", type=optional_float, default=None,
                   help="Multiply beta by this every iteration (dm, raar_standard, dnc).")
    g.add_argument("--beta_min", type=float, default=0.5,
                   help="Floor for beta when --beta_decay is used.")
    g.add_argument("--dm_estimate", type=str, default="data", choices=["data", "model"],
                   help="DM: fit object/pupil to the shadow P_D(2P_M - 1) ('data') or the iterate ('model').")
    g.add_argument("--lam", type=float, default=1.0,
                   help="DnC: weight of F against Psi in the overlap projection.")

    # pupil
    g = parser.add_argument_group("pupil")
    g.add_argument("--meta_pupil", type=str, default="random",
                   help="Initial pupil phase: 'random' (random square-Zernike coefficients), 'flat', "
                        "or a path to a phase .npy (fitted with square Zernikes, e.g. speckle tracking).")
    g.add_argument("--pupil_delay", type=int, default=0,
                   help="Keep the pupil fixed for the first n iterations (counted over all stages).")
    g.add_argument("--object_steps", type=int, default=0,
                   help="0 = update object and pupil every iteration; n > 0 = n object-only iterations "
                        "per --pupil_steps pupil iterations.")
    g.add_argument("--pupil_steps", type=int, default=1,
                   help="Pupil iterations per cycle when --object_steps > 0.")
    g.add_argument("--zernike_steps", type=int, default=0,
                   help="Project the pupil phase onto square Zernikes every n-th iteration (0 = off; "
                        "not used by dnc).")
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
    g.add_argument("--no_curvature", action="store_true",
                   help="Flat slices (equals the old ifft_images_z stacking).")

    # support
    g = parser.add_argument_group("support")
    g.add_argument("--support_file", type=str, default=None,
                   help=".npy bool support, (Nz, n0, n1) [z, x, y] or (n0, n1) (same lateral mask "
                        "for every z). Nz is printed by the binning step.")
    g.add_argument("--z_extent_nm", type=optional_float, default=None,
                   help="Slab support |z| <= z_extent/2 (nm), combined (AND) with --support_file.")
    g.add_argument("--shrinkwrap_steps", type=int, default=0,
                   help="Update the support by shrinkwrap every n iterations (0 = off).")
    g.add_argument("--sw_start", type=int, default=0,
                   help="First iteration (over all stages) at which shrinkwrap is active.")
    g.add_argument("--sw_sigma", type=float, default=1.0, help="Shrinkwrap blur (voxels).")
    g.add_argument("--sw_threshold", type=float, default=0.1, help="Shrinkwrap threshold (fraction of max).")

    # initialisation / output
    g = parser.add_argument_group("initialisation / output")
    g.add_argument("--init", type=str, default="ones", choices=["ones", "amplitude", "amplitude_random"],
                   help="Initial slices: unit spectrum, or the FFT of sqrt(mean coherent image) of each "
                        "streak with zero or random phase.")
    g.add_argument("--seed", type=int, default=None, help="Random seed (pupil and init phases).")
    g.add_argument("--outdir", type=str, default="recon_ewald", help="Output folder (created if missing).")
    g.add_argument("--tag", type=str, default=None, help="Tag added to the output file names (e.g. the job id).")

    args = parser.parse_args()
    args.stages = [parse_stage(s, args.beta) for s in args.steps]
    return args


def parse_stage(spec, default_beta):
    parts = spec.split(":")
    if len(parts) not in (2, 3) or parts[0] not in KERNELS:
        raise SystemExit(f"bad --steps entry '{spec}': use name:iterations[:beta] with name in {KERNELS}")
    beta = float(parts[2]) if len(parts) == 3 else default_beta
    return parts[0], int(parts[1]), beta


def make_kernel(name, beta, args):
    if name == "dm":
        return DMEwald(beta=beta, beta_decay=args.beta_decay, beta_min=args.beta_min, estimate=args.dm_estimate)
    if name == "raar":
        return RAAREwald(beta=beta, beta_decay=args.beta_decay)
    if name == "raar_standard":
        return RAAR_standardEwald(beta=beta, beta_decay=args.beta_decay, beta_min=args.beta_min)
    if name == "er":
        return EREwald()
    return DnCEwald(beta=beta, lam=args.lam, beta_decay=args.beta_decay, beta_min=args.beta_min)


# _______________ Geometry helpers _______________

def flat(v):
    return np.concatenate([np.asarray(a) for a in v]) if isinstance(v, list) else np.asarray(v)


def estimate_dqz_from_pixels(kout_coords, qz0):
    """q_z spacing between neighbouring detector pixels (median |grad q_z|),
    same rule as recon_3d.py / reciprocity_sim.sampling.plan_grid."""
    rc = kout_coords - kout_coords.min(axis=0)
    grid = np.full(rc.max(axis=0) + 1, np.nan)
    grid[rc[:, 0], rc[:, 1]] = qz0
    g0, g1 = np.gradient(grid)
    return float(np.nanmedian(np.hypot(g0, g1)))


def place_streaks(images, kins, qz_c, dqz, nz_pad, order):
    """Put the given streaks on the uniform q_z grid (index Nz//2 <-> q_z = qz_offset).
    images, kins: per-streak lists; qz_c: mean q_z of each streak.

    The grid is shifted by qz_offset (the circular mean of the streak centres modulo dqz)
    so the streaks sit on grid points instead of up to half a step away. The engine then
    treats slice s as q_z = (s - Nz//2) dqz, i.e. the object carries an extra factor
    exp(-i qz_offset z): a linear phase along z only, |obj| is unchanged.
    Returns imgs, ks, counts, qz_offset."""
    n0, n1 = images[0].shape[1:]
    qz_offset = 0.0
    if order == "as_given":
        imgs, ks = list(images), list(kins)
    else:
        frac = qz_c / dqz - np.round(qz_c / dqz)
        qz_offset = float(dqz * np.angle(np.mean(np.exp(2j * np.pi * frac))) / (2 * np.pi))
        idx = np.round((qz_c - qz_offset) / dqz).astype(int)
        if len(np.unique(idx)) < len(idx):
            raise SystemExit("two streaks fall on the same q_z grid point: check --dqz, "
                             "or use --streak_order as_given")
        Nz = 2 * (max(-idx.min(), idx.max() + 1) + nz_pad)
        imgs = [np.zeros((0, n0, n1), images[0].dtype) for _ in range(Nz)]
        ks = [np.zeros((0, 3)) for _ in range(Nz)]
        for j, im, ki in zip(idx + Nz // 2, images, kins):
            imgs[j], ks[j] = im, ki
        off = qz_c - qz_offset - idx * dqz
        print(f"grid offset qz_offset = {qz_offset:.4e} rad/nm ({qz_offset/dqz:.2f} dqz) | streak q_z - grid "
              f"point: max {np.abs(off).max()/dqz:.2f} dqz, rms {np.sqrt(np.mean(off**2))/dqz:.2f} dqz")
    counts = np.array([len(i) for i in imgs])
    print(f"Streaks: Nz = {len(imgs)} (dqz = {dqz:.4e} rad/nm, {order}), {np.sum(counts > 0)} with images, "
          f"images per streak: min {counts[counts > 0].min()}, median {int(np.median(counts[counts > 0]))}, "
          f"max {counts.max()}")
    return imgs, ks, counts, qz_offset


def initial_pupil(meta_pupil, rng):
    if meta_pupil == "flat":
        print("Using a flat pupil phase\n")
        return np.ones((256, 256), complex)
    if meta_pupil == "random":
        modes = zernike.SquarePolynomials().get_function_list()
        coeffs = rng.random(len(modes))
        print("Using random pupil phase\n")
    else:
        coeffs = np.array(zernike.SquarePolynomials.extract_square_coefficients_vectorized(np.load(meta_pupil)))
        print(f"Using pupil phase from {meta_pupil}\n")
    return np.exp(1j * zernike.SquarePolynomials.get_zernike_wavefront(coeffs, (256, 256)))


def initial_slices(imgs, mode, rng):
    Nz = len(imgs)
    shape = next(i for i in imgs if len(i)).shape[1:]
    if mode == "ones":
        return np.ones((Nz, *shape), complex)
    global_amp = np.sqrt(np.mean(np.concatenate([i for i in imgs if len(i)]), axis=0))
    amp = np.array([np.sqrt(i.mean(axis=0)) if len(i) else global_amp for i in imgs])
    phase = rng.uniform(-np.pi, np.pi, amp.shape) if mode == "amplitude_random" else 0.0
    obj = amp * np.exp(1j * phase)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))


def build_support(args, Nz, img_shape, dqz):
    support = None
    if args.support_file is not None:
        support = np.load(args.support_file).astype(bool)
        if support.shape == img_shape:
            support = np.broadcast_to(support, (Nz, *img_shape)).copy()
        if support.shape != (Nz, *img_shape):
            raise SystemExit(f"--support_file has shape {support.shape}; need {(Nz, *img_shape)} or {img_shape}")
    if args.z_extent_nm is not None:
        z = (np.arange(Nz) - Nz // 2) * 2 * np.pi / (Nz * dqz)
        slab = np.broadcast_to((np.abs(z) <= args.z_extent_nm / 2)[:, None, None], (Nz, *img_shape))
        support = slab.copy() if support is None else support & slab
    if support is not None:
        print(f"Support: {100 * support.mean():.1f}% of the box")
    return support


# _______________ Running the stages _______________

def run_stages(engine, args):
    """Run every stage; split them where the pupil delay ends and shrinkwrap starts."""
    obj_steps, pup_steps = (args.object_steps, args.pupil_steps) if args.object_steps > 0 else (None, None)
    zern = args.zernike_steps if args.zernike_steps > 0 else None
    common = dict(constrain_pupil=not args.no_constrain_pupil,
                  normalise_pupil=not args.no_normalise_pupil,
                  sw_sigma=args.sw_sigma, sw_threshold=args.sw_threshold, live_plot=False)

    it0 = 0
    for name, n, beta in args.stages:
        kernel = make_kernel(name, beta, args)
        end = it0 + n
        cuts = sorted({it0, end} | {b for b in (args.pupil_delay, args.sw_start) if it0 < b < end})
        for a, b in zip(cuts[:-1], cuts[1:]):
            fixed = a < args.pupil_delay
            sw = args.shrinkwrap_steps if (args.shrinkwrap_steps > 0 and a >= args.sw_start) else None
            print(f"Running {kernel.__class__.__name__} (beta {beta}) for {b - a} iters "
                  f"[{'pupil fixed' if fixed else 'pupil free'}, shrinkwrap {sw}]")
            engine.solve(kernel, b - a,
                         object_steps=1 if fixed else obj_steps,
                         pupil_steps=0 if fixed else pup_steps,
                         zernike_steps=None if fixed else zern,
                         shrinkwrap_steps=sw, **common)
        it0 = end
    return it0


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
    Nz, img_shape = len(imgs), images[0].shape[1:]

    # _______________ Initialisation ____________
    pupil_guess_init = initial_pupil(args.meta_pupil, rng)
    init_fourier = initial_slices(imgs, args.init, rng)
    support = build_support(args, Nz, img_shape, dqz)

    # _______________ Recon _______________
    engine = FourierPtychoEwaldEngine(images=imgs,
                                      pupil_func=pupil_guess_init,
                                      kins=ks,
                                      ks_pupil=pupil_kins,
                                      lr_psize=scan_step_size,
                                      rec_fourier_images=init_fourier,
                                      wavelength=wavelength,
                                      dqz=dqz,
                                      num_jobs=64)
    engine.prepare(extend=args.extend, curvature=not args.no_curvature, support=support)
    pupil_init = engine.pupil_func.copy()

    total = run_stages(engine, args)

    print(f"Pupil change ||P - P_init|| / ||P_init|| = "
          f"{np.linalg.norm(engine.pupil_func - pupil_init) / np.linalg.norm(pupil_init):.3e}")

    # ______________ Saving __________________
    meta = {
        'data_path': str(data_path),
        'pupil_path': str(pupil_path),
        'axis_flip': args.axis_flip,
        'steps': " ".join(args.steps),
        'beta': args.beta,
        'beta_decay': str(args.beta_decay),
        'lam': args.lam,
        'dm_estimate': args.dm_estimate,
        'pupil': args.meta_pupil,
        'pupil_delay': args.pupil_delay,
        'object_steps': args.object_steps,
        'pupil_steps': args.pupil_steps,
        'zernike_steps': args.zernike_steps,
        'curvature': not args.no_curvature,
        'dqz': dqz,
        'dqz_source': args.dqz,
        'streak_order': args.streak_order,
        'qz_offset': qz_offset,
        'dz': engine.dz,
        'Nz': Nz,
        'z_extent_nm': str(args.z_extent_nm),
        'support_file': str(args.support_file),
        'shrinkwrap_steps': args.shrinkwrap_steps,
        'sw_start': args.sw_start,
        'sw_sigma': args.sw_sigma,
        'sw_threshold': args.sw_threshold,
        'init': args.init,
        'seed': str(args.seed),
        'G': G,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    save_path = str(out_dir / f"recon_ewald_scan{roi.scan_num}_{args.axis_flip}_iters{total}{tag}_{time_str}")

    engine._post_process()
    engine.save_reconsturction(save_path + ".h5", metadata=meta)
    with h5py.File(save_path + ".h5", "a") as f:
        f["Reconstructed_data"].create_dataset("images_per_streak", data=counts)

    obj3D = np.pad(engine.obj_3D, ((1, 1), (1, 1), (1, 1)))
    with h5py.File(save_path + "obj3d.h5", "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("voxel_size", data=np.array([engine.dz, roi.step_size_y, roi.step_size_x]))  # (z, scan_y, scan_x)
        grp.create_dataset("array", data=obj3D)
        grp.create_dataset("parent", data=str(data_path))

        grp = f.create_group("meta")
        grp.create_dataset("distance_cm", data=roi.det_distance * 1e-7)
        grp.create_dataset("step_xy_nm", data=roi.step_size_x)
        grp.create_dataset("delta_q_inv_nm", data=roi.delta_q)
        grp.create_dataset("dqz_inv_nm", data=dqz)
        grp.create_dataset("qz_offset_inv_nm", data=qz_offset)
        grp.create_dataset("G_inv_nm", data=G)

    print(f"Saved {save_path}.h5 and {save_path}obj3d.h5")


if __name__ == '__main__':

    main()
