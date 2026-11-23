"""
Gradient-based 3D reconstruction of real data (a saved ptyxplore Bragg ROI) with
ptyphase.optimisation.engine.OptimFourierPtycho3D.

The unknown is ONE 3D spectrum S(q_z, K0, K1) plus the pupil. Every (image, K) sample sits at
its own Ewald q_z (one global G), the pupil patch is P(k_in0 + K), and the model image
|IFFT2(P * S)|^2 is fitted to the measured coherent images with Adam on an L2 loss.

The input can be split into streaks or be one flat stack: the 3D model places every pixel by
its own q_z, so the streak grouping is not used (except for --dqz streaks).

    python recon_opt.py --input roi.h5 --epochs 2000 --lr_spectrum 3e-2 --pupil_start 200
    python recon_opt.py --input roi.h5 --pupil_model poly --meta_pupil st_phase.npy --epochs 1000

See `python recon_opt.py --help` for everything else.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (registers the compression filters of the ROI files)
import torch

import ptyxplore.loader as db
import ptyxplore.roi_ops as dp
from ptyxplore.kvectors import fit_global_G, kins_from_G, qz_from_G

from ptyphase.optimisation.engine import OptimFourierPtycho3D

# shared with the other real-data scripts, so all load and initialise the same way
from recon_ewald import optional_float, flat, estimate_dqz_from_pixels, initial_pupil

time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Gradient-based 3D Fourier ptychography (ptyphase.optimisation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    g = parser.add_argument_group("data")
    g.add_argument("--input", type=str, required=True,
                   help="Saved Bragg ROI (ptyxplore ROI.save_data), streak-split or one stack.")
    g.add_argument("--pupil_file", type=str, default=None,
                   help="Saved pupil ROI (its kins set the pupil grid). "
                        "Default: pupil_scan-<scan>.h5 next to --input.")
    g.add_argument("--axis_flip", type=str, default="xy",
                   choices=["xy", "x_neg_y", "neg_x_y", "neg_x_neg_y"])

    # optimisation
    g = parser.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=1000, help="Passes over all images.")
    g.add_argument("--loss", type=str, default="l2_intensity", choices=["l2_intensity", "l2_amplitude"])
    g.add_argument("--optimiser", type=str, default="adam", choices=["adam", "sgd"])
    g.add_argument("--lr_spectrum", type=float, default=1e-2,
                   help="Learning rate of the spectrum (relative: S is stored in units of sqrt(n0*n1)).")
    g.add_argument("--lr_pupil", type=float, default=1e-3, help="Learning rate of the pupil.")
    g.add_argument("--lr_decay", type=optional_float, default=None,
                   help="Multiply both learning rates by this every epoch (ExponentialLR).")
    g.add_argument("--batch_size", type=int, default=None,
                   help="Images per optimiser step. None = all (full-batch gradient).")
    g.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    g.add_argument("--dtype", type=str, default="float64", choices=["float64", "float32"])
    g.add_argument("--log_every", type=int, default=10)

    # pupil
    g = parser.add_argument_group("pupil")
    g.add_argument("--pupil_model", type=str, default="pixel", choices=["pixel", "poly"],
                   help="pixel: free complex pupil inside the aperture; poly: phase = sum c_i S_i "
                        "(square polynomials S1..S29, fitted to the initial pupil).")
    g.add_argument("--meta_pupil", type=str, default="flat",
                   help="Initial pupil phase: 'flat', 'random' (random square-Zernike coefficients), "
                        "or a path to a phase .npy or .h5 (fitted with square Zernikes).")
    g.add_argument("--meta_pupil_dset", type=str, default="speckle_tracking/phase",
                   help="Dataset inside the --meta_pupil .h5 file that holds the pupil phase.")
    g.add_argument("--fixed_pupil", action="store_true", help="Never update the pupil.")
    g.add_argument("--freeze_pupil_amp", action="store_true", help="Learn the pupil phase only.")
    g.add_argument("--pupil_start", type=int, default=0, help="Epoch from which the pupil is learnt.")
    g.add_argument("--alternate", type=int, nargs=2, default=None, metavar=("N_SPEC", "N_PUP"),
                   help="Cycle N_SPEC spectrum-only epochs with N_PUP pupil-only epochs (after --pupil_start).")
    g.add_argument("--no_normalise_pupil", action="store_true",
                   help="Do not rescale rms|P| = 1 on the covered pupil after each pupil step.")
    g.add_argument("--extend", type=str, default="double",
                   choices=["double", "triple", "quadruple", "quintuple", "sixtuple"])

    # geometry
    g = parser.add_argument_group("geometry")
    g.add_argument("--dqz", type=str, default="pixels",
                   help="q_z grid spacing (rad/nm): a number, 'pixels' (median q_z step between "
                        "neighbouring detector pixels) or 'streaks' (median spacing of the streaks' "
                        "mean q_z; needs streak-split input).")
    g.add_argument("--qz_oversample", type=float, default=1.0, help="q_z grid spacing = dqz / this.")
    g.add_argument("--nz_pad", type=int, default=2, help="Extra q_z nodes on each side of the sampled range.")
    g.add_argument("--ttheta_deg", type=optional_float, default=None,
                   help="2θ (deg) for --refit_G. None = estimate from the intensity.")
    g.add_argument("--refit_G", action="store_true",
                   help="Refit the global G (and kins) even if the input stores one.")

    # initialisation / output
    g = parser.add_argument_group("initialisation / output")
    g.add_argument("--init", type=str, default="amplitude", choices=["ones", "amplitude"],
                   help="Initial spectrum: unit on measured voxels, or the LSQ fit of sqrt(I) with zero phase "
                        "through the initial pupil.")
    g.add_argument("--seed", type=int, default=None, help="Random seed (pupil, minibatch order).")
    g.add_argument("--outdir", type=str, default="recon_opt", help="Output folder (created if missing).")
    g.add_argument("--tag", type=str, default=None, help="Tag added to the output file names (e.g. the job id).")

    return parser.parse_args()


def as_list(v):
    return v if isinstance(v, list) else [v]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # _______________ Data _______________
    data_path = Path(args.input)
    print(f"Input  file: {data_path}")
    with h5py.File(data_path, "r") as f:
        print(dict(f["processing_params"].attrs))

    roi = db.ROI.load_data(data_path)
    pupil_path = Path(args.pupil_file) if args.pupil_file else data_path.parent / f"pupil_scan-{roi.scan_num}.h5"
    pupil_kins = db.ROI.load_data(pupil_path).kins
    if isinstance(pupil_kins, list):
        raise SystemExit(f"{pupil_path} is streak-split (list of kins); use the unsplit pupil file "
                         "(e.g. pupil_scan-<scan>_binned4.h5)")
    print(f"Pupil  file: {pupil_path}")

    images = [np.asarray(im, dtype=np.float64) for im in as_list(dp.flip_images(roi, args.axis_flip))]
    kouts = [np.asarray(ko) for ko in as_list(roi.kouts)]
    k = float(np.median(np.linalg.norm(flat(kouts), axis=1)))
    wavelength = 2 * np.pi / k
    # (row, col) = (scan_y, scan_x): coherent-image rows are scan_y
    scan_step_size = np.abs([roi.step_size_y, roi.step_size_x])

    # ______________ Global G, kins, q_z ______________
    G = getattr(roi, "G", None)
    if G is None or args.refit_G:
        weights = flat([im.mean(axis=(1, 2)) for im in images])
        ttheta = None if args.ttheta_deg is None else np.deg2rad(args.ttheta_deg)
        G, info = fit_global_G(flat(kouts), pupil_kins, wavelength, weights=weights, ttheta=ttheta)
        print(f"Global G fitted: {G}, 2θ = {np.rad2deg(info['ttheta']):.4f} deg")
        kins = [kins_from_G(ko, G, k) for ko in kouts]
    else:
        print(f"Using global G stored in the input file: {G}")
        kins = [np.asarray(ki) for ki in as_list(roi.kins)]
    G = np.asarray(G)
    qz0 = [qz_from_G(ko, G, k) for ko in kouts]

    if args.dqz == "pixels":
        dqz = estimate_dqz_from_pixels(flat(as_list(roi.kout_coords)), flat(qz0))
    elif args.dqz == "streaks":
        if len(images) < 2:
            raise SystemExit("--dqz streaks needs streak-split input")
        dqz = float(np.median(np.diff(np.sort([q.mean() for q in qz0]))))
    else:
        dqz = float(args.dqz)
    print(f"dqz ({args.dqz}) = {dqz:.4e} rad/nm, wavelength = {wavelength:.5f} nm, "
          f"{sum(len(i) for i in images)} images in {len(images)} stack(s)")

    # _______________ Engine _______________
    pupil_guess = initial_pupil(args.meta_pupil, rng, args.meta_pupil_dset)
    engine = OptimFourierPtycho3D(images=images, kins=kins, kouts=kouts, G=G, wavelength=wavelength,
                                  pupil_func=pupil_guess, ks_pupil=pupil_kins, lr_psize=scan_step_size)
    engine.prepare(extend=args.extend, dqz=dqz, qz_oversample=args.qz_oversample, nz_pad=args.nz_pad,
                   init=args.init, device=args.device, dtype=args.dtype, pupil_model=args.pupil_model,
                   freeze_pupil_amp=args.freeze_pupil_amp)
    pupil_init = engine.pupil_func.copy()
    cover = engine.pupil_cover_mask()
    print(f"Pupil coverage: {100 * cover.sum() / max((np.abs(engine.ctf) > 0).sum(), 1):.1f}% of the aperture")

    opt_cls = torch.optim.Adam if args.optimiser == "adam" else torch.optim.SGD
    engine.set_optimiser(lr_spectrum=args.lr_spectrum,
                         lr_pupil=None if args.fixed_pupil else args.lr_pupil, optimiser=opt_cls)
    if args.lr_decay is not None:
        engine.set_scheduler(torch.optim.lr_scheduler.ExponentialLR, "spectrum", gamma=args.lr_decay)
        if not args.fixed_pupil:
            engine.set_scheduler(torch.optim.lr_scheduler.ExponentialLR, "pupil", gamma=args.lr_decay)

    # _______________ Optimise _______________
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    engine.run(epochs=args.epochs, batch_size=args.batch_size, pupil_start=args.pupil_start,
               alternate=args.alternate, loss=args.loss, normalise_pupil=not args.no_normalise_pupil,
               log_every=args.log_every, seed=args.seed)
    peak = torch.cuda.max_memory_allocated() / 1e9 if engine.device.type == "cuda" else float("nan")
    rt = float(np.sum(engine.epoch_times))
    print(f"Optimisation: {args.epochs} epochs, {rt:.0f} s, final loss {engine.losses[-1]:.4e}, "
          f"data misfit {engine.data_misfits[-1]:.4e}, peak GPU memory {peak:.2f} GB")
    # the gauge rescales P, so compare on the covered pupil up to a scale
    p0, p1 = pupil_init[cover], engine.pupil_func[cover]
    p1 = p1 * (np.vdot(p1, p0) / np.vdot(p1, p1))
    print(f"Pupil change on the coverage ||P - P_init|| / ||P_init|| (best scale) = "
          f"{np.linalg.norm(p1 - p0) / np.linalg.norm(p0):.3e}")

    # ______________ Saving __________________
    meta = {
        'data_path': str(data_path),
        'pupil_path': str(pupil_path),
        'axis_flip': args.axis_flip,
        'epochs': args.epochs,
        'loss': args.loss,
        'optimiser': args.optimiser,
        'lr_spectrum': args.lr_spectrum,
        'lr_pupil': args.lr_pupil,
        'lr_decay': str(args.lr_decay),
        'batch_size': str(args.batch_size),
        'dtype': args.dtype,
        'pupil': args.meta_pupil,
        'pupil_dset': args.meta_pupil_dset,
        'fixed_pupil': args.fixed_pupil,
        'freeze_pupil_amp': args.freeze_pupil_amp,
        'pupil_start': args.pupil_start,
        'alternate': str(args.alternate),
        'normalise_pupil': not args.no_normalise_pupil,
        'extend': args.extend,
        'dqz': engine.geom.dqz,
        'dqz_source': args.dqz,
        'qz_oversample': args.qz_oversample,
        'nz_pad': args.nz_pad,
        'dz': engine.dz,
        'init': args.init,
        'seed': str(args.seed),
        'G': G,
        'wavelength_nm': wavelength,
        'det_distance_nm': roi.det_distance,
        'delta_q': roi.delta_q,
        'runtime_s': rt,
        'peak_gpu_memory_GB': peak,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    save_path = str(out_dir / f"recon_opt_scan{roi.scan_num}_{args.axis_flip}_{args.pupil_model}"
                              f"_ep{args.epochs}{tag}_{time_str}.h5")
    engine.save_reconsturction(save_path, metadata=meta)
    print(f"Saved {save_path}")


if __name__ == '__main__':

    main()
