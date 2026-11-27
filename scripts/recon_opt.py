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

Per-streak position refinement (one offset per streak along the in-pupil normal of the streaks):
    python recon_opt.py --input roi_streaked.h5 --epochs 3000 --streak_shifts streaks \
        --lr_shift 0.05 --shift_start 1000 --shift_scan_epochs 1000 2000
Scan an existing result only (per-streak loss vs offset, S and P fixed):
    python recon_opt.py --input roi_streaked.h5 --init_from old_result.h5 --epochs 0 \
        --streak_shifts streaks --shift_scan_epochs 0

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
from ptyphase.optimisation.regularisers import TV3D, Support3D, PupilSmooth, pupil_roughness

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
    g.add_argument("--loss", type=str, default="l2_intensity", choices=["l2_intensity", "l2_amplitude", "poisson"])
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
    g.add_argument("--pixel_threshold", type=float, default=None,
                   help="Intensity threshold (input-image units, same for all images). Pixels below it are "
                        "excluded from the data loss (--threshold_mode mask) or set to 0 (zero). The "
                        "start-up log prints intensity percentiles to help choose it.")
    g.add_argument("--threshold_mode", type=str, default="mask", choices=["mask", "zero"],
                   help="mask: below-threshold pixels do not enter the loss; zero: they are fitted as dark.")
    g.add_argument("--mask_unmeasured", action=argparse.BooleanOptionalAction, default=True,
                   help="Hold spectrum voxels that no data sample reaches (q_z padding, uncovered edges) at 0, "
                        "so regularisers cannot fill them (default on; --no-mask_unmeasured to free them).")
    g.add_argument("--ttheta_deg", type=optional_float, default=None,
                   help="2θ (deg) for --refit_G. None = estimate from the intensity.")
    g.add_argument("--refit_G", action="store_true",
                   help="Refit the global G (and kins) even if the input stores one.")

    # streak position refinement
    g = parser.add_argument_group("streak shifts (per-streak offset of the pupil position, perpendicular to the streaks)")
    g.add_argument("--streak_shifts", type=str, default=None, choices=["streaks", "qz"],
                   help="Enable: one offset per input streak ('streaks', needs streak-split input) or per q_z bin "
                        "of width dqz ('qz', for a flat stack). Uses the sub-pixel forward model.")
    g.add_argument("--shift_mode", type=str, default="kout", choices=["kout", "kin"],
                   help="kout: k_out and k_in move together, q_z follows (a detector-mapping error, one G); "
                        "kin: only k_in moves (a per-streak G).")
    g.add_argument("--lr_shift", type=optional_float, default=None,
                   help="Adam learning rate of the offsets (pupil px per step). None = offsets only change in scans.")
    g.add_argument("--shift_start", type=int, default=0, help="Epoch from which the offsets are learnt.")
    g.add_argument("--shift_max", type=optional_float, default=None,
                   help="Bound on |offset| (pupil px). None = 1.5 streak spacings.")
    g.add_argument("--shift_scan_epochs", type=int, nargs="*", default=[],
                   help="Epochs at which to grid-scan the per-streak loss vs offset (S, P fixed) and apply the "
                        "best offsets. An epoch equal to the final epoch count scans after the last epoch.")
    g.add_argument("--shift_scan_range", type=optional_float, default=None,
                   help="Scan offsets -R..R (pupil px) around the current ones. None = --shift_max.")
    g.add_argument("--shift_scan_step", type=float, default=1.0, help="Scan step (pupil px).")
    g.add_argument("--subpixel", action="store_true",
                   help="Sub-pixel (bilinear) pupil placement, also without --streak_shifts.")
    g.add_argument("--placement", type=str, default="bilinear", choices=["bilinear", "nearest"],
                   help="Pupil placement with --streak_shifts: bilinear (sub-pixel, needed to learn the shifts) or "
                        "nearest (rounded to pupil pixels as in runs without shifts: use it to scan a result that was "
                        "trained without --subpixel/--streak_shifts, with an integer --shift_scan_step).")

    # regularisation
    g = parser.add_argument_group("regularisation (values tracked per epoch in Losses/reg_<name>)")
    g.add_argument("--tv_weight", type=float, default=0.0,
                   help="Weight of the 3D total variation of the object (0 = no TV term).")
    g.add_argument("--tv_mode", type=str, default="amplitude", choices=["amplitude", "complex"],
                   help="TV of |obj| (phase left free) or of the complex obj (penalises phase changes too).")
    g.add_argument("--tv_start", type=int, default=0, help="Epoch at which the TV weight switches on.")
    g.add_argument("--tv_ramp", type=int, default=0,
                   help="Ramp the TV weight linearly from 0 to --tv_weight over this many epochs.")
    g.add_argument("--tv_per_nm", action="store_true",
                   help="Weight the z/x/y differences by 1/voxel size (relative to the x step), i.e. "
                        "penalise gradients per nm instead of per voxel (dz differs from the scan step).")
    g.add_argument("--tv_track", action="store_true",
                   help="Track the TV value per epoch even when --tv_weight is 0.")
    g.add_argument("--support_size", type=float, nargs="+", default=None, metavar="NM",
                   help="Support box in nm, centred on the origin: SZ (|z| <= SZ/2, all x, y) or SZ SX SY. "
                        "Enables the support term: fraction of the object energy outside the support.")
    g.add_argument("--support_weight", type=float, default=0.0,
                   help="Weight of the support term (0 = tracked only, if --support_size is given).")
    g.add_argument("--support_start", type=int, default=0, help="Epoch at which the support weight switches on.")
    g.add_argument("--support_ramp", type=int, default=0,
                   help="Ramp the support weight linearly from 0 over this many epochs.")
    g.add_argument("--sw_threshold", type=float, default=None,
                   help="Shrinkwrap threshold (fraction of max of the blurred |obj|). Not given = no shrinkwrap, "
                        "the support stays the box.")
    g.add_argument("--sw_sigma", type=float, default=1.0, help="Shrinkwrap Gaussian blur sigma (voxels).")
    g.add_argument("--sw_every", type=int, default=100, help="Shrinkwrap update interval (epochs).")
    g.add_argument("--sw_start", type=int, default=0, help="First shrinkwrap epoch.")
    g.add_argument("--sw_free", action="store_true",
                   help="Let shrinkwrap replace the support; default keeps it inside the --support_size box.")
    g.add_argument("--pupil_smooth_weight", type=float, default=0.0,
                   help="Weight of the pupil roughness term: mean squared finite difference of P over the covered "
                        "aperture / mean |P|^2 (0 = off). Suppresses pixel-scale pupil texture and the dips at the "
                        "image-centre pixels that lead to the pixel-pupil blow-up. Scan e.g. 1e-3 .. 1e-1.")
    g.add_argument("--pupil_smooth_order", type=int, default=2, choices=[1, 2],
                   help="2: second differences along each pupil axis (curvature; a phase slope of g rad/px costs "
                        "~g^4, a checkerboard 16 eps^2); 1: first differences (gradient; a slope costs ~g^2).")
    g.add_argument("--pupil_smooth_start", type=int, default=0,
                   help="Epoch at which the pupil smoothness weight switches on.")
    g.add_argument("--pupil_smooth_ramp", type=int, default=0,
                   help="Ramp the pupil smoothness weight linearly from 0 over this many epochs.")
    g.add_argument("--pupil_smooth_track", action="store_true",
                   help="Track the pupil roughness per epoch even when --pupil_smooth_weight is 0.")

    # initialisation / output
    g = parser.add_argument_group("initialisation / output")
    g.add_argument("--init", type=str, default="amplitude", choices=["ones", "amplitude"],
                   help="Initial spectrum: unit on measured voxels, or the LSQ fit of sqrt(I) with zero phase "
                        "through the initial pupil.")
    g.add_argument("--seed", type=int, default=None, help="Random seed (pupil, minibatch order).")
    g.add_argument("--init_from", type=str, default=None,
                   help="Start from the spectrum and pupil of a previous recon_opt result (same input, --extend, "
                        "--dqz, --axis_flip). With --epochs 0 and --shift_scan_epochs 0 this only scans it.")
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

    # _______________ Warm start _______________
    init_spectrum = init_pupil = None
    if args.init_from:
        with h5py.File(args.init_from, "r") as f:
            R = f["Reconstructed_data"]
            init_spectrum = (R["Fourier_amplitude"][()] * np.exp(1j * R["Fourier_phase"][()]),
                             f["Axes/qz_rad_nm_inv"][()])
            init_pupil = R["Pupil_amplitude"][()] * np.exp(1j * R["Pupil_phase"][()])
            old = dict(f["Recon_Params"].attrs)
        for key, new in (("axis_flip", args.axis_flip), ("extend", args.extend)):
            if key in old and str(old[key]) != str(new):
                raise SystemExit(f"--init_from was run with {key} = {old[key]}, this run has {new}")
        print(f"Warm start from {args.init_from} (epochs {old.get('Num_iters', '?')}, dqz {old.get('dqz', '?')})")

    # _______________ Engine _______________
    pupil_guess = initial_pupil(args.meta_pupil, rng, args.meta_pupil_dset)
    engine = OptimFourierPtycho3D(images=images, kins=kins, kouts=kouts, G=G, wavelength=wavelength,
                                  pupil_func=pupil_guess, ks_pupil=pupil_kins, lr_psize=scan_step_size)
    engine.prepare(extend=args.extend, dqz=dqz, qz_oversample=args.qz_oversample, nz_pad=args.nz_pad,
                   init=args.init, device=args.device, dtype=args.dtype, pupil_model=args.pupil_model,
                   freeze_pupil_amp=args.freeze_pupil_amp,
                   spectrum_mask="measured" if args.mask_unmeasured else None,
                   pixel_threshold=args.pixel_threshold, threshold_mode=args.threshold_mode,
                   streak_shifts=args.streak_shifts, shift_mode=args.shift_mode, shift_max=args.shift_max,
                   subpixel=args.subpixel, placement=args.placement,
                   init_spectrum=init_spectrum, init_pupil=init_pupil)
    pupil_init = engine.pupil_func.copy()
    cover = engine.pupil_cover_mask()
    print(f"Pupil coverage: {100 * cover.sum() / max((np.abs(engine.ctf) > 0).sum(), 1):.1f}% of the aperture")

    opt_cls = torch.optim.Adam if args.optimiser == "adam" else torch.optim.SGD
    engine.set_optimiser(lr_spectrum=args.lr_spectrum,
                         lr_pupil=None if args.fixed_pupil else args.lr_pupil, optimiser=opt_cls,
                         lr_shift=args.lr_shift if args.streak_shifts else None)
    if args.lr_decay is not None:
        engine.set_scheduler(torch.optim.lr_scheduler.ExponentialLR, "spectrum", gamma=args.lr_decay)
        if not args.fixed_pupil:
            engine.set_scheduler(torch.optim.lr_scheduler.ExponentialLR, "pupil", gamma=args.lr_decay)

    # _______________ Optimise _______________
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    regs = []
    if args.tv_weight > 0 or args.tv_track:
        dx, dy = scan_step_size
        axw = (dx / engine.dz, 1.0, dx / dy) if args.tv_per_nm else (1.0, 1.0, 1.0)
        regs.append(TV3D(weight=args.tv_weight, start=args.tv_start, ramp=args.tv_ramp,
                         mode=args.tv_mode, axis_weights=axw))
    if args.support_size is not None:
        regs.append(Support3D(args.support_size, weight=args.support_weight, start=args.support_start,
                              ramp=args.support_ramp, sw_threshold=args.sw_threshold, sw_sigma=args.sw_sigma,
                              sw_every=args.sw_every, sw_start=args.sw_start, sw_in_box=not args.sw_free))
    if args.pupil_smooth_weight > 0 or args.pupil_smooth_track:
        regs.append(PupilSmooth(weight=args.pupil_smooth_weight, start=args.pupil_smooth_start,
                                ramp=args.pupil_smooth_ramp, order=args.pupil_smooth_order))

    engine.run(epochs=args.epochs, batch_size=args.batch_size, pupil_start=args.pupil_start, regularisers=regs,
               alternate=args.alternate, loss=args.loss, normalise_pupil=not args.no_normalise_pupil,
               log_every=args.log_every, seed=args.seed, shift_start=args.shift_start,
               shift_scan_epochs=args.shift_scan_epochs, shift_scan_range=args.shift_scan_range,
               shift_scan_step=args.shift_scan_step)
    peak = torch.cuda.max_memory_allocated() / 1e9 if engine.device.type == "cuda" else float("nan")
    rt = float(np.sum(engine.epoch_times))
    if engine.losses:
        print(f"Optimisation: {args.epochs} epochs, {rt:.0f} s, final loss {engine.losses[-1]:.4e}, "
              f"data misfit {engine.data_misfits[-1]:.4e}, peak GPU memory {peak:.2f} GB")
    if engine.shifts is not None:
        d = engine.shifts.numpy()
        sp = engine._shift_geom["spacing"]
        print(f"Streak shifts ({args.shift_mode}): rms {np.sqrt(np.mean(d**2)):.2f} px, max |.| {np.abs(d).max():.2f} px "
              f"(streak spacing {sp:.2f} px); per streak: " + " ".join(f"{v:+.1f}" for v in d))
    # pupil health: roughness on the covered aperture, and |P| at the image centres vs their neighbours
    rough = pupil_roughness(engine.pupil_func, engine.current_pupil_coverage(), order=args.pupil_smooth_order)
    centre_ratio = float(np.nanmedian(engine.pupil_centre_ratio()))
    print(f"Pupil roughness (order {args.pupil_smooth_order}, covered aperture) = {rough:.4e}; "
          f"|P| at the image centres / median of the 8 neighbours: median {centre_ratio:.3f} "
          f"(1 = no dips; it falls towards 0 before a pixel-pupil blow-up)")
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
        'tv_weight': args.tv_weight,
        'tv_mode': args.tv_mode,
        'tv_start': args.tv_start,
        'tv_ramp': args.tv_ramp,
        'tv_per_nm': args.tv_per_nm,
        'support_size_nm': str(args.support_size),
        'support_weight': args.support_weight,
        'support_start': args.support_start,
        'support_ramp': args.support_ramp,
        'sw_threshold': str(args.sw_threshold),
        'sw_sigma': args.sw_sigma,
        'sw_every': args.sw_every,
        'sw_start': args.sw_start,
        'sw_in_box': not args.sw_free,
        'pupil_smooth_weight': args.pupil_smooth_weight,
        'pupil_smooth_order': args.pupil_smooth_order,
        'pupil_smooth_start': args.pupil_smooth_start,
        'pupil_smooth_ramp': args.pupil_smooth_ramp,
        'pupil_roughness': rough,
        'pupil_centre_ratio': centre_ratio,
        'extend': args.extend,
        'dqz': engine.geom.dqz,
        'dqz_source': args.dqz,
        'qz_oversample': args.qz_oversample,
        'nz_pad': args.nz_pad,
        'mask_unmeasured': args.mask_unmeasured,
        'dz': engine.dz,
        'init': args.init,
        'seed': str(args.seed),
        'G': G,
        'wavelength_nm': wavelength,
        'streak_shifts': str(args.streak_shifts),
        'shift_mode': args.shift_mode,
        'lr_shift': str(args.lr_shift),
        'shift_start': args.shift_start,
        'shift_max': str(args.shift_max),
        'shift_scan_epochs': str(args.shift_scan_epochs),
        'shift_scan_range': str(args.shift_scan_range),
        'shift_scan_step': args.shift_scan_step,
        'subpixel': args.subpixel,
        'placement': args.placement,
        'init_from': str(args.init_from),
        'det_distance_nm': roi.det_distance,
        'delta_q': roi.delta_q,
        'runtime_s': rt,
        'peak_gpu_memory_GB': peak,
    }

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    save_path = str(out_dir / f"recon_opt_scan{roi.scan_num}_{args.axis_flip}_{args.pupil_model}"
                              f"_ep{args.epochs}{'_tv%g' % args.tv_weight if args.tv_weight > 0 else ''}{'_sup%g' % args.support_weight if args.support_size and args.support_weight > 0 else ''}{'_psm%g' % args.pupil_smooth_weight if args.pupil_smooth_weight > 0 else ''}{'_sw' if args.support_size and args.sw_threshold is not None else ''}{('_thr%g%s' % (args.pixel_threshold, 'z' if args.threshold_mode == 'zero' else '')) if args.pixel_threshold is not None else ''}{'_shift' + args.shift_mode + ('N' if args.placement == 'nearest' else '') if args.streak_shifts else ''}{'_sub' if args.subpixel and not args.streak_shifts else ''}{'_warm' if args.init_from else ''}{tag}_{time_str}.h5")
    engine.save_reconsturction(save_path, metadata=meta)
    print(f"Saved {save_path}")


if __name__ == '__main__':

    main()
