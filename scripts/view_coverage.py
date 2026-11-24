"""
Reciprocal-space coverage of a Bragg Fourier-ptychography dataset, viewed with pyvista.

Same geometry as the 3D engines (phase_abstract_3d._qz_samples / _build_geometry): every coherent
image j (one detector pixel) samples the cap

    q_z,j(K) = k_out,z - sqrt(k^2 - |k_in0,j + K|^2) - G_z ,   k_in0,j = k_out,j - G

over the lateral scan-frequency grid K (the lateral spectrum coordinate is K itself), and each
sample is linearly interpolated onto the two neighbouring q_z nodes of the uniform grid.

Subcommands
-----------
  export : load a ptyxplore ROI (needs ptyxplore + hdf5plugin) and write a small geometry .npz
  volume : 3D view of the covered (q_z, K0, K1) volume (number of samples per voxel)
  pixel  : interactive - choose a detector pixel (sliders or right-click on the detector map)
           and see the cap it samples inside the covered volume

`volume` and `pixel` need only numpy + pyvista when given an .npz, so you can export on Maxwell
and view locally; they also accept --input/--pupil_file directly.

Examples
--------
  python view_coverage.py export --input roi-A_scan-410_binned4_streaked.h5 --out scan410_geom.npz
  python view_coverage.py volume scan410_geom.npz
  python view_coverage.py volume scan410_geom.npz --weight intensity --style volume
  python view_coverage.py pixel  scan410_geom.npz --show_streak
  python view_coverage.py volume --input roi.h5 --pupil_file pupil.h5 --refit_G
"""
import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------- geometry

class CoverageGeometry:
    """Per-image k vectors + K grid, and the q_z grid of the 3D engines."""

    KEYS = ("kouts", "kins", "G", "k", "img_shape", "dk", "kout_coords", "mean_I", "streak")

    def __init__(self, kouts, kins, G, k, img_shape, dk, kout_coords, mean_I, streak,
                 dqz=None, qz_oversample=1.0, nz_pad=2):
        self.kouts = np.asarray(kouts, float)
        self.kins = np.asarray(kins, float)
        self.G = np.asarray(G, float)
        self.k = float(k)
        self.n0, self.n1 = (int(n) for n in img_shape)
        self.dk0, self.dk1 = (float(d) for d in dk)
        self.kout_coords = np.asarray(kout_coords, int)
        self.mean_I = np.asarray(mean_I, float)
        self.streak = np.asarray(streak, int)
        self.N = len(self.kouts)
        self.K0 = (np.arange(self.n0) - self.n0 // 2) * self.dk0
        self.K1 = (np.arange(self.n1) - self.n1 // 2) * self.dk1
        self.qz0 = self.qz_cap_point(np.arange(self.N))
        self.dqz_pixels = self._dqz_from_pixels()
        self.set_grid(dqz, qz_oversample, nz_pad)

    # -- I/O
    @classmethod
    def load(cls, path, **grid_kw):
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.KEYS}, **grid_kw)

    # -- q_z
    def qz_cap(self, j):
        """q_z over the K grid for image(s) j: (n0, n1) or (len(j), n0, n1)."""
        scalar = np.ndim(j) == 0
        j = np.atleast_1d(j)
        kin0 = self.kins[j, 0, None, None] + self.K0[None, :, None]
        kin1 = self.kins[j, 1, None, None] + self.K1[None, None, :]
        kz = np.sqrt(np.clip(self.k**2 - kin0**2 - kin1**2, 0.0, None))
        q = self.kouts[j, 2, None, None] - kz - self.G[2]
        return q[0] if scalar else q

    def qz_cap_point(self, j):
        """q_z at K = 0 for images j."""
        kz = np.sqrt(np.clip(self.k**2 - self.kins[j, 0]**2 - self.kins[j, 1]**2, 0.0, None))
        return self.kouts[j, 2] - kz - self.G[2]

    def _dqz_from_pixels(self):
        rc = self.kout_coords - self.kout_coords.min(axis=0)
        grid = np.full(rc.max(axis=0) + 1, np.nan)
        grid[rc[:, 0], rc[:, 1]] = self.qz0
        if min(grid.shape) < 2:
            return float(np.nanmedian(np.abs(np.diff(np.sort(self.qz0)))))
        g0, g1 = np.gradient(grid)
        return float(np.nanmedian(np.hypot(g0, g1)))

    def _chunks(self, size=256):
        for a in range(0, self.N, size):
            yield np.arange(a, min(a + size, self.N))

    def set_grid(self, dqz=None, qz_oversample=1.0, nz_pad=2):
        """Uniform q_z grid exactly as PhaseRetrieval3DBase._build_geometry."""
        dqz = (self.dqz_pixels if dqz is None else float(dqz)) / qz_oversample
        qmax = max(np.abs(self.qz_cap(j)).max() for j in self._chunks())
        Nz = 2 * int(np.ceil(qmax / dqz)) + 2 * nz_pad + 2
        Nz += Nz % 2
        self.dqz, self.Nz = dqz, Nz
        self.qz_axis = (np.arange(Nz) - Nz // 2) * dqz
        self.dz = 2 * np.pi / (Nz * dqz)

    def counts(self, weight="count"):
        """(Nz, n0, n1): interpolation weight landing on each voxel, summed over all samples.
        weight='intensity' multiplies each image's samples by its mean intensity."""
        M = self.n0 * self.n1
        out = np.zeros(self.Nz * M)
        lat = np.arange(M).reshape(self.n0, self.n1)
        for j in self._chunks():
            f = self.qz_cap(j) / self.dqz + self.Nz // 2
            j0 = np.floor(f).astype(np.int64)
            w = f - j0
            wi = 1.0 if weight == "count" else self.mean_I[j][:, None, None]
            node0 = (j0 * M + lat[None]).ravel()
            out += np.bincount(node0, weights=((1 - w) * wi).ravel(), minlength=self.Nz * M)
            out += np.bincount(node0 + M, weights=(w * wi).ravel(), minlength=self.Nz * M)
        return out.reshape(self.Nz, self.n0, self.n1)

    def summary(self, cnt):
        cov = cnt > 0
        sl = np.flatnonzero(cov.any(axis=(1, 2)))
        dq = (sl[-1] - sl[0] + 1) * self.dqz
        return (f"{self.N} images {self.n0}x{self.n1} | grid Nz={self.Nz}, dqz={self.dqz:.3e} rad/nm, "
                f"dz={self.dz:.1f} nm\ncovered: {100 * cov.mean():.1f}% of the box, q_z slices "
                f"{sl[0]}..{sl[-1]} of 0..{self.Nz - 1} (dq_z = {dq:.3f} rad/nm -> z resolution ~ "
                f"{2 * np.pi / dq:.0f} nm)")


def export_geometry(input_path, pupil_file=None, refit_G=False, ttheta_deg=None):
    """Load a ptyxplore ROI (streak-split or flat) into a CoverageGeometry (needs ptyxplore)."""
    import hdf5plugin  # noqa: F401
    import ptyxplore.loader as db
    import ptyxplore.roi_ops as dp
    from ptyxplore.kvectors import fit_global_G, kins_from_G

    def as_list(v):
        return v if isinstance(v, list) else [v]

    roi = db.ROI.load_data(input_path)
    images = as_list(dp.flip_images(roi, "xy"))
    kouts = [np.asarray(ko, float) for ko in as_list(roi.kouts)]
    coords = [np.asarray(c) for c in as_list(roi.kout_coords)]
    streak = np.concatenate([np.full(len(ko), s if len(kouts) > 1 else -1) for s, ko in enumerate(kouts)])
    mean_I = np.concatenate([np.asarray(im, float).mean(axis=(1, 2)) for im in images])
    kouts_f = np.concatenate(kouts)
    k = float(np.median(np.linalg.norm(kouts_f, axis=1)))

    G = getattr(roi, "G", None)
    if G is None or refit_G:
        if pupil_file is None:
            raise SystemExit("no global G stored in the input: pass --pupil_file (and --refit_G)")
        pupil_kins = db.ROI.load_data(pupil_file).kins
        if isinstance(pupil_kins, list):
            raise SystemExit(f"{pupil_file} is streak-split; use the unsplit pupil file")
        ttheta = None if ttheta_deg is None else np.deg2rad(ttheta_deg)
        G, info = fit_global_G(kouts_f, pupil_kins, 2 * np.pi / k, weights=mean_I, ttheta=ttheta)
        kins_f = kins_from_G(kouts_f, G, k)
        print(f"Global G fitted: {G}, 2θ = {np.rad2deg(info['ttheta']):.4f} deg")
    else:
        kins_f = np.concatenate([np.asarray(ki, float) for ki in as_list(roi.kins)])
        print(f"Using global G stored in the input file: {np.asarray(G)}")

    n0, n1 = np.asarray(images[0]).shape[-2:]
    step = np.abs([roi.step_size_y, roi.step_size_x])        # rows = scan_y, as recon_opt.py
    dk = (2 * np.pi / (n0 * step[0]), 2 * np.pi / (n1 * step[1]))
    return dict(kouts=kouts_f, kins=kins_f, G=np.asarray(G, float), k=k, img_shape=(n0, n1), dk=dk,
                kout_coords=np.concatenate(coords), mean_I=mean_I, streak=streak)


def get_geometry(args):
    grid_kw = dict(dqz=args.dqz, qz_oversample=args.qz_oversample, nz_pad=args.nz_pad)
    if args.geom is not None:
        return CoverageGeometry.load(args.geom, **grid_kw)
    if args.input is None:
        raise SystemExit("give a geometry .npz or --input")
    return CoverageGeometry(**export_geometry(args.input, args.pupil_file, args.refit_G, args.ttheta_deg),
                            **grid_kw)


# ---------------------------------------------------------------------------- pyvista helpers

def coverage_grid(geo, cnt, zscale):
    """pyvista ImageData: x = K0, y = K1, z = q_z * zscale (all rad/nm)."""
    import pyvista as pv
    grid = pv.ImageData(dimensions=(geo.n0, geo.n1, geo.Nz),
                        spacing=(geo.dk0, geo.dk1, geo.dqz * zscale),
                        origin=(geo.K0[0], geo.K1[0], geo.qz_axis[0] * zscale))
    grid.point_data["samples"] = np.transpose(cnt, (1, 2, 0)).ravel(order="F")
    return grid


def axes_labels(zscale):
    zl = "q_z (rad/nm)" if zscale == 1 else f"q_z x {zscale:g} (rad/nm)"
    return dict(xtitle="K0 (rad/nm)", ytitle="K1 (rad/nm)", ztitle=zl, fmt="%.3f")


def cap_points(geo, j, zscale, stride=1):
    K0, K1 = np.meshgrid(geo.K0[::stride], geo.K1[::stride], indexing="ij")
    q = geo.qz_cap(j)[::stride, ::stride]
    return K0, K1, q * zscale, q


# ---------------------------------------------------------------------------- views

def view_volume(geo, args):
    import pyvista as pv
    cnt = geo.counts(args.weight)
    print(geo.summary(cnt))
    grid = coverage_grid(geo, cnt, args.zscale)
    covered = grid.threshold(1e-9 * cnt.max(), scalars="samples")
    if args.log:
        covered.point_data["samples"] = np.log10(covered.point_data["samples"] + 1e-12 * cnt.max())
    label = ("log10 " if args.log else "") + ("samples / voxel" if args.weight == "count"
                                              else "intensity-weighted samples / voxel")

    pl = pv.Plotter(off_screen=args.screenshot is not None, window_size=(1200, 900))
    pl.add_text(geo.summary(cnt), font_size=9)
    pl.add_mesh(grid.outline(), color="grey")
    if args.style == "volume":
        vol = grid.copy()
        if args.log:
            vol.point_data["samples"] = np.log10(vol.point_data["samples"] + 1e-12 * cnt.max())
        pl.add_volume(vol, scalars="samples", cmap="viridis", opacity="sigmoid",
                      scalar_bar_args=dict(title=label))
    elif args.screenshot is None:
        pl.add_mesh_clip_plane(covered, scalars="samples", cmap="viridis", opacity=args.opacity,
                               scalar_bar_args=dict(title=label), normal="x")
    else:
        pl.add_mesh(covered, scalars="samples", cmap="viridis", opacity=args.opacity,
                    scalar_bar_args=dict(title=label))
    pl.show_grid(**axes_labels(args.zscale))
    pl.view_isometric()

    # inset: per q_z slice, fraction of the K plane covered and relative number of samples
    chart = pv.Chart2D(size=(0.3, 0.3), loc=(0.68, 0.62), x_label="q_z (rad/nm)", y_label="")
    frac = (cnt > 0).mean(axis=(1, 2))
    per = cnt.sum(axis=(1, 2))
    chart.line(geo.qz_axis, frac, color="tab:blue", width=2, label="K-plane covered")
    chart.line(geo.qz_axis, per / per.max(), color="tab:orange", width=2,
               label=("samples" if args.weight == "count" else "intensity") + " (rel.)")
    chart.y_range = [0, 1.05]
    chart.legend_visible = True
    pl.add_chart(chart)

    if args.screenshot:
        pl.screenshot(args.screenshot)
        print(f"saved {args.screenshot}")
    else:
        pl.show()


def view_pixel(geo, args):
    import pyvista as pv
    pv.global_theme.allow_empty_mesh = True          # the streak mesh starts empty
    cnt = geo.counts("count")
    print(geo.summary(cnt))
    grid = coverage_grid(geo, cnt, args.zscale)
    covered = grid.threshold(1e-9 * cnt.max(), scalars="samples")
    rows, cols = geo.kout_coords[:, 0], geo.kout_coords[:, 1]
    qlim = (geo.qz_axis[0], geo.qz_axis[-1])
    stride = max(1, int(args.stride))

    pl = pv.Plotter(shape=(1, 2), off_screen=args.screenshot is not None, window_size=(1600, 800))

    # ---- right: detector map (col -> x, row -> y), coloured by q_z(K=0)
    pl.subplot(0, 1)
    det = pv.PolyData(np.c_[cols, -rows, np.zeros(geo.N)].astype(float))   # rows downwards
    det.point_data["q_z(0) (rad/nm)"] = geo.qz0
    pl.add_mesh(det, scalars="q_z(0) (rad/nm)", cmap="coolwarm", clim=qlim, point_size=8,
                render_points_as_spheres=True, pickable=True,
                scalar_bar_args=dict(title="q_z(K=0) (rad/nm)", vertical=True))
    marker = pv.PolyData(np.array([[cols[0], -rows[0], 0.5]], float))
    pl.add_mesh(marker, color="black", point_size=18, render_points_as_spheres=True, pickable=False)
    pl.add_text("detector pixels (right-click / 'p' to pick)", font_size=10)
    pl.view_xy()
    h = np.ptp(rows) + 1
    pl.reset_camera(bounds=(cols.min() - 1, cols.max() + 1, -rows.max() - 0.8 * h, -rows.min() + 1, 0, 1))
    det_renderer = pl.renderer

    # ---- left: covered volume + cap of the selected pixel
    pl.subplot(0, 0)
    pl.add_mesh(grid.outline(), color="grey", pickable=False)
    pl.add_mesh(covered.extract_surface(), color="lightsteelblue", opacity=args.opacity, pickable=False)
    K0, K1, Z, q = cap_points(geo, 0, args.zscale, stride)
    cap = pv.StructuredGrid(K0, K1, Z)
    cap.point_data["dq_z"] = np.zeros(cap.n_points)
    cap_actor = pl.add_mesh(cap, scalars="dq_z", cmap="coolwarm", clim=(-1, 1), pickable=False,
                            scalar_bar_args=dict(title="q_z - q_z(K=0)  (dqz)"))
    streak_mesh = pv.PolyData()
    streak_actor = pl.add_mesh(streak_mesh, color="black", style="wireframe", opacity=0.25, pickable=False)
    streak_actor.SetVisibility(bool(args.show_streak))
    info = pl.add_text("", position="upper_right", font_size=9)
    pl.show_grid(**axes_labels(args.zscale))
    pl.view_isometric()

    state = dict(j=0, row=int(np.median(rows)), col=int(np.median(cols)))

    def streak_caps(j):
        s = geo.streak[j]
        if s < 0:
            return pv.PolyData()
        members = np.flatnonzero(geo.streak == s)[::max(1, int(args.streak_every))]
        meshes = []
        for m in members:
            a, b, z, _ = cap_points(geo, m, args.zscale, max(stride, 4))
            meshes.append(pv.StructuredGrid(a, b, z).extract_surface())
        return pv.merge(meshes) if meshes else pv.PolyData()

    def select(j):
        j = int(j)
        state["j"] = j
        _, _, z, qq = cap_points(geo, j, args.zscale, stride)
        cap.points[:, 2] = z.ravel(order="F")
        rel = (qq - geo.qz0[j]) / geo.dqz
        cap.point_data["dq_z"] = rel.ravel(order="F")
        m = max(np.abs(rel).max(), 1e-3)
        cap_actor.mapper.scalar_range = (-m, m)
        marker.points = np.array([[cols[j], -rows[j], 0.5]], float)
        if streak_actor.GetVisibility():
            streak_mesh.copy_from(streak_caps(j))
        qc = geo.qz_cap(j)
        g0 = (qc[-1, geo.n1 // 2] - qc[0, geo.n1 // 2]) / (geo.K0[-1] - geo.K0[0])
        g1 = (qc[geo.n0 // 2, -1] - qc[geo.n0 // 2, 0]) / (geo.K1[-1] - geo.K1[0])
        info.SetText(3, (
            f"pixel row {rows[j]}, col {cols[j]}   image {j}"
            + (f"   streak {geo.streak[j]}" if geo.streak[j] >= 0 else "")
            + f"\nq_z(K=0) = {geo.qz0[j]:+.4f} rad/nm = {geo.qz0[j] / geo.dqz:+.2f} dqz"
            + f"\nq_z on cap: {qc.min():+.4f} .. {qc.max():+.4f}  (span {np.ptp(qc) / geo.dqz:.2f} dqz)"
            + f"\ntilt dq_z/dK = ({g0:+.4f}, {g1:+.4f})   k_in0/k_z = "
            + f"({geo.kins[j, 0] / geo.kins[j, 2]:+.4f}, {geo.kins[j, 1] / geo.kins[j, 2]:+.4f})"
            + f"\nmean intensity {geo.mean_I[j]:.3g}"))
        pl.render()

    def nearest(row, col):
        return int(np.argmin((rows - row)**2 + (cols - col)**2))

    def on_row(v):
        state["row"] = int(round(v))
        select(nearest(state["row"], state["col"]))

    def on_col(v):
        state["col"] = int(round(v))
        select(nearest(state["row"], state["col"]))

    def on_pick(point, picker=None):
        if point is None:
            return
        if picker is not None and picker.GetRenderer() is not det_renderer:
            return
        select(nearest(-point[1], point[0]))

    def on_streak_toggle(flag):
        streak_actor.SetVisibility(bool(flag))
        if flag:
            streak_mesh.copy_from(streak_caps(state["j"]))
        pl.render()

    pl.subplot(0, 1)
    pl.add_slider_widget(on_row, (rows.min(), rows.max()), value=state["row"], title="detector row",
                         pointa=(0.1, 0.08), pointb=(0.9, 0.08), style="modern", fmt="%.0f")
    pl.add_slider_widget(on_col, (cols.min(), cols.max()), value=state["col"], title="detector col",
                         pointa=(0.1, 0.22), pointb=(0.9, 0.22), style="modern", fmt="%.0f")
    pl.subplot(0, 0)
    if geo.streak.max() >= 0:
        pl.add_checkbox_button_widget(on_streak_toggle, value=bool(args.show_streak), size=30,
                                      position=(10, 10))
        pl.add_text("show caps of the same streak", position=(50, 15), font_size=9)
    pl.enable_point_picking(on_pick, use_picker=True, show_message=False, show_point=False,
                            left_clicking=False)
    select(nearest(state["row"], state["col"]))

    if args.screenshot:
        pl.screenshot(args.screenshot)
        print(f"saved {args.screenshot}")
    else:
        pl.show()


# ---------------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp, need_input=False):
        sp.add_argument("--input", required=need_input, help="ptyxplore ROI h5 (streak-split or flat)")
        sp.add_argument("--pupil_file", default=None, help="pupil ROI h5 (only needed to fit G)")
        sp.add_argument("--refit_G", action="store_true", help="fit a global G even if one is stored")
        sp.add_argument("--ttheta_deg", type=float, default=None, help="2θ for the G fit")

    def grid_args(sp):
        sp.add_argument("geom", nargs="?", default=None, help="geometry .npz from `export`")
        data_args(sp)
        sp.add_argument("--dqz", type=float, default=None, help="q_z step (rad/nm); default from pixels")
        sp.add_argument("--qz_oversample", type=float, default=1.0)
        sp.add_argument("--nz_pad", type=int, default=2)
        sp.add_argument("--zscale", type=float, default=1.0, help="stretch q_z in the display")
        sp.add_argument("--opacity", type=float, default=0.35)
        sp.add_argument("--screenshot", default=None, help="render off-screen to this PNG instead")

    sp = sub.add_parser("export", help="write the geometry .npz (needs ptyxplore)")
    data_args(sp, need_input=True)
    sp.add_argument("--out", required=True)

    sp = sub.add_parser("volume", help="covered reciprocal-space volume")
    grid_args(sp)
    sp.add_argument("--weight", choices=["count", "intensity"], default="count",
                    help="samples per voxel, or weighted by each image's mean intensity")
    sp.add_argument("--style", choices=["surface", "volume"], default="surface",
                    help="covered voxels with a clip plane, or volume rendering")
    sp.add_argument("--log", action="store_true", help="log10 colour scale")

    sp = sub.add_parser("pixel", help="interactive: detector pixel -> sampled cap")
    grid_args(sp)
    sp.add_argument("--show_streak", action="store_true", help="start with the streak's caps shown")
    sp.add_argument("--streak_every", type=int, default=1, help="draw every n-th cap of the streak")
    sp.add_argument("--stride", type=int, default=1, help="K subsampling of the drawn cap")

    args = p.parse_args(argv)
    if args.cmd == "export":
        d = export_geometry(args.input, args.pupil_file, args.refit_G, args.ttheta_deg)
        np.savez_compressed(args.out, **d)
        print(f"saved {args.out} ({len(d['kouts'])} images)")
        return
    geo = get_geometry(args)
    (view_volume if args.cmd == "volume" else view_pixel)(geo, args)


if __name__ == "__main__":
    main()
