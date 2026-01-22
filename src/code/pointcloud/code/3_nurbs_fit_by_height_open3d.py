#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nurbs_fit_by_height_open3d.py

Slice the point cloud by HEIGHT (in the ORIGINAL/world frame), fit a cubic B-spline curve
to each height slice, and visualize in Open3D (points + polylines).

Dependencies:
  pip install open3d scipy numpy pandas
"""
from pathlib import Path
import json
import pickle
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.interpolate import splprep, splev
from collections import defaultdict


def median_nn_distance(pts: np.ndarray, k: int = 8) -> float:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    kdt = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    idxs = np.random.choice(len(pts), size=min(2000, len(pts)), replace=False)
    for i in idxs:
        _, idx, _ = kdt.search_knn_vector_3d(pcd.points[i], k)
        if len(idx) >= 2:
            p0 = np.asarray(pcd.points[i])
            p1 = np.asarray(pcd.points[idx[1]])
            dists.append(np.linalg.norm(p1 - p0))
    return float(np.median(dists)) if len(dists) else 0.0


def pca_axis(points: np.ndarray):
    mean = points.mean(axis=0)
    X = points - mean
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    R = Vt.T
    Xp = X @ R
    return Xp, mean, R


def fit_bspline_curve(points_xyz: np.ndarray, smooth: float, degree: int):
    diffs = np.diff(points_xyz, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    t = np.concatenate([[0.0], np.cumsum(dists)])
    if t[-1] <= 1e-9:
        t = np.linspace(0.0, 1.0, len(points_xyz))
    u = (t - t.min()) / (t.max() - t.min() + 1e-12)
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    k = min(degree, max(1, len(points_xyz) - 1))
    tck, _ = splprep([x, y, z], u=u, s=smooth, k=k)
    return tck, (u.min(), u.max())


def evaluate_tck(tck, umin, umax, n_samples: int):
    us = np.linspace(umin, umax, n_samples)
    x, y, z = splev(us, tck)
    return us, np.vstack([x, y, z]).T


def run_height_slicing_fit(
    pcd_path: Path,
    height_axis: str = "z",
    height_step: float = 0.0,
    half_width_factor: float = 2.0,
    min_points_per_slice: int = 20,
    degree: int = 3,
    smooth: float = 1e-4,
    samples_per_curve: int = 600,
    save_outputs: bool = False,
    outdir: Path = Path("outputs"),
    export_ply: bool = False,
    export_pcd: bool = False,
):
    if not pcd_path.exists():
        raise FileNotFoundError(f"Input PCD not found: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if len(pcd.points) == 0:
        raise RuntimeError("Empty point cloud.")
    pts = np.asarray(pcd.points, dtype=np.float64)

    spans_orig = pts.max(axis=0) - pts.min(axis=0)
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map[height_axis]
    mnn = median_nn_distance(pts, k=8)

    if height_step <= 0:
        step_by_span = (spans_orig[ax] / 150.0) if spans_orig[ax] > 0 else 0.01
        step_by_density = max(8.0 * mnn, 1e-4)
        height_step = max(step_by_span, step_by_density)

    vals = pts[:, ax]
    vmin, vmax = float(vals.min()), float(vals.max())
    num = int(np.ceil((vmax - vmin) / height_step))
    halfw = (height_step * half_width_factor) / 2.0

    rows = []
    tck_bank = {}
    fitted_sizes = []
    fit_count = 0

    for i in range(num + 1):
        c = vmin + i * height_step
        mask = np.abs(vals - c) <= halfw
        idx = np.where(mask)[0]
        if idx.size < min_points_per_slice:
            continue
        slice_pts = pts[idx]

        Xp, mean_s, R_s = pca_axis(slice_pts)
        spans_slice = Xp.max(axis=0) - Xp.min(axis=0)
        order_axis = int(np.argmax(spans_slice))
        sort_idx = np.argsort(Xp[:, order_axis])
        slice_pts_ord = slice_pts[sort_idx]

        if slice_pts_ord.shape[0] > 10000:
            stride = int(np.ceil(slice_pts_ord.shape[0] / 10000))
            slice_pts_ord = slice_pts_ord[::stride]

        try:
            tck, (umin, umax) = fit_bspline_curve(slice_pts_ord, smooth=smooth, degree=degree)
        except Exception as e:
            print(f"[WARN] Height slice {i} fit failed: {e}")
            continue

        us, curve_xyz = evaluate_tck(tck, umin, umax, samples_per_curve)

        for uval, p in zip(us, curve_xyz):
            rows.append((i, float(c), float(uval), float(p[0]), float(p[1]), float(p[2])))

        tck_bank[int(i)] = {
            "height_center": float(c),
            "umin": float(umin),
            "umax": float(umax),
            "tck": tck,
            "slice_size": int(idx.size),
            "height_axis": height_axis,
            "height_step": float(height_step),
            "half_width_factor": float(half_width_factor),
        }
        fit_count += 1
        fitted_sizes.append(int(idx.size))

    if fit_count == 0:
        raise RuntimeError("No height slice was successfully fitted.")

    pcd.paint_uniform_color([0.8, 0.8, 0.8])

    df = pd.DataFrame(rows, columns=["slice_id", "height_center", "t", "x", "y", "z"])
    curve_pts = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    curve_pcd = o3d.geometry.PointCloud()
    curve_pcd.points = o3d.utility.Vector3dVector(curve_pts)

    sid_vals = df["slice_id"].to_numpy()
    sid_norm = (sid_vals - sid_vals.min()) / (sid_vals.max() - sid_vals.min() + 1e-12)
    colors = np.column_stack([sid_norm, 1.0 - sid_norm, 0.5 * np.ones_like(sid_norm)])
    curve_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Build polylines per slice
    line_sets = []
    grouped = defaultdict(list)
    for row in df.itertuples(index=False):
        grouped[int(row.slice_id)].append([row.x, row.y, row.z])
    for sid, arr in grouped.items():
        arr = np.asarray(arr, dtype=float)
        points = o3d.utility.Vector3dVector(arr)
        lines = [[i, i+1] for i in range(len(arr)-1)]
        ls = o3d.geometry.LineSet(points=points, lines=o3d.utility.Vector2iVector(lines))
        col = [float((sid - sid_vals.min()) / (sid_vals.max() - sid_vals.min() + 1e-12)), 0.2, 1.0]
        ls.colors = o3d.utility.Vector3dVector([col for _ in lines])
        line_sets.append(ls)

    o3d.visualization.draw_geometries([pcd, curve_pcd] + line_sets,
                                      window_name=f"B-spline by Height ({height_axis.upper()}-slicing)",
                                      width=1400, height=900, left=40, top=40)

    print("=== Height-Slicing Diagnostics ===")
    print(f"Axis (original frame): {height_axis}  idx={ax}")
    print(f"Spans (orig): {spans_orig}")
    print(f"Median NN distance: {mnn:.6f}")
    print(f"height_step: {height_step:.6f}, half_width_factor: {half_width_factor}")
    print(f"min_points_per_slice: {min_points_per_slice}")
    print(f"#height slices fitted: {fit_count}  (candidates: {num+1})")
    if fitted_sizes:
        print(f"slice sizes: min={min(fitted_sizes)}, median={int(np.median(fitted_sizes))}, max={max(fitted_sizes)}")
    print("==================================")

    if save_outputs:
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / "height_curves_points.csv"
        df.to_csv(csv_path, index=False)

        pkl_path = outdir / "height_tck_params.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(tck_bank, f)

        if export_ply or export_pcd:
            cpcd = o3d.geometry.PointCloud()
            cpcd.points = o3d.utility.Vector3dVector(curve_pts)
            cpcd.colors = o3d.utility.Vector3dVector(colors)
            if export_ply:
                o3d.io.write_point_cloud(str(outdir / "height_curves_points.ply"), cpcd, write_ascii=True)
            if export_pcd:
                o3d.io.write_point_cloud(str(outdir / "height_curves_points.pcd"), cpcd, write_ascii=True)

        manifest = {
            "input_pcd": str(pcd_path),
            "height_axis": height_axis,
            "height_step": float(height_step),
            "half_width_factor": float(half_width_factor),
            "min_points_per_slice": int(min_points_per_slice),
            "degree": int(degree),
            "smooth": float(smooth),
            "samples_per_curve": int(samples_per_curve),
            "csv": str(csv_path) if save_outputs else None,
            "pkl": str(pkl_path) if save_outputs else None,
            "export_ply": bool(export_ply if save_outputs else False),
            "export_pcd": bool(export_pcd if save_outputs else False),
        }
        with open(outdir / "height_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved outputs under: {outdir.resolve()}")
    else:
        print("[INFO] save_outputs=False; nothing written to disk.")

    return df, tck_bank


if __name__ == "__main__":
    # --------------------------- CONFIG ---------------------------
    PCD_PATH = "/home/olivier/wwx/jaka_s5_ws/src/code/pointcloud/uniformized_sink.pcd"  # <-- 改成你的 .pcd 路径
    HEIGHT_AXIS = "x"                        # "x"|"y"|"z" 这里将 z 当作“高度”
    HEIGHT_STEP = 0.0                        # 0 => 自动估计；或填入实际值(注意单位)
    HALF_WIDTH_FACTOR = 2.0                  # 层窗口半宽 = (HEIGHT_STEP * 该因子)/2
    MIN_POINTS_PER_SLICE = 20
    DEGREE = 3
    SMOOTH = 1e-4
    SAMPLES_PER_CURVE = 800
    SAVE_OUTPUTS = False
    OUTDIR = Path("outputs")
    EXPORT_PLY = False
    EXPORT_PCD = False
    # --------------------------- RUN ---------------------------
    df, tck_bank = run_height_slicing_fit(
        pcd_path=Path(PCD_PATH),
        height_axis=HEIGHT_AXIS,
        height_step=HEIGHT_STEP,
        half_width_factor=HALF_WIDTH_FACTOR,
        min_points_per_slice=MIN_POINTS_PER_SLICE,
        degree=DEGREE,
        smooth=SMOOTH,
        samples_per_curve=SAMPLES_PER_CURVE,
        save_outputs=SAVE_OUTPUTS,
        outdir=OUTDIR,
        export_ply=EXPORT_PLY,
        export_pcd=EXPORT_PCD,
    )
