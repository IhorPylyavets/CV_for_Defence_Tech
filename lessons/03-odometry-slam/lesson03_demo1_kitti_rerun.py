"""
Computer Vision для Defence Tech -- Заняття 3 -- Demo 1
KITTI odometry in Rerun: "mapping with a known trajectory" -> "now estimate it yourself".

What the student sees (synchronised on one timeline):
  * the ego pose (GT from OXTS RTK GNSS/INS) as a moving frame with a camera frustum;
  * the current LiDAR scan in the *sensor* frame, placed in the world through T_world_cam0 * Tr_cam0_velo;
  * a map that grows from keyframe scans -- this is the "Mapping" quadrant (trajectory is given);
  * the rectified stereo pair (left + right) with LiDAR points projected into both cameras using P_i and Tr
    from calib.txt -- same point, same image row, shifted by the disparity (link to Lesson 2);
  * optionally, an estimated trajectory (KITTI pose format) next to GT with ATE and drift % -- "Odometry" quadrant.

Usage:
    python lesson03_demo1_kitti_rerun.py --kitti-path /data/kitti --frames 500
    python lesson03_demo1_kitti_rerun.py --frames 500 --save kitti00_500.rrd      # record once, hand students a small .rrd
    python lesson03_demo1_kitti_rerun.py --est-poses my_vo_00.txt                  # overlay your VO/ICP poses
    python lesson03_demo1_kitti_rerun.py --camera color --mono                     # only the left colour camera

Install:
    pip install rerun-sdk numpy Pillow

Expected layout (KITTI odometry):
    <kitti_path>/sequences/00/{velodyne/*.bin, times.txt, calib.txt,
                               image_0..1/*.png (gray stereo, optional), image_2..3/*.png (colour stereo, optional)}
    <kitti_path>/poses/00.txt          (GT poses exist only for sequences 00-10)

Frames used by KITTI odometry:
    poses/XX.txt   -- T_world_cam0, 3x4 row-major; cam0 frame is x right, y down, z forward (RDF)
    calib.txt: Tr  -- T_cam0_velo, maps Velodyne points into cam0
    calib.txt: P0  -- 3x4 projection of cam0 (rectified), K = P0[:, :3]
    calib.txt: P1..P3 -- projections of cam1 (gray right), cam2/cam3 (colour left/right) for points in cam0;
                     rectified, so P_i = K [I | t_i] -- same rotation as cam0, camera centre at -t_i
    velodyne frame -- x forward, y left, z up; sensor is ~1.73 m above the road

Data licence: KITTI is CC BY-NC-SA 3.0 (non-commercial) -- fine for teaching, not for a product.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# ---- helpers ----------------------------------------------------------------

def height_colors(z, z_min=-2.5, z_max=2.0):
    """Colour by height in the Velodyne frame with a FIXED range (no per-frame flicker)."""
    t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    r = np.clip(2 * t - 0.5, 0, 1)
    g = np.clip(2 * t, 0, 1) * (1 - 0.4 * t)
    b = np.clip(1 - 2 * t, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def depth_colors(d, d_max=50.0):
    """Near = red, far = blue, for the image overlay."""
    t = np.clip(d / d_max, 0, 1)
    return (np.stack([1 - t, 0.3 + 0.4 * (1 - abs(2 * t - 1)), t], axis=1) * 255).astype(np.uint8)


def load_velodyne(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_poses(path):
    """KITTI pose file -> (N, 4, 4) homogeneous T_world_cam0."""
    raw = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(raw), 1, 1))
    T[:, :3, :4] = raw
    return T


def load_calib(path):
    """Return ({'P0'..'P3': 3x4}, Tr 4x4) from sequences/XX/calib.txt."""
    vals = {}
    for line in Path(path).read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            vals[k.strip()] = np.array(v.split(), dtype=np.float64)
    if 'Tr' not in vals:
        # data_odometry_color.zip ships a calib.txt with P0-P3 only; Tr comes from data_odometry_calib.zip
        sys.exit(f'{path} has no Tr (Velodyne -> cam0) line.\n'
                 'Extract data_odometry_calib.zip over the dataset (it overwrites calib.txt with the full version).')
    P = {k: vals[k].reshape(3, 4) for k in ('P0', 'P1', 'P2', 'P3') if k in vals}
    Tr = np.eye(4)
    Tr[:3, :4] = vals['Tr'].reshape(3, 4)
    return P, Tr


def cam_offset(P):
    """Rectified P = K [I | t]  ->  t; the camera centre in cam0 coordinates is -t (e.g. cam1 ~0.54 m to the right)."""
    return np.linalg.solve(P[:, :3], P[:, 3])


def try_load_image(path):
    try:
        from PIL import Image
        return np.array(Image.open(path))
    except Exception:
        return None


def voxel_down(xyz, voxel):
    """Cheap voxel filter: keep one point per voxel (first occurrence)."""
    if voxel <= 0:
        return xyz
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def transform(T, xyz):
    return xyz @ T[:3, :3].T + T[:3, 3]


def project(P, xyz_cam, w, h):
    """Project cam0 points with P (any of P0..P3); return pixel uv and depth for points in front and inside the image."""
    front = xyz_cam[:, 2] > 0.5
    pc = xyz_cam[front]
    uvw = pc @ P[:, :3].T + P[:, 3]
    uv = uvw[:, :2] / uvw[:, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[inside], pc[inside, 2]


def drift_stats(gt_xyz, est_xyz):
    """ATE (RMSE, no alignment: both trajectories start at the identity) and end-point drift in % of path length."""
    err = np.linalg.norm(gt_xyz - est_xyz, axis=1)
    length = np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1))
    ate = float(np.sqrt(np.mean(err ** 2)))
    drift = float(100 * err[-1] / length) if length > 1 else 0.0
    return ate, drift


# ---- args -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    artifacts = Path(__file__).resolve().parents[2] / 'artifacts' / 'kitti'      # where tools/fetch_kitti.py puts it
    p.add_argument('--kitti-path', type=Path, default=artifacts)
    p.add_argument('--sequence', default='00')
    p.add_argument('--start', type=int, default=0, help='First frame index')
    p.add_argument('--frames', type=int, default=500, help='Number of source frames to cover (before --every)')
    p.add_argument('--every', type=int, default=1, help='Use every Nth frame')
    p.add_argument('--point-stride', type=int, default=2, help='Keep every Nth LiDAR point of the live scan')
    p.add_argument('--keyframe-every', type=int, default=10, help='Add a scan to the map every N used frames (0 = no map)')
    p.add_argument('--map-voxel', type=float, default=0.3, help='Voxel size of map keyframes, m')
    p.add_argument('--est-poses', type=Path, default=None, help='Estimated trajectory in KITTI pose format')
    p.add_argument('--camera', choices=['auto', 'gray', 'color'], default='auto',
                   help='Stereo pair: gray = image_0/1, color = image_2/3, auto = gray if present, else colour')
    p.add_argument('--mono', action='store_true', help='Show only the left camera of the pair')
    p.add_argument('--save', type=Path, default=None, help='Write an .rrd file instead of spawning the viewer')
    return p.parse_args()


# ---- main -------------------------------------------------------------------

def main():
    args = parse_args()
    seq = args.kitti_path / 'sequences' / args.sequence
    velo_dir = seq / 'velodyne'
    poses_path = args.kitti_path / 'poses' / f'{args.sequence}.txt'

    if not velo_dir.exists():
        print(f'KITTI sequence not found: {seq}')
        print('Download from the repo root:  python tools/fetch_kitti.py   (see artifacts/kitti/README.md)')
        sys.exit(1)

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError:
        print('rerun-sdk not installed:  pip install rerun-sdk')
        sys.exit(1)

    # One index list drives every stream, so scans, images, timestamps and poses can never desynchronise.
    all_bins = sorted(velo_dir.glob('*.bin'))
    idx = list(range(args.start, min(args.start + args.frames, len(all_bins)), args.every))
    times = np.loadtxt(seq / 'times.txt')
    P, Tr = load_calib(seq / 'calib.txt')
    poses = load_poses(poses_path) if poses_path.exists() else None
    est = load_poses(args.est_poses) if args.est_poses else None
    if est is not None and poses is None:
        print('--est-poses needs GT poses to compare against (sequences 00-10); ignoring the estimate.')
        est = None

    # Cameras to show as (KITTI id, label), left first; the right one is dropped with --mono or if its folder is missing.
    pairs = {'gray': [(0, 'ліва'), (1, 'права')], 'color': [(2, 'ліва'), (3, 'права')]}
    kind = args.camera
    if kind == 'auto':
        kind = 'gray' if (seq / 'image_0').exists() else 'color'
    cams = [(c, label) for c, label in pairs[kind][:1 if args.mono else 2]
            if (seq / f'image_{c}').exists() and f'P{c}' in P]

    cam_names = ', '.join(f'image_{c}' for c, _ in cams) or 'none'
    print(f'Sequence {args.sequence}: {len(idx)} frames | GT poses: {"yes" if poses is not None else "no"} | '
          f'cameras: {cam_names} | estimate: {"yes" if est is not None else "no"}')
    if poses is None:
        print('No GT poses (sequences 11-21): scans will be shown at the origin, map disabled.')

    rr.init('cv_deftech_l3_kitti', spawn=args.save is None)
    if args.save:
        rr.save(str(args.save))

    views = [rrb.Spatial3DView(name='Світ: скан + карта + траєкторія', origin='/world')]
    if cams:
        # KITTI frames are ~1241x376, so the pair reads best stacked: left above right, image rows line up.
        cam_views = [rrb.Spatial2DView(name=f'Камера {c} ({label}) + LiDAR', origin=f'/world/ego/cam{c}')
                     for c, label in cams]
        views.append(cam_views[0] if len(cam_views) == 1 else rrb.Vertical(*cam_views))
    if est is not None:
        views.append(rrb.TimeSeriesView(name='Похибка оцінки, м', origin='/metrics'))
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views, column_shares=[3, 2, 2][:len(views)])))

    # KITTI world = cam0 of frame 0, i.e. Right-Down-Forward. Tell the viewer, instead of hand-rotating data.
    rr.log('world', rr.ViewCoordinates.RDF, static=True)

    # Static sensor rig. The ego frame IS cam0 (KITTI poses are T_world_cam0): Velodyne -> cam0 via Tr,
    # every camera is cam0 shifted by -t_i (rectified, so no rotation) with its own pinhole K = P_i[:, :3].
    rr.log('world/ego/velodyne',
           rr.Transform3D(translation=Tr[:3, 3], mat3x3=Tr[:3, :3]), static=True)
    for c, _ in cams:
        first_img = try_load_image(sorted((seq / f'image_{c}').glob('*.png'))[0])
        if first_img is None:
            continue
        h, w = first_img.shape[:2]
        rr.log(f'world/ego/cam{c}', rr.Transform3D(translation=-cam_offset(P[f'P{c}'])), static=True)
        rr.log(f'world/ego/cam{c}',
               rr.Pinhole(image_from_camera=P[f'P{c}'][:, :3], width=w, height=h, image_plane_distance=1.5),
               static=True)
    if len(cams) == 2:
        baseline = np.linalg.norm(cam_offset(P[f'P{cams[1][0]}']) - cam_offset(P[f'P{cams[0][0]}']))
        print(f'Stereo baseline: {baseline:.3f} m')

    if poses is not None:
        gt_xyz = poses[idx, :3, 3]
        rr.log('world/gt_full', rr.LineStrips3D([gt_xyz], colors=[[120, 130, 145]], radii=0.05), static=True)
    if est is not None:
        est_xyz = est[idx, :3, 3]

    t_wall = time.time()
    for k, i in enumerate(idx):
        rr.set_time('frame', sequence=i)
        # times.txt holds seconds since the start of the sequence -> a *duration*, not a Unix timestamp.
        rr.set_time('time', duration=float(times[i]))

        T_wc = poses[i] if poses is not None else np.eye(4)
        rr.log('world/ego', rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]))

        scan = load_velodyne(all_bins[i])[:, :3]
        live = scan[::args.point_stride]
        rr.log('world/ego/velodyne/scan',
               rr.Points3D(live, colors=height_colors(live[:, 2]), radii=0.03))

        # Map from keyframes: each keyframe is its own entity, so scrubbing back in time hides "future" map.
        if poses is not None and args.keyframe_every and k % args.keyframe_every == 0:
            kf = voxel_down(transform(T_wc @ Tr, scan), args.map_voxel)
            # World is RDF (y down) with origin at cam0 of frame 0, ~1.65 m above the road: height = 1.65 - y.
            col = (height_colors(1.65 - kf[:, 1], 0.0, 6.0) * 0.6).astype(np.uint8)
            rr.log(f'world/map/kf_{i:06d}', rr.Points3D(kf, colors=col, radii=0.05))

        if poses is not None and k % 5 == 0:
            rr.log('world/gt_so_far', rr.LineStrips3D([gt_xyz[:k + 1]], colors=[[46, 196, 182]], radii=0.12))

        if est is not None and k % 5 == 0:
            rr.log('world/estimate', rr.LineStrips3D([est_xyz[:k + 1]], colors=[[255, 92, 92]], radii=0.12))
            ate, _ = drift_stats(gt_xyz[:k + 1], est_xyz[:k + 1])
            rr.log('metrics/position_error', rr.Scalars(float(np.linalg.norm(gt_xyz[k] - est_xyz[k]))))
            rr.log('metrics/ate_rmse', rr.Scalars(ate))

        scan_cam0 = transform(Tr, scan) if cams else None
        for c, _ in cams:
            img = try_load_image(seq / f'image_{c}' / f'{i:06d}.png')
            if img is not None:
                rr.log(f'world/ego/cam{c}/image', rr.Image(img))
                # P_i already contains the cam0 -> cam_i shift, so the same cam0 points go into every image.
                uv, d = project(P[f'P{c}'], scan_cam0, img.shape[1], img.shape[0])
                rr.log(f'world/ego/cam{c}/lidar_uv', rr.Points2D(uv, colors=depth_colors(d), radii=1.2))

        if (k + 1) % 50 == 0 or k == len(idx) - 1:
            print(f'  frame {k + 1}/{len(idx)}  ({(k + 1) / (time.time() - t_wall + 1e-9):.0f} frames/s logged)',
                  end='\r')

    print(f'\nLogged {len(idx)} frames in {time.time() - t_wall:.1f} s')
    if est is not None:
        ate, drift = drift_stats(gt_xyz, est_xyz)
        print(f'Estimate vs GT: ATE {ate:.2f} m, end-point drift {drift:.2f} % of path')
    if args.save:
        print(f'Saved {args.save}  ->  open with:  rerun {args.save}')
    print('Space = play, drag the timeline. Grey line = full GT, teal = GT so far, red = your estimate.')


if __name__ == '__main__':
    main()
