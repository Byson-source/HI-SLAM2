import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))   # nopep8
import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import lietorch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from hi2 import Hi2


def show_image(image, depth_prior, depth, normal):
    from util.utils import colorize_np
    image = image[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    depth = colorize_np(np.concatenate((depth_prior.cpu().numpy(), depth.cpu().numpy()), axis=1), range=(0, 4))
    normal = normal.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('rgb / prior normal / aligned prior depth / JDSA depth', np.concatenate((image / 255.0, (normal[...,[2,1,0]]+1.)/2., depth), axis=1)[::2,::2])
    cv2.waitKey(1)


def mono_stream(queue, imagedir, calib, undistort=False, cropborder=False, start=0, length=100000):
    """ image generator """
    RES = 341 * 640
    # Optional working-resolution downscale: HISLAM2_RES_DIV=k shrinks the processing
    # resolution linearly by k (area / k^2). Fewer pixels -> fewer per-keyframe gaussians
    # and less GPU memory, which avoids the mid-sequence stall on long handheld captures
    # (e.g. LaMAR). Default 1.0 = unchanged. Intrinsics scale with the resize below.
    res_div = float(os.environ.get('HISLAM2_RES_DIV', '1.0'))
    if res_div > 1.0:
        RES = RES / (res_div * res_div)

    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])

    image_list = sorted(os.listdir(imagedir))[start:start+length]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intrinsics = torch.tensor(calib[:4])
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        image = cv2.resize(image, (w1, h1))
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)

        is_last = (t == len(image_list)-1)
        queue.put((t, image[None], intrinsics[None], is_last))

    time.sleep(10)


def save_trajectory(hi2, traj_full, imagedir, output, start=0):
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), hi2.video.intrinsics[0].cpu().numpy()*8)

    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in sorted(os.listdir(imagedir))[start:]])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    np.savetxt(f"{output}/traj_kf.txt", ttraj_kf)                     #  for evo evaluation 
    if traj_full is not None:
        ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full.txt", ttraj_full)


# okvis_pose_graph_msgs/PoseGraph definition (vendored so the bag is self-contained).
POSE_GRAPH_MSGDEF = """\
std_msgs/Header        header
uint64[]               vertex_id
int64[]                vertex_stamp_ns
geometry_msgs/Pose[]   vertex_pose
uint64[]               edge_i
uint64[]               edge_j
geometry_msgs/Pose[]   edge_rel
float64[]              edge_info_trans
float64[]              edge_info_rot
uint8[]                edge_type
"""

POSE_GRAPH_CONVENTION = (
    "vertex_pose/edge_rel = [tx,ty,tz,qx,qy,qz,qw]; "
    "vertex_pose = T_WS (world<-sensor); edge_rel = T_AB = T_WA^-1*T_WB (A<-B); "
    "info = 1/sigma^2 diagonal")


def build_pose_graph(hi2, imagedir, start=0, covis_thresh=None,
                     info_scale=500.0, covis_max=50.0, n=None):
    """ build the keyframe pose graph following the ``okvis_pose_graph_msgs/PoseGraph``
    schema (see zfloc_pgo_handoff.md §2). The returned dict is consumed both by
    save_pose_graph (npz) and save_rosbag (ROS2 bag). Aligned arrays, vertices
    indexed 0..N-1:

      vertex_id        int64   [N]    keyframe id (= keyframe index)
      vertex_stamp_ns  int64   [N]    timestamp parsed from the image filename
      vertex_pose      float64 [N,7]  T_WS world<-sensor as [tx,ty,tz, qx,qy,qz,qw]
      edge_i, edge_j   int64   [E]    vertex ids of the two edge ends (A, B)
      edge_rel         float64 [E,7]  T_AB = T_WA^-1 * T_WB  (A<-B)
      edge_info_trans  float64 [E]    translational diagonal information (1/sigma^2)
      edge_info_rot    float64 [E]    rotational diagonal information
      edge_type        uint8   [E]    0 = sequential VO, 1 = covisibility/loop

    Covisibility weighting (mirrors the OKVIS2-X design ``info = K*max(1,covis)``):
    DROID/HI-SLAM2 has no integer landmark count, so the per-edge covisibility
    strength is derived from the dense flow-distance ``D`` between keyframes
    (lower D => more co-visible). ``s = clamp((thresh-D)/thresh, 0, 1)`` is mapped
    to a pseudo-count ``covis = 1 + s*(covis_max-1)`` and ``info = info_scale*covis``.
    Strongly co-observed pairs therefore form stiff edges, so a downstream
    localizer's absolute priors correct global drift without distorting local shape.

    ``n`` selects an incremental prefix (vertices 0..n-1) for real-time growing
    publication; ``None`` uses all current keyframes.
    """
    t = hi2.video.counter.value if n is None else int(n)

    # world<-sensor poses: video.poses are T_cw, so invert to get T_wc (= T_WS)
    vertex_pose = lietorch.SE3(hi2.video.poses[:t].clone()).inv().data.detach().cpu().double().numpy()
    vertex_id = np.arange(t, dtype=np.int64)

    # per-keyframe timestamp parsed from the image filename (same as save_trajectory)
    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1])
                             for x in sorted(os.listdir(imagedir))[start:]])
    kf_frame_idx = hi2.video.tstamp[:t].cpu().numpy().astype(int)
    vertex_stamp_ns = tstamps_full[kf_frame_idx].astype(np.int64)

    # covisibility flow-distance matrix between keyframes (lower = more covisible)
    D = hi2.video.distance().detach().cpu().numpy()[:t, :t]
    if covis_thresh is None:
        covis_thresh = float(hi2.config['Tracking']['backend']['backend_thresh'])

    def covis_info(i, j):
        # flow-distance -> covisibility strength s in [0,1] -> pseudo landmark count
        s = max(0.0, min(1.0, (covis_thresh - float(D[i, j])) / max(covis_thresh, 1e-6)))
        covis = 1.0 + s * (covis_max - 1.0)
        return info_scale * covis

    edge_i, edge_j, edge_type, edge_info = [], [], [], []
    for i in range(t - 1):                       # sequential VO edges
        edge_i.append(i); edge_j.append(i + 1); edge_type.append(0)
        edge_info.append(covis_info(i, i + 1))
    for i in range(t):                           # non-adjacent covisibility / loop edges
        for j in range(i + 2, t):
            if D[i, j] < covis_thresh:
                edge_i.append(i); edge_j.append(j); edge_type.append(1)
                edge_info.append(covis_info(i, j))
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edge_j = np.asarray(edge_j, dtype=np.int64)
    edge_type = np.asarray(edge_type, dtype=np.uint8)

    if len(edge_i):
        # relative measurement T_AB = T_WA^-1 * T_WB computed from the world poses
        Ti = lietorch.SE3(torch.as_tensor(vertex_pose[edge_i], device='cuda', dtype=torch.float))
        Tj = lietorch.SE3(torch.as_tensor(vertex_pose[edge_j], device='cuda', dtype=torch.float))
        edge_rel = (Ti.inv() * Tj).data.detach().cpu().double().numpy()
    else:
        edge_rel = np.zeros((0, 7), dtype=np.float64)

    # covisibility-weighted diagonal information (info = info_scale * max(1, covis))
    edge_info_trans = np.asarray(edge_info, dtype=np.float64)
    edge_info_rot = edge_info_trans.copy()

    return dict(vertex_id=vertex_id, vertex_stamp_ns=vertex_stamp_ns, vertex_pose=vertex_pose,
                edge_i=edge_i, edge_j=edge_j, edge_rel=edge_rel,
                edge_info_trans=edge_info_trans, edge_info_rot=edge_info_rot, edge_type=edge_type)


def save_pose_graph(pg, output):
    """ save the pose graph dict (build_pose_graph) as ``{output}/pose_graph.npz``. """
    np.savez(f"{output}/pose_graph.npz",
             convention=np.array(POSE_GRAPH_CONVENTION, dtype=object), **pg)
    et = pg['edge_type']
    print(f"Saved pose graph: {len(pg['vertex_id'])} vertices, {len(pg['edge_i'])} edges "
          f"({int((et == 0).sum())} VO / {int((et == 1).sum())} loop) -> {output}/pose_graph.npz")


def save_rosbag(hi2, imagedir, output, pg, start=0, rate_hz=10.0, jpeg_quality=92,
                image_topic='/okvis/okvis_cam0/image/compressed',
                odometry_topic='/okvis/okvis_odometry',
                trajectory_topic='/okvis/okvis_trajectory',
                final_trajectory_topic='/okvis/okvis_trajectory_final',
                pose_graph_topic='/okvis/okvis_pose_graph',
                world_frame='map', camera_frame='cam0'):
    """ write a ROS2 bag that is a drop-in replacement for the OKVIS bag the
    ScaRF-SLAM mapping node consumes, using the pure-python ``rosbags`` library
    (no ROS install needed). Topic names/types match ScaRF-SLAM's
    okvis_slam_hilti.yaml so HI-SLAM2 can stand in for OKVIS:

      {image_topic}             sensor_msgs/CompressedImage   keyframe rgb (jpeg)
      {odometry_topic}          nav_msgs/Odometry             T_wc per keyframe (KF selection)
      {trajectory_topic}        nav_msgs/Path                 camera-to-world snapshot (incremental)
      {final_trajectory_topic}  nav_msgs/Path                 final loop-closed trajectory
      {pose_graph_topic}        okvis_pose_graph_msgs/PoseGraph  for z-floc PGO (growing snapshot per keyframe)

    ScaRF-SLAM admits a frame once it has a compressed image, an odometry pose,
    and is covered by a trajectory snapshot (slam_topic_source.py). Poses are
    T_wc (camera-to-world). Header stamps carry each keyframe's timestamp (parsed
    from the image filename) so they match pose_graph vertex_stamp_ns / priors;
    the bag record clock is a synthetic {rate_hz} Hz so the bag stays replayable
    even when filenames are plain frame indices (e.g. Gibson).
    """
    import shutil
    import cv2
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    ts = get_typestore(Stores.LATEST)
    ts.register(get_types_from_msg(POSE_GRAPH_MSGDEF, 'okvis_pose_graph_msgs/msg/PoseGraph'))
    Time = ts.types['builtin_interfaces/msg/Time']
    Header = ts.types['std_msgs/msg/Header']
    CompressedImage = ts.types['sensor_msgs/msg/CompressedImage']
    Odometry = ts.types['nav_msgs/msg/Odometry']
    Path = ts.types['nav_msgs/msg/Path']
    PoseStamped = ts.types['geometry_msgs/msg/PoseStamped']
    PoseWithCovariance = ts.types['geometry_msgs/msg/PoseWithCovariance']
    TwistWithCovariance = ts.types['geometry_msgs/msg/TwistWithCovariance']
    Twist = ts.types['geometry_msgs/msg/Twist']
    Vector3 = ts.types['geometry_msgs/msg/Vector3']
    Pose = ts.types['geometry_msgs/msg/Pose']
    Point = ts.types['geometry_msgs/msg/Point']
    Quaternion = ts.types['geometry_msgs/msg/Quaternion']
    PoseGraph = ts.types['okvis_pose_graph_msgs/msg/PoseGraph']

    def mk_time(ns):
        ns = int(ns)
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def mk_pose(p):
        return Pose(position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                    orientation=Quaternion(x=float(p[3]), y=float(p[4]), z=float(p[5]), w=float(p[6])))

    def mk_posestamped(p, stamp, frame):
        return PoseStamped(header=Header(stamp=mk_time(stamp), frame_id=frame), pose=mk_pose(p))

    zero36 = np.zeros(36, dtype=np.float64)
    zero_twist = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0))

    bagpath = os.path.join(output, 'rosbag')
    if os.path.exists(bagpath):
        shutil.rmtree(bagpath)

    t = len(pg['vertex_id'])
    period = int(1e9 / rate_hz)
    base = int(1e9)
    stamps = pg['vertex_stamp_ns'].astype(np.int64)
    poses = pg['vertex_pose']
    kf_frame_idx = hi2.video.tstamp[:t].cpu().numpy().astype(int)

    # full pose-graph edge arrays (edges stored with edge_i < edge_j); the real-time
    # growing snapshot at keyframe i = vertices 0..i + edges fully inside that prefix.
    pg_ei = pg['edge_i'].astype(np.int64)
    pg_ej = pg['edge_j'].astype(np.int64)

    def mk_posegraph(stamp, k):
        """ growing pose-graph snapshot containing keyframes 0..k-1 (current poses).
        Mirrors the OKVIS2-X per-keyframe incremental publication so the downstream
        z-floc PGO sees the graph grow over the bag timeline rather than one final dump. """
        em = pg_ej < k                     # both ends inside prefix (edge_i < edge_j < k)
        return PoseGraph(
            header=Header(stamp=mk_time(stamp), frame_id=world_frame),
            vertex_id=pg['vertex_id'][:k].astype(np.uint64),
            vertex_stamp_ns=stamps[:k],
            vertex_pose=[mk_pose(p) for p in poses[:k]],
            edge_i=pg_ei[em].astype(np.uint64),
            edge_j=pg_ej[em].astype(np.uint64),
            edge_rel=[mk_pose(p) for p in pg['edge_rel'][em]],
            edge_info_trans=pg['edge_info_trans'][em].astype(np.float64),
            edge_info_rot=pg['edge_info_rot'][em].astype(np.float64),
            edge_type=pg['edge_type'][em].astype(np.uint8))

    with Writer(bagpath, version=Writer.VERSION_LATEST) as writer:
        c_img = writer.add_connection(image_topic, CompressedImage.__msgtype__, typestore=ts)
        c_odom = writer.add_connection(odometry_topic, Odometry.__msgtype__, typestore=ts)
        c_traj = writer.add_connection(trajectory_topic, Path.__msgtype__, typestore=ts)
        c_final = writer.add_connection(final_trajectory_topic, Path.__msgtype__, typestore=ts)
        c_pg = writer.add_connection(pose_graph_topic, PoseGraph.__msgtype__, typestore=ts)

        traj_poses = []   # grows so /okvis/okvis_trajectory is an incremental snapshot
        for i in range(t):
            stamp = int(stamps[i])
            bag_t = base + i * period

            # compressed (jpeg) keyframe image; ScaRF decodes the bytes to rgb
            rgb = hi2.images[int(kf_frame_idx[i])][0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
            ok, buf = cv2.imencode('.jpg', rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            img = CompressedImage(header=Header(stamp=mk_time(stamp), frame_id=camera_frame),
                                  format='bgr8; jpeg compressed bgr8',
                                  data=np.asarray(buf, dtype=np.uint8).reshape(-1))
            writer.write(c_img, bag_t, ts.serialize_cdr(img, CompressedImage.__msgtype__))

            # odometry: camera-to-world pose, used for keyframe selection
            odom = Odometry(header=Header(stamp=mk_time(stamp), frame_id=world_frame),
                            child_frame_id=camera_frame,
                            pose=PoseWithCovariance(pose=mk_pose(poses[i]), covariance=zero36.copy()),
                            twist=TwistWithCovariance(twist=zero_twist, covariance=zero36.copy()))
            writer.write(c_odom, bag_t, ts.serialize_cdr(odom, Odometry.__msgtype__))

            # incremental camera-to-world trajectory snapshot
            traj_poses.append(mk_posestamped(poses[i], stamp, world_frame))
            traj = Path(header=Header(stamp=mk_time(stamp), frame_id=world_frame), poses=list(traj_poses))
            writer.write(c_traj, bag_t, ts.serialize_cdr(traj, Path.__msgtype__))

            # growing pose graph (vertices 0..i + enclosed edges) -> real-time PGO
            writer.write(c_pg, bag_t,
                         ts.serialize_cdr(mk_posegraph(stamp, i + 1), PoseGraph.__msgtype__))

        # final loop-closed trajectory (full) + pose graph, after the last keyframe
        last_stamp = int(stamps[-1])
        final = Path(header=Header(stamp=mk_time(last_stamp), frame_id=world_frame),
                     poses=[mk_posestamped(poses[i], int(stamps[i]), world_frame) for i in range(t)])
        writer.write(c_final, base + t * period, ts.serialize_cdr(final, Path.__msgtype__))

        pg_msg = PoseGraph(
            header=Header(stamp=mk_time(last_stamp), frame_id=world_frame),
            vertex_id=pg['vertex_id'].astype(np.uint64),
            vertex_stamp_ns=stamps,
            vertex_pose=[mk_pose(p) for p in poses],
            edge_i=pg['edge_i'].astype(np.uint64),
            edge_j=pg['edge_j'].astype(np.uint64),
            edge_rel=[mk_pose(p) for p in pg['edge_rel']],
            edge_info_trans=pg['edge_info_trans'].astype(np.float64),
            edge_info_rot=pg['edge_info_rot'].astype(np.float64),
            edge_type=pg['edge_type'].astype(np.uint8))
        writer.write(c_pg, base + (t + 1) * period, ts.serialize_cdr(pg_msg, PoseGraph.__msgtype__))

    print(f"Saved rosbag (OKVIS-compatible): {t} keyframes -> {bagpath}\n"
          f"  {image_topic} (CompressedImage), {odometry_topic} (Odometry),\n"
          f"  {trajectory_topic}/_final (Path), {pose_graph_topic} (PoseGraph)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")

    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=-1, help="number of keyframes to buffer (default: 1/10 of total frames)")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true")
    parser.add_argument("--gsvis", action="store_true")

    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")

    parser.add_argument("--no_gs_refine", action="store_true",
                        help="skip the final Global GS color refinement (much faster; lower-quality final map/renderings, "
                             "poses come straight from the global BA)")
    parser.add_argument("--pose_graph", action="store_true",
                        help="export the keyframe pose graph (pose_graph.npz) for the z-floc PGO bridge")
    parser.add_argument("--rosbag", action="store_true",
                        help="save a ROS2 bag (keyframe image + pose + pose graph) for the z-floc / scarf pipeline")
    parser.add_argument("--pg_covis_thresh", type=float, default=None,
                        help="flow-distance threshold for covisibility/loop edges in the pose graph export "
                             "(default: Tracking.backend.backend_thresh from the config)")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')

    hi2 = None
    queue = Queue(maxsize=8)
    reader = Process(target=mono_stream, args=(queue, args.imagedir, args.calib, args.undistort, args.cropborder, args.start, args.length))
    reader.start()

    N = len(os.listdir(args.imagedir))
    args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer
    pbar = tqdm(range(N), desc="Processing keyframes")
    while 1:
        (t, image, intrinsics, is_last) = queue.get()
        pbar.update()

        if hi2 is None:
            args.image_size = [image.shape[2], image.shape[3]]
            hi2 = Hi2(args)

        hi2.track(t, image, intrinsics=intrinsics, is_last=is_last)

        if args.droidvis and hi2.video.tstamp[hi2.video.counter.value-1] == t:
            from geom.ba import get_prior_depth_aligned
            index = hi2.video.counter.value-2
            depth_prior, _ = get_prior_depth_aligned(hi2.video.disps_prior_up[index][None].cuda(), hi2.video.dscales[index][None])
            show_image(image[0], 1./depth_prior.squeeze(), 1./hi2.video.disps_up[index], hi2.video.normals[index])
        pbar.set_description(f"Processing keyframe {hi2.video.counter.value} gs {hi2.gs.gaussians._xyz.shape[0]}")

        if is_last:
            pbar.close()
            break

    reader.join()

    traj = hi2.terminate()
    save_trajectory(hi2, traj, args.imagedir, args.output, start=args.start)
    if args.pose_graph or args.rosbag:
        pg = build_pose_graph(hi2, args.imagedir, start=args.start, covis_thresh=args.pg_covis_thresh)
        if args.pose_graph:
            save_pose_graph(pg, args.output)
        if args.rosbag:
            save_rosbag(hi2, args.imagedir, args.output, pg, start=args.start)

    print("Done")
