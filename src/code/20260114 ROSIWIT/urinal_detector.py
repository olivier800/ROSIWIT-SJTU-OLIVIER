#!/usr/bin/env python3
import rospy
import numpy as np
import open3d as o3d
import time
import os
import threading

from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, ColorRGBA
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

class UrinalDetector:
    def __init__(self):
        # Load parameters from ROS parameter server
        self.load_parameters()
        
        # State management
        self.lock = threading.Lock()
        self.latest_msg = None
        self.processed = False
        self.frame_id = self.default_frame_id
        
        # Cached results
        self.cached_processed_xyz = None
        self.cached_uniform_xyz = None
        self.cached_plane_path = None      # (points, normals) tuple
        self.cached_remain_path = None     # (points, normals) tuple
        self.cached_center_point = None
        
        # Logging flags (only log once)
        self._logged_processed = False
        self._logged_uniform = False
        self._logged_plane = False
        self._logged_remain = False
        
        # ROS Publishers
        self.processed_pc_pub = rospy.Publisher(
            self.processed_pointcloud_topic, PointCloud2, queue_size=1, latch=True)
        self.uniform_pc_pub = rospy.Publisher(
            self.uniform_topic, PointCloud2, queue_size=1, latch=True)
        self.plane_path_pub = rospy.Publisher(
            self.plane_path_topic, Marker, queue_size=1, latch=True)
        self.remain_path_pub = rospy.Publisher(
            self.remain_path_topic, Marker, queue_size=1, latch=True)
        self.plane_normals_pub = rospy.Publisher(
            self.plane_path_topic + "_normals", MarkerArray, queue_size=1, latch=True)
        self.remain_normals_pub = rospy.Publisher(
            self.remain_path_topic + "_normals", MarkerArray, queue_size=1, latch=True)
        self.center_point_pub = rospy.Publisher(
            self.center_point_topic, PointStamped, queue_size=1, latch=True)
        
        # ROS Subscriber
        self.sub = rospy.Subscriber(
            self.input_cloud_topic, PointCloud2, self.cb_cloud, queue_size=1)
        
        # Timers
        self.proc_timer = rospy.Timer(rospy.Duration(0.05), self.try_process_once)
        self.repub_timer = rospy.Timer(
            rospy.Duration(1.0/max(1e-6, self.pub_rate)), self.republish_cached)
        
        self.is_active = True
        rospy.loginfo("UrinalDetector initialized as standalone node")

    def load_parameters(self):
        """Load all parameters from ROS parameter server with default values"""
        # Topic names (matching clean_path_urinal_node.py)
        self.input_cloud_topic = rospy.get_param("~input_cloud_topic", "target_pointcloud")
        self.processed_pointcloud_topic = rospy.get_param("~processed_pointcloud_topic", "processed_pointcloud")
        self.uniform_topic = rospy.get_param("~uniform_topic", "uniform_pointcloud")
        self.plane_path_topic = rospy.get_param("~plane_path_topic", "clean_path_plane")
        self.remain_path_topic = rospy.get_param("~remain_path_topic", "clean_path_remain")
        self.center_point_topic = rospy.get_param("~center_point_topic", "clean_path_center_point")
        self.default_frame_id = rospy.get_param("~default_frame_id", "base_link")
        
        # Publishing rate
        self.pub_rate = rospy.get_param("~pub_rate", 2.0)
        
        # ROI parameters (for preprocessing)
        self.roi_min = rospy.get_param("~roi_min", [-1.0, -1.0, -1.0])
        self.roi_max = rospy.get_param("~roi_max", [1.0, 1.0, 1.0])
        
        # Preprocessing parameters
        self.voxel_size = rospy.get_param("~voxel_size", 0.005)
        self.ror_radius = rospy.get_param("~ror_radius", 0.012)
        self.ror_min_pts = rospy.get_param("~ror_min_pts", 8)
        self.trim_top = rospy.get_param("~trim_top", 0.02)
        self.trim_bottom = rospy.get_param("~trim_bottom", 0.00)
        
        # Tool orientation parameters
        self.predefined_rpy = rospy.get_param("~predefined_rpy", [0.0, 0.0, 0.0])
        self.tool_pointing_height = rospy.get_param("~tool_pointing_height", 0.1)
        self.tool_pointing_x_offset_ratio = rospy.get_param("~tool_pointing_x_offset_ratio", 0.12)
        
        # Urinal detection parameters
        self.points_distance = rospy.get_param("~urinal_detector/points_distance", 0.1)
        self.distance_between_rotations = rospy.get_param("~urinal_detector/distance_between_rotations", 0.1)
        self.default_opening_angle = rospy.get_param("~urinal_detector/default_opening_angle", 120.0)
        self.path_expand = rospy.get_param("~urinal_detector/path_expand", 0.0)
        
        # Alpha Shape algorithm parameters (NEW)
        self.use_alpha_shape = rospy.get_param("~urinal_detector/use_alpha_shape", False)
        self.alpha_value = rospy.get_param("~urinal_detector/alpha_value", 0.20)
        self.enable_plane_detect = rospy.get_param("~urinal_detector/enable_plane_detect", False)
        self.plane_raster_spacing = rospy.get_param("~urinal_detector/plane_raster_spacing", 0.02)
        
        # Advanced layering parameters (from sink_detector)
        self.slice_mode = rospy.get_param("~urinal_detector/slice_mode", "by_bins")  # "by_bins" or "by_distance"
        self.slice_bins = rospy.get_param("~urinal_detector/slice_bins", 10)
        self.layer_distance = rospy.get_param("~urinal_detector/layer_distance", 0.05)  # Layer height in meters
        
        # Boundary expansion (2D XY plane outward offset)
        self.boundary_expansion = rospy.get_param("~urinal_detector/boundary_expansion", 0.0)
        
        # Layer point extension (for gap filling)
        self.enable_layer_point_extension = rospy.get_param("~urinal_detector/enable_layer_point_extension", False)
        self.layer_point_extension_distance = rospy.get_param("~urinal_detector/layer_point_extension_distance", 0.03)
        
        # Visualization parameters
        self.path_line_width = rospy.get_param("~path_line_width", 0.003)
        self.normal_arrow_len = rospy.get_param("~normal_arrow_len", 0.05)
        
        rospy.loginfo("UrinalDetector parameters loaded successfully")
    
    # ========== ROS Utility Functions ==========
    
    def ros_pc2_to_xyz_array(self, msg, remove_nans=True):
        """Convert ROS PointCloud2 to numpy array"""
        pts = []
        for p in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=remove_nans):
            pts.append([p[0], p[1], p[2]])
        if not pts:
            return np.zeros((0,3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)
    
    def xyz_array_to_pc2(self, xyz, frame_id, stamp=None):
        """Convert numpy array to ROS PointCloud2"""
        header = Header()
        header.stamp = stamp or rospy.Time.now()
        header.frame_id = frame_id
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, xyz.astype(np.float32))
    
    def path_xyz_to_marker(self, path_xyz, frame_id, rgba=(0.9,0.2,0.2,1.0), width=0.003):
        """Convert path to LINE_STRIP Marker"""
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = "clean_path"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = float(width)
        m.color = ColorRGBA(*rgba)
        m.pose.orientation.w = 1.0
        m.lifetime = rospy.Duration(0)
        m.points = [Point(x=float(x), y=float(y), z=float(z)) for x,y,z in path_xyz]
        return m
    
    def rpy_to_normals(self, rpy):
        """Convert RPY angles to normal vectors (approximate)"""
        normals = np.zeros_like(rpy)
        for i, (roll, pitch, yaw) in enumerate(rpy):
            # Simple approximation: normal points in direction of pitch/yaw
            normals[i] = [
                np.cos(yaw) * np.cos(pitch),
                np.sin(yaw) * np.cos(pitch),
                np.sin(pitch)
            ]
        return normals
    
    def quat_align_x_to_vec(self, vec, up_hint=np.array([0,0,1.0])):
        """Generate quaternion that aligns X-axis to vec"""
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        z_axis = vec
        y_axis = np.cross(up_hint, z_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-6:
            y_axis = np.array([0, 1, 0])
        else:
            y_axis = y_axis / y_norm
        x_axis = np.cross(y_axis, z_axis)
        
        # Rotation matrix to quaternion
        R = np.column_stack([x_axis, y_axis, z_axis])
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2,1] - R[1,2]) * s
            qy = (R[0,2] - R[2,0]) * s
            qz = (R[1,0] - R[0,1]) * s
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
                qw = (R[2,1] - R[1,2]) / s
                qx = 0.25 * s
                qy = (R[0,1] + R[1,0]) / s
                qz = (R[0,2] + R[2,0]) / s
            elif R[1,1] > R[2,2]:
                s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
                qw = (R[0,2] - R[2,0]) / s
                qx = (R[0,1] + R[1,0]) / s
                qy = 0.25 * s
                qz = (R[1,2] + R[2,1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
                qw = (R[1,0] - R[0,1]) / s
                qx = (R[0,2] + R[2,0]) / s
                qy = (R[1,2] + R[2,1]) / s
                qz = 0.25 * s
        return np.array([qx, qy, qz, qw], dtype=np.float64)
    
    # ========== Preprocessing Functions ==========
    
    def preprocess_pcd(self, pcd):
        """Preprocess point cloud"""
        rospy.loginfo("[UrinalDetector] Preprocessing: voxel=%.3f, ror_radius=%.3f/min_pts=%d",
                     self.voxel_size, self.ror_radius, self.ror_min_pts)
        
        input_points = len(pcd.points)
        
        # Voxel downsampling
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Radius outlier removal
        pcd, _ = pcd.remove_radius_outlier(nb_points=self.ror_min_pts, radius=self.ror_radius)
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.03, max_nn=50))
        try:
            pcd.orient_normals_consistent_tangent_plane(k=30)
        except:
            pcd.orient_normals_towards_camera_location(camera_location=(0.0, 0.0, 0.0))
        
        # Trim by height
        pcd = self.trim_by_height(pcd, self.trim_bottom, self.trim_top)
        
        rospy.loginfo("[UrinalDetector] Preprocessing complete: %d -> %d points",
                     input_points, len(pcd.points))
        return pcd
    
    def trim_by_height(self, pcd, trim_bottom=0.0, trim_top=0.0):
        """Trim point cloud by height"""
        if trim_bottom <= 0 and trim_top <= 0:
            return pcd
        
        bbox = pcd.get_axis_aligned_bounding_box()
        minb = bbox.get_min_bound()
        maxb = bbox.get_max_bound()
        zmin, zmax = float(minb[2]), float(maxb[2])
        
        new_zmin = zmin + max(0.0, trim_bottom)
        new_zmax = zmax - max(0.0, trim_top)
        
        if new_zmax <= new_zmin:
            return pcd
        
        new_min = np.array([minb[0], minb[1], new_zmin])
        new_max = np.array([maxb[0], maxb[1], new_zmax])
        aabb = o3d.geometry.AxisAlignedBoundingBox(new_min, new_max)
        
        return pcd.crop(aabb)
    
    # ========== Callback and Processing ==========
    
    def cb_cloud(self, msg):
        """Callback for point cloud input"""
        with self.lock:
            if not self.processed:
                self.latest_msg = msg
                self.frame_id = msg.header.frame_id or self.frame_id
    
    def try_process_once(self, _evt):
        """Try to process point cloud once"""
        if self.processed:
            return
        
        with self.lock:
            msg = self.latest_msg
            self.latest_msg = None
        
        if msg is None:
            return
        
        try:
            t0 = time.time()
            xyz = self.ros_pc2_to_xyz_array(msg, remove_nans=True)
            
            if xyz.shape[0] == 0:
                rospy.logwarn("[UrinalDetector] Empty point cloud, skipping")
                return
            
            rospy.loginfo("[UrinalDetector] Processing point cloud: %d points", xyz.shape[0])
            
            # Convert to Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # 1) Preprocess
            pcd_clean = self.preprocess_pcd(pcd)
            self.cached_processed_xyz = np.asarray(pcd_clean.points)
            
            # 2) Uniformize (simple downsampling for now)
            pcd_uniform = pcd_clean.voxel_down_sample(voxel_size=self.voxel_size * 2)
            self.cached_uniform_xyz = np.asarray(pcd_uniform.points)
            
            rospy.loginfo("[UrinalDetector] Preprocessing complete: %d -> %d -> %d points",
                         xyz.shape[0], len(self.cached_processed_xyz), len(self.cached_uniform_xyz))
            
            # 3) Calculate center point
            if len(self.cached_uniform_xyz) > 0:
                self.cached_center_point = np.mean(self.cached_uniform_xyz, axis=0)
            
            # 4) Generate cleaning path
            clean_path = self.generate_clean_path(self.cached_uniform_xyz)
            
            if clean_path is not None and len(clean_path) > 0:
                # Split path into position (Nx3) and orientation (Nx3 as RPY)
                if clean_path.shape[1] == 6:
                    path_xyz = clean_path[:, :3]
                    path_rpy = clean_path[:, 3:6]
                    # Convert RPY to normals (approximate)
                    path_normals = self.rpy_to_normals(path_rpy)
                else:
                    path_xyz = clean_path
                    path_normals = np.tile([0, 0, 1], (len(path_xyz), 1))
                
                # For now, treat entire path as "remain" (side wall)
                # Future: implement plane detection to split
                self.cached_plane_path = (np.empty((0, 3)), np.empty((0, 3)))
                self.cached_remain_path = (path_xyz, path_normals)
                
                rospy.loginfo("[UrinalDetector] Path generated: %d points", len(path_xyz))
            else:
                rospy.logwarn("[UrinalDetector] Failed to generate path")
                self.cached_plane_path = (np.empty((0, 3)), np.empty((0, 3)))
                self.cached_remain_path = (np.empty((0, 3)), np.empty((0, 3)))
            
            self.processed = True
            self.publish_all()
            
            rospy.loginfo("[UrinalDetector] Processing complete: elapsed=%.3fs", time.time() - t0)
            self.proc_timer.shutdown()
            
        except Exception as e:
            import traceback
            rospy.logerr("[UrinalDetector] Processing error: %s", str(e))
            rospy.logerr("[UrinalDetector] Traceback:\n%s", traceback.format_exc())
            self.processed = True
            self.proc_timer.shutdown()
    
    def publish_all(self):
        """Publish all cached results"""
        now = rospy.Time.now()
        
        # 1. Publish processed point cloud
        if self.cached_processed_xyz is not None:
            self.processed_pc_pub.publish(
                self.xyz_array_to_pc2(self.cached_processed_xyz, frame_id=self.frame_id, stamp=now))
            if not self._logged_processed:
                rospy.loginfo("[UrinalDetector] Publishing: /%s (%d points)",
                             self.processed_pointcloud_topic, len(self.cached_processed_xyz))
                self._logged_processed = True
        
        # 2. Publish uniform point cloud
        if self.cached_uniform_xyz is not None:
            self.uniform_pc_pub.publish(
                self.xyz_array_to_pc2(self.cached_uniform_xyz, frame_id=self.frame_id, stamp=now))
            if not self._logged_uniform:
                rospy.loginfo("[UrinalDetector] Publishing: /%s (%d points)",
                             self.uniform_topic, len(self.cached_uniform_xyz))
                self._logged_uniform = True
        
        # 3. Publish plane path
        if self.cached_plane_path is not None:
            plane_pts, plane_nrm = self.cached_plane_path
            if len(plane_pts) > 0:
                mk_plane = self.path_xyz_to_marker(
                    plane_pts, frame_id=self.frame_id,
                    rgba=(0.9, 0.2, 0.2, 1.0), width=self.path_line_width)
                mk_plane.header.stamp = now
                mk_plane.ns = "plane"
                self.plane_path_pub.publish(mk_plane)
                
                if not self._logged_plane:
                    rospy.loginfo("[UrinalDetector] Publishing: /%s (%d points)",
                                 self.plane_path_topic, len(plane_pts))
                    self._logged_plane = True
                
                # Publish normals
                if len(plane_nrm) > 0:
                    ma_plane = self.create_normal_markers(
                        plane_pts, plane_nrm, "plane_normals", now)
                    self.plane_normals_pub.publish(ma_plane)
        
        # 4. Publish remain path
        if self.cached_remain_path is not None:
            remain_pts, remain_nrm = self.cached_remain_path
            if len(remain_pts) > 0:
                mk_remain = self.path_xyz_to_marker(
                    remain_pts, frame_id=self.frame_id,
                    rgba=(0.9, 0.2, 0.2, 1.0), width=self.path_line_width)
                mk_remain.header.stamp = now
                mk_remain.ns = "remain"
                self.remain_path_pub.publish(mk_remain)
                
                if not self._logged_remain:
                    rospy.loginfo("[UrinalDetector] Publishing: /%s (%d points)",
                                 self.remain_path_topic, len(remain_pts))
                    self._logged_remain = True
                
                # Publish normals
                if len(remain_nrm) > 0:
                    ma_remain = self.create_normal_markers(
                        remain_pts, remain_nrm, "remain_normals", now)
                    self.remain_normals_pub.publish(ma_remain)
        
        # 5. Publish center point
        if self.cached_center_point is not None:
            center_msg = PointStamped()
            center_msg.header.stamp = now
            center_msg.header.frame_id = self.frame_id
            center_msg.point.x = float(self.cached_center_point[0])
            center_msg.point.y = float(self.cached_center_point[1])
            center_msg.point.z = float(self.cached_center_point[2])
            self.center_point_pub.publish(center_msg)
    
    def create_normal_markers(self, points, normals, ns, stamp):
        """Create MarkerArray for normal visualization"""
        ma = MarkerArray()
        stride = max(1, len(points) // 80)  # Max 80 arrows
        
        for i, (p, n) in enumerate(zip(points[::stride], normals[::stride])):
            arrow = Marker()
            arrow.header.frame_id = self.frame_id
            arrow.header.stamp = stamp
            arrow.ns = ns
            arrow.id = i
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            
            qn = self.quat_align_x_to_vec(n, up_hint=np.array([0,0,1.0]))
            arrow.pose.position.x = float(p[0])
            arrow.pose.position.y = float(p[1])
            arrow.pose.position.z = float(p[2])
            arrow.pose.orientation.x = float(qn[0])
            arrow.pose.orientation.y = float(qn[1])
            arrow.pose.orientation.z = float(qn[2])
            arrow.pose.orientation.w = float(qn[3])
            
            arrow.scale.x = self.normal_arrow_len
            arrow.scale.y = self.normal_arrow_len * 0.2
            arrow.scale.z = self.normal_arrow_len * 0.2
            arrow.color.r = 0.2
            arrow.color.g = 0.9
            arrow.color.b = 0.3
            arrow.color.a = 0.95
            
            ma.markers.append(arrow)
        
        return ma
    
    def republish_cached(self, _evt):
        """Republish cached results periodically"""
        if self.processed:
            self.publish_all()
    
    # ========== Path Generation (existing methods) ==========
    
    def analyze_urinal_geometry(self, points):
        """Analyze urinal geometry to extract key parameters"""
        points_clean = points

        def find_bowl_center(points_clean, grid_resolution=0.01, reasonable_radius=0.3):
            # Top 20% for rim center
            top_threshold = np.percentile(points_clean[:, 2], 80)
            top_points = points_clean[points_clean[:, 2] > top_threshold]
            rim_center = np.median(top_points, axis=0)
            
            # Overall center
            overall_center = np.median(points_clean, axis=0)
            
            # Find the direction from rim to wall (assuming wall is at max X)
            wall_x = np.max(points_clean[:, 0])
            
            # Shift center away from wall, toward the rim
            # Weight rim center more (60%) since it's cleaner
            # bowl_x = 0.6 * rim_center[0] + 0.4 * overall_center[0]
            # bowl_x = overall_center[0] - (wall_x - rim_center[0])
            bowl_x = (np.percentile(points_clean[:, 0], 99) + np.percentile(points_clean[:, 0], 1))/2
            rospy.logwarn("%f,%f",np.percentile(points_clean[:, 0], 99),np.percentile(points_clean[:, 0], 1))
            rospy.logwarn("%f,%f",np.max(points_clean[:, 0]),np.min(points_clean[:, 0]))
            
            # For Y, use rim center directly (less wall contamination in Y)
            # bowl_y = (rim_center[1] + overall_center[1]) / 2
            bowl_y = np.average(points_clean[:, 1])
            print(f"rim_center={rim_center}, overall_center={overall_center}, wall_x={wall_x}, bowl_x={bowl_x}, bowl_y={bowl_y}")

            bottom_threshold = np.percentile(points_clean[:, 2], 20)
            bottom_points = points_clean[points_clean[:, 2] < bottom_threshold]
            bowl_z = np.median(bottom_points[:, 2])
            
            # For Z, use overall median
            # bowl_z = overall_center[2]
            
            return np.array([bowl_x, bowl_y, bowl_z])
                
        
        # Find cylinder center (median of all points)
        center = find_bowl_center(points_clean)
        rospy.loginfo(f"UrinalDetector: Analyzed geometry - center={center}")
        
        # Calculate basic dimensions
        min_z = np.min(points_clean[:, 2])
        max_z = np.max(points_clean[:, 2]) - 0.2
        total_height = max_z - min_z
        
        # Calculate diameters at different heights
        def diameter_at_height(height, tolerance=0.02):
            height_mask = np.abs(points_clean[:, 2] - height) < tolerance
            slice_points = points_clean[height_mask]
            if len(slice_points) < 10:
                return 0.0
            xy_distances = np.linalg.norm(slice_points[:, :2] - center[:2], axis=1)
            
            # Use 25th percentile to ignore rim points (focus on bowl)
            return 2 * np.percentile(xy_distances, 25)
        
        bottom_diameter = np.max([0.1,diameter_at_height(min_z + 0.1)])
        top_diameter = np.min([0.3, diameter_at_height((max_z + min_z) / 2)])

        height_increment = 0.2
        height = np.arange(min_z, max_z + height_increment, height_increment)
        diameters = np.array([diameter_at_height(z) for z in height])
        diameters = np.max(diameters)

        # top_diameter = diameters
        # bottom_diameter = diameters

        # Detect opening angle (simplified)        
        def detect_opening_angle(height):
            height_threshold = 0.05
            height_min = height - height_threshold
            height_max = height + height_threshold
            points = points_clean[(points_clean[:, 2] >= height_min) & (points_clean[:, 2] <= height_max)]
            
            vectors = points[:, :2] - center[:2]
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            angles = np.mod(angles, 2 * np.pi)
            
            # Better angle range calculation for circular data
            sorted_angles = np.sort(angles)
            gaps = np.diff(sorted_angles)
            # Add the wrap-around gap between last and first
            wrap_gap = (2 * np.pi) - sorted_angles[-1] + sorted_angles[0]
            gaps = np.append(gaps, wrap_gap)
            
            # The largest gap indicates the missing segment
            largest_gap = np.max(gaps)
            
            # Opening angle = 360Â° minus the largest gap
            opening_angle = 360.0 - np.degrees(largest_gap)

            # print(np.degrees(sorted_angles))

            print(f"Opening angle: {opening_angle:.1f}Â°")
            return min(360.0, opening_angle)
        
        # opening_angle = detect_opening_angle()
        
        # Find transition between full and partial cylinder
        def find_transition_height():
            test_heights = np.linspace(min_z, max_z, 10)
            opening_angles = [self.default_opening_angle]# Default opening angle in case no transition is found
            for height in test_heights[::-1]:
                rospy.loginfo(f"UrinalDetector: Analyzed geometry - height={height:.3f}")
                opening_angle = detect_opening_angle(height)
                if opening_angle > 0.9* 360:
                    return [height, np.average(opening_angles)]
                opening_angles.append(opening_angle)

            # return [min_z + 0.3*(max_z - min_z), np.average(opening_angles)]  # Default
            return [min_z + 0.15, np.min(opening_angles)]  # Default

        
        [full_spiral_height,opening_angle] = find_transition_height() 

        rospy.loginfo(f"UrinalDetector: Analyzed geometry - top_diameter={top_diameter:.3f}, bottom_diameter={bottom_diameter:.3f}, "
                      f"total_height={total_height:.3f}, full_spiral_height={full_spiral_height:.3f}, opening_angle={opening_angle:.2f}")
        
        return {
            'd_top': float(top_diameter),
            'd_bottom': float(bottom_diameter),
            'total_height': float(total_height),
            'full_spiral_height': float(full_spiral_height),
            'opening_angle': float(np.average(opening_angle)),
            'center': center
        }

    def generate_spiral_path(self, geometry_params):
        """Generate continuous spiral cleaning path in local cylinder coordinates"""
        d_top = geometry_params['d_top']
        d_bottom = geometry_params['d_bottom']
        total_height = geometry_params['total_height']
        full_spiral_height = geometry_params['full_spiral_height']
        opening_angle = geometry_params['opening_angle']
        center = geometry_params['center']
        
        r_top = d_top / 2
        r_bottom = d_bottom / 2
        opening_angle_rad = opening_angle * np.pi / 180
        
        all_points = [[center[0], center[1], full_spiral_height]]
        all_points.append([center[0], center[1], center[2]])
        
        def get_radius_at_height(height):
            if height <= full_spiral_height:
                fraction = height / full_spiral_height
                return r_bottom + fraction * (r_top - r_bottom)
            else:
                return r_top
        
        # Generate full spiral region (bottom)
        if full_spiral_height - center[2] > 0:
            full_rotations = (full_spiral_height - center[2]) / self.distance_between_rotations
            full_circumference = 2 * np.pi * (r_bottom + r_top) / 2
            full_spiral_length = np.sqrt((full_circumference * full_rotations)**2 + full_spiral_height**2)
            full_points_count = int(full_spiral_length / self.points_distance)
            
            for i in range(full_points_count):
                z = (i / (full_points_count - 1)) * (full_spiral_height - center[2])
                radius = get_radius_at_height(z) + self.path_expand
                angle = 2 * np.pi * (z / self.distance_between_rotations)
                x = radius * np.cos(angle) + center[0]
                y = radius * np.sin(angle) + center[1]
                z += center[2]
                # Return points in local cylinder coordinates
                all_points.append([x, y, z])
        
        # Generate partial spiral region (top)
        if total_height > full_spiral_height - center[2]:
            partial_height = total_height - (full_spiral_height - center[2])
            start_angle = -opening_angle_rad / 2
            end_angle = opening_angle_rad / 2
            angle_range = end_angle - start_angle
            
            # Calculate number of back-and-forth passes
            target_passes = max(2, int((partial_height / self.distance_between_rotations) * (2 * np.pi / angle_range)))
            arc_length_per_pass = r_top * angle_range
            vertical_per_pass = partial_height / target_passes
            passes_spiral_length = np.sqrt(arc_length_per_pass**2 + vertical_per_pass**2)
            total_partial_length = passes_spiral_length * target_passes

            partial_points_count = int(total_partial_length / self.points_distance)
            
            for i in range(partial_points_count):
                z = full_spiral_height + (i / (partial_points_count - 1)) * partial_height
                radius = get_radius_at_height(z) + self.path_expand
                
                pass_num = (i / partial_points_count) * target_passes
                pass_progress = pass_num % 1.0
                
                if int(pass_num) % 2 == 0:
                    angle = start_angle + pass_progress * angle_range
                else:
                    angle = end_angle - pass_progress * angle_range
                
                x = radius * np.cos(angle) + center[0]
                y = radius * np.sin(angle) + center[1]
                

                # Return points in local cylinder coordinates
                all_points.append([x, y, z])
        
        return np.array(all_points)

    def add_direction(self, cleaning_path_3d, tool_pointing_height):
        """Add direction to cleaning path
        
        Args:
            cleaning_path_3d: Nx3 numpy array of points [x, y, z]
            tool_pointing_height: float, height offset for tool pointing target
        
        Returns:
            Nx6 numpy array [x, y, z, roll, pitch, yaw]
        """
        if cleaning_path_3d is None or len(cleaning_path_3d) == 0:
            return None
        
        # Calculate the center point (mean of all points) and apply height offset
        center_point = np.mean(cleaning_path_3d, axis=0)
        center_x = center_point[0]
        center_y = center_point[1]

        # Create target points with fixed X, Y but Z following cleaning path height
        target_points = np.column_stack([
            np.full(len(cleaning_path_3d), center_x + tool_pointing_height),  # Same X for all
            np.full(len(cleaning_path_3d), center_y),  # Same Y for all  
            cleaning_path_3d[:, 2] + tool_pointing_height  # Z from path + offset
        ])

        # Current points as column vectors (3, N)
        current_points = cleaning_path_3d.T

        # Target points as column vectors (3, N)  
        target_points = target_points.T  # Shape (3, N)

        # Z-axis: direction from each current point to its corresponding target point
        z_axis = target_points - current_points
        z_norms = np.linalg.norm(z_axis, axis=0)
        z_axis = z_axis / z_norms
        
        # Preferred Y direction in XY plane (from predefined yaw)
        xy_dir = np.array([
            np.cos(self.predefined_rpy[2]), 
            np.sin(self.predefined_rpy[2]), 
            0.0
        ])
        
        # Alternative Y direction for degenerate cases
        xy_dir_alt = np.array([0.0, 1.0, 0.0])
        
        # Project xy_dir onto plane perpendicular to z_axis
        dot_y = np.sum(xy_dir[:, np.newaxis] * z_axis, axis=0)
        y_axis = xy_dir[:, np.newaxis] - dot_y * z_axis
        
        # Normalize y_axis
        norms_y = np.linalg.norm(y_axis, axis=0)
        
        # Handle degenerate cases where y_axis is too small
        is_degenerate = norms_y < 1e-6
        if np.any(is_degenerate):
            dot_alt = np.sum(xy_dir_alt[:, np.newaxis] * z_axis, axis=0)
            y_axis_alt = xy_dir_alt[:, np.newaxis] - dot_alt * z_axis
            y_axis[:, is_degenerate] = y_axis_alt[:, is_degenerate]
            norms_y[is_degenerate] = np.linalg.norm(y_axis[:, is_degenerate], axis=0)
        
        # Final normalization
        y_axis = y_axis / norms_y
        
        # X-axis = Y Ã Z (cross product)
        x_axis = np.cross(y_axis, z_axis, axis=0)
        x_axis = x_axis / np.linalg.norm(x_axis, axis=0)
        
        # Extract RPY angles from rotation matrix
        yaw = np.arctan2(x_axis[1, :], x_axis[0, :])      # Rotation around Z
        pitch = np.arcsin(x_axis[2, :])                    # Rotation around Y
        roll = np.arctan2(-y_axis[2, :], z_axis[2, :])     # Rotation around X
        
        # Combine position and orientation
        path_with_direction = np.column_stack([
            cleaning_path_3d,  # x, y, z
            roll,
            pitch,
            yaw
        ])
        
        return path_with_direction

    # ========== Alpha Shape Implementation (from sink_detector.py) ==========
    
    def _generate_path_alpha_shape(self, pts3d):
        """Generate path using Alpha Shape algorithm (from sink_detector)"""
        import open3d as o3d
        from scipy.spatial import Delaunay, ConvexHull
        from sklearn.cluster import DBSCAN
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d)
        
        # 1. Plane detection
        plane_points, remain_points, plane_model = self._detect_plane_simple(pcd)
        
        # 2. Generate plane path
        plane_path = []
        if self.enable_plane_detect and len(plane_points) > 50:
            plane_path = self._generate_raster_path(plane_points, self.plane_raster_spacing, 0.01)
            rospy.loginfo(f"[UrinalDetector] Plane路径: {len(plane_path)} points")
        
        # 3. Wall layered path
        wall_path = []
        if len(remain_points) > 100:
            wall_path = self._generate_layered_path(remain_points)
            rospy.loginfo(f"[UrinalDetector] 侧壁路径: {len(wall_path)} points")
        
        # 4. Merge paths
        if len(plane_path) > 0 and len(wall_path) > 0:
            final_path = np.vstack([plane_path, wall_path])
        elif len(wall_path) > 0:
            final_path = wall_path
        elif len(plane_path) > 0:
            final_path = plane_path
        else:
            rospy.logwarn("[UrinalDetector] No path generated")
            return None
        
        # 5. Set start/end points following legacy method convention
        if len(final_path) > 2:
            # Find top and bottom boundary centers
            z_vals = final_path[:, 2]
            z_top = z_vals.max()
            z_bottom = z_vals.min()
            
            # Top center: geometric center of highest layer points
            top_mask = z_vals > (z_top - 0.02)  # Points within 2cm of top
            if np.sum(top_mask) > 0:
                top_center = final_path[top_mask, :2].mean(axis=0)
            else:
                top_center = final_path[:, :2].mean(axis=0)
            
            # Bottom center: geometric center of lowest layer points
            bottom_mask = z_vals < (z_bottom + 0.02)  # Points within 2cm of bottom
            if np.sum(bottom_mask) > 0:
                bottom_center = final_path[bottom_mask, :2].mean(axis=0)
            else:
                bottom_center = final_path[:, :2].mean(axis=0)
            
            # Set start point: bottom center at top height + 5cm offset (entry from above)
            final_path[0, 0] = bottom_center[0]
            final_path[0, 1] = bottom_center[1]
            final_path[0, 2] = z_top + 0.05
            
            # Set end point: top center at top height (exit at top)
            final_path[-1, 0] = top_center[0]
            final_path[-1, 1] = top_center[1]
            final_path[-1, 2] = z_top
            
            rospy.loginfo(f"[UrinalDetector] Start/end points set: start=({bottom_center[0]:.3f}, {bottom_center[1]:.3f}, {z_top+0.05:.3f}), end=({top_center[0]:.3f}, {top_center[1]:.3f}, {z_top:.3f})")
        
        # 6. Add orientation (RPY) to each point
        final_path_with_rpy = self._add_orientation_to_path(final_path)
        
        rospy.loginfo(f"[UrinalDetector] Final path: {len(final_path_with_rpy)} points")
        return final_path_with_rpy
    
    def _detect_plane_simple(self, pcd):
        """简化的Plane detection (from sink_detector)"""
        pts = np.asarray(pcd.points)
        
        if not self.enable_plane_detect or len(pts) < 300:
            return np.empty((0, 3)), pts, None
        
        # RANSAC Plane detection
        try:
            model, inliers = pcd.segment_plane(
                distance_threshold=0.005,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < 200:
                return np.empty((0, 3)), pts, None
            
            # Check if horizontal
            a, b, c, d = model
            angle = np.degrees(np.arccos(abs(c) / np.sqrt(a**2 + b**2 + c**2)))
            
            if angle > 15:  # Not a horizontal plane
                return np.empty((0, 3)), pts, None
            
            plane_pts = pts[inliers]
            remain_mask = np.ones(len(pts), dtype=bool)
            remain_mask[inliers] = False
            remain_pts = pts[remain_mask]
            
            rospy.loginfo(f"[UrinalDetector] Plane: {len(plane_pts)} points, remaining: {len(remain_pts)} points")
            return plane_pts, remain_pts, model
            
        except Exception as e:
            rospy.logwarn(f"Plane detection失败: {e}")
            return np.empty((0, 3)), pts, None
    
    def _generate_raster_path(self, plane_points, spacing, step):
        """Plane往复扫描路径 (from sink_detector)"""
        if len(plane_points) < 10:
            return np.empty((0, 3))
        
        # PCA to find principal directions
        c = plane_points.mean(axis=0)
        X = plane_points - c
        C = (X.T @ X) / max(1, len(plane_points) - 1)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        
        v1, v2 = evecs[:, 0], evecs[:, 1]
        
        # Project to 2D
        t1 = X @ v1
        t2 = X @ v2
        
        t1_min, t1_max = t1.min(), t1.max()
        t2_min, t2_max = t2.min(), t2.max()
        
        # Scan lines
        n_lines = max(1, int(np.ceil((t2_max - t2_min) / spacing)))
        path_segments = []
        
        for i in range(n_lines):
            t2_val = t2_min + i * spacing
            n_samples = max(2, int(np.ceil((t1_max - t1_min) / step)))
            t1_samples = np.linspace(t1_min, t1_max, n_samples)
            
            if i % 2 == 1:  # Zigzag pattern
                t1_samples = t1_samples[::-1]
            
            segment = [c + t1_val * v1 + t2_val * v2 for t1_val in t1_samples]
            path_segments.append(np.array(segment))
        
        return np.vstack(path_segments) if path_segments else np.empty((0, 3))
    
    def _generate_layered_path(self, remain_points):
        """
        Generate layered Alpha Shape path with advanced features (from sink_detector):
        - Requirement 1: Support both by_bins and by_distance layering modes
        - All layers maintain same rotation direction (no reversal)
        """
        if len(remain_points) < 100:
            return np.empty((0, 3))
        
        # Z-axis layering setup
        z_vals = remain_points[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        total_height = z_max - z_min
        
        # Requirement 1: Choose layering mode
        layer_ranges = []
        
        if self.slice_mode == "by_distance":
            # By distance mode: fixed layer height
            layer_height = self.layer_distance
            n_layers = int(np.ceil(total_height / layer_height))
            rospy.loginfo(f"[UrinalDetector] By-distance mode: {n_layers} layers (height={layer_height:.3f}m)")
            
            for i in range(n_layers):
                z_low = z_min + i * layer_height
                z_high = min(z_low + layer_height, z_max)
                layer_ranges.append((z_low, z_high))
        else:
            # By bins mode: divide total height into fixed number of layers
            rospy.loginfo(f"[UrinalDetector] By-bins mode: {self.slice_bins} layers")
            for i in range(self.slice_bins):
                z_low = z_min + i * total_height / self.slice_bins
                z_high = z_min + (i + 1) * total_height / self.slice_bins
                layer_ranges.append((z_low, z_high))
        
        # Generate contour for each layer
        import open3d as o3d
        layers = []
        for i, (z_low, z_high) in enumerate(layer_ranges):
            # Point cloud extension: extend downward to fill gaps
            if self.enable_layer_point_extension:
                # Extend downward by specified distance, but not below z_min
                z_low_extended = max(z_min, z_low - self.layer_point_extension_distance)
                
                # Avoid excessive overlap with previous layer
                if i > 0:
                    prev_z_high = layer_ranges[i-1][1]
                    z_low_extended = max(z_low_extended, prev_z_high)
                
                rospy.logdebug(f"[UrinalDetector] Layer {i}: original=[{z_low:.3f}, {z_high:.3f}], extended=[{z_low_extended:.3f}, {z_high:.3f}]")
            else:
                z_low_extended = z_low
            
            # Extract points using extended range
            mask_extended = (z_vals >= z_low_extended) & (z_vals <= z_high)
            layer_pts = remain_points[mask_extended]
            
            if len(layer_pts) < 30:
                continue
            
            # 保存当前层点云供过滤使用
            pcd_layer = o3d.geometry.PointCloud()
            pcd_layer.points = o3d.utility.Vector3dVector(layer_pts)
            self._current_layer_pcd = pcd_layer
            
            # Generate contour using extended point cloud
            path = self._generate_layer_contour(layer_pts)
            
            if len(path) > 0:
                # Important: Keep path Z coordinate at original layer height
                # This ensures layer separation is maintained
                mask_original = (z_vals >= z_low) & (z_vals <= z_high)
                if np.sum(mask_original) > 0:
                    original_z_mean = np.mean(remain_points[mask_original, 2])
                    path[:, 2] = original_z_mean
                    rospy.logdebug(f"[UrinalDetector] Layer {i} path Z set to {original_z_mean:.3f}m (using {len(layer_pts)} extended points)")
                
                layers.append(path)
        
        if len(layers) == 0:
            return np.empty((0, 3))
        
        # Step 1: Unify rotation direction for all layers
        # Ensure all layers rotate in the same direction (clockwise or counter-clockwise)
        if len(layers) > 1:
            ref_direction = self._calculate_layer_direction(layers[0])
            rospy.logdebug(f"[UrinalDetector] Reference layer direction: {ref_direction}")
            
            for i in range(1, len(layers)):
                curr_direction = self._calculate_layer_direction(layers[i])
                
                # If direction is opposite to reference, reverse the layer
                if curr_direction * ref_direction < 0:
                    layers[i] = layers[i][::-1]  # Reverse point order
                    rospy.logdebug(f"[UrinalDetector] Layer {i} reversed to match direction")
        
        # 7. 层间连接 - 智能处理开口路径和闭合路径
        if len(layers) == 0:
            return np.empty((0, 3))
        
        # 初始化：第一层
        optimized_layers = [layers[0]]
        
        for i in range(1, len(layers)):
            prev_layer = optimized_layers[-1]
            curr_layer = layers[i]
            
            # ★ 判断当前层是否为开口路径
            closing_dist_curr = np.linalg.norm(curr_layer[-1] - curr_layer[0])
            segment_dists_curr = np.linalg.norm(np.diff(curr_layer, axis=0), axis=1)
            mean_segment_curr = float(np.mean(segment_dists_curr)) if len(segment_dists_curr) > 0 else 0.01
            is_open_curr = closing_dist_curr > mean_segment_curr * 2.0
            
            # 上一层末端点
            endp = prev_layer[-1]
            
            if is_open_curr:
                # ★ 开口路径：从端点连接，不旋转
                # 选择从上层末端到当前层首端或尾端中较近的连接
                dist_to_head = np.linalg.norm(curr_layer[0] - endp)
                dist_to_tail = np.linalg.norm(curr_layer[-1] - endp)
                
                if dist_to_tail < dist_to_head:
                    # 反转当前层，使其尾端成为起点
                    curr_layer = curr_layer[::-1].copy()
                    rospy.loginfo("[UrinalDetector] 层%d->%d: 开口路径，连接到尾端(反转)，距离=%.4fm", 
                                i-1, i, dist_to_tail)
                else:
                    rospy.loginfo("[UrinalDetector] 层%d->%d: 开口路径，连接到首端，距离=%.4fm", 
                                i-1, i, dist_to_head)
                
                optimized_layers.append(curr_layer)
            else:
                # ★ 闭合路径：旋转使最近点位于首位
                # 找最近点（正向）
                d = np.linalg.norm(curr_layer - endp, axis=1)
                j = int(np.argmin(d))
                
                # 尝试反向
                curr_layer_rev = curr_layer[::-1]
                d_rev = np.linalg.norm(curr_layer_rev - endp, axis=1)
                j_rev = int(np.argmin(d_rev))
                
                # 选择距离更近的方向
                if d_rev[j_rev] < d[j]:
                    # 使用反向 + 旋转
                    curr_layer = np.roll(curr_layer_rev, -j_rev, axis=0)
                    rospy.loginfo("[UrinalDetector] 层%d->%d: 闭合路径，反转+旋转%d点到首位，距离=%.4fm", 
                                i-1, i, j_rev, d_rev[j_rev])
                else:
                    # 使用正向 + 旋转
                    curr_layer = np.roll(curr_layer, -j, axis=0)
                    rospy.loginfo("[UrinalDetector] 层%d->%d: 闭合路径，旋转%d点到首位，距离=%.4fm", 
                                i-1, i, j, d[j])
                
                # 添加闭合点完成循环
                curr_layer_closed = np.vstack([curr_layer, curr_layer[0:1]])
                rospy.logdebug("[UrinalDetector] 层%d 添加闭合点，总计%d点", i, len(curr_layer_closed))
                optimized_layers.append(curr_layer_closed)
        
        # Stack all optimized layers
        return np.vstack(optimized_layers)
    
    def _generate_layer_contour(self, layer_points):
        """Single layer Alpha Shape boundary (from sink_detector)"""
        from scipy.spatial import Delaunay, ConvexHull
        from sklearn.cluster import DBSCAN
        
        if len(layer_points) < 20:
            return np.empty((0, 3))
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=0.02, min_samples=5).fit(layer_points)
        labels = clustering.labels_
        
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique_labels) == 0:
            main_pts = layer_points
        else:
            main_label = unique_labels[np.argmax(counts)]
            main_pts = layer_points[labels == main_label]
        
        if len(main_pts) < 20:
            return np.empty((0, 3))
        
        # PCA Project to 2D
        c = main_pts.mean(axis=0)
        X = main_pts - c
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        A = Vt[:2, :]
        pts2 = (A @ X.T).T
        
        # Alpha Shape
        try:
            order = self._alpha_shape_2d(pts2, self.alpha_value)
            if len(order) == 0:
                hull = ConvexHull(pts2)
                order = hull.vertices
        except:
            try:
                hull = ConvexHull(pts2)
                order = hull.vertices
            except:
                return np.empty((0, 3))
        
        # Project back to 3D
        layer2o = pts2[order]
        X3 = (A.T @ layer2o.T).T + c
        X3[:, 2] = np.mean(main_pts[:, 2])
        
        # Requirement 3: Boundary expansion (2D outward offset in XY plane)
        if self.boundary_expansion > 0:
            # Calculate geometric center of the contour (XY plane only)
            center_xy = X3[:, :2].mean(axis=0)
            
            # Calculate outward direction for each point (away from center)
            directions = X3[:, :2] - center_xy
            
            # Normalize directions (handle zero-length vectors)
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Prevent division by zero
            normalized_directions = directions / norms
            
            # Apply expansion (move points outward)
            X3[:, :2] += normalized_directions * self.boundary_expansion
            
            rospy.logdebug(f"[UrinalDetector] Applied boundary expansion: {self.boundary_expansion:.3f}m")
        
        # 6. 关键！过滤虚假路径 - 使用距离过滤移除虚假闭合
        if len(X3) > 0 and self.enable_path_filter and hasattr(self, '_current_layer_pcd'):
            X3 = self._filter_path_by_distance_to_cloud(
                X3, 
                self._current_layer_pcd,
                max_distance=self.path_filter_max_dist,
                min_segment_length=self.path_filter_min_segment
            )
        
        return X3
    
    def _alpha_shape_2d(self, pts2, alpha):
        """Simplified Alpha Shape boundary extraction (from sink_detector)"""
        from scipy.spatial import Delaunay
        from collections import defaultdict
        
        if len(pts2) < 4:
            return np.arange(len(pts2))
        
        tri = Delaunay(pts2)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                edges.add(edge)
        
        # Alpha filtering
        alpha_edges = []
        for i, j in edges:
            if np.linalg.norm(pts2[i] - pts2[j]) < 1.0 / alpha:
                alpha_edges.append((i, j))
        
        if not alpha_edges:
            return np.array([])
        
        # Build graph
        graph = defaultdict(list)
        for i, j in alpha_edges:
            graph[i].append(j)
            graph[j].append(i)
        
        boundary = [n for n, neighbors in graph.items() if len(neighbors) == 2]
        if len(boundary) < 3:
            return np.array([])
        
        # Path ordering
        visited = set()
        path = [boundary[0]]
        visited.add(boundary[0])
        current = boundary[0]
        
        while len(path) < len(boundary):
            neighbors = [n for n in graph[current] if n not in visited and n in boundary]
            if not neighbors:
                break
            current = neighbors[0]
            path.append(current)
            visited.add(current)
        
        return np.array(path) if len(path) >= 3 else np.array([])
    
    def _filter_path_by_distance_to_cloud(self, path_pts, pcd_uniform, max_distance=0.03, min_segment_length=5):
        """
        6. 关键！过滤虚假路径
        根据路径点到uniform点云的距离过滤路径点，保留最长连续有效段
        
        参数:
            path_pts: 路径点数组 (N, 3) - 通常是闭合路径
            pcd_uniform: uniform点云（实际表面几何）
            max_distance: 最大允许距离（米），超过此距离的点将被删除
            min_segment_length: 最小连续段长度，短于此长度的段将被丢弃
        
        返回:
            filtered_pts: 过滤后的路径点（可能不再闭合）
        """
        import open3d as o3d
        
        if len(path_pts) == 0 or not pcd_uniform.has_points():
            return path_pts
        
        # 构建KDTree
        kd_uniform = o3d.geometry.KDTreeFlann(pcd_uniform)
        
        # 计算每个路径点到点云的最近距离
        distances = []
        valid_mask = []
        
        for idx, pt in enumerate(path_pts):
            k, nn_idx, nn_dist2 = kd_uniform.search_knn_vector_3d(pt, 1)
            if k > 0:
                dist = np.sqrt(nn_dist2[0])
                distances.append(dist)
                valid_mask.append(dist <= max_distance)
            else:
                distances.append(float('inf'))
                valid_mask.append(False)
        
        valid_mask = np.array(valid_mask)
        distances = np.array(distances)
        
        if not np.any(valid_mask):
            rospy.logwarn("[UrinalDetector] 路径过滤: 所有点都被过滤掉了!")
            return np.empty((0, 3))
        
        # ★ 关键：找到最长的连续有效段（断开路径）
        segments = []
        start_idx = None
        
        for i in range(len(valid_mask)):
            if valid_mask[i]:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    segments.append((start_idx, i))
                    start_idx = None
        
        if start_idx is not None:
            segments.append((start_idx, len(valid_mask)))
        
        if len(segments) == 0:
            rospy.logwarn("[UrinalDetector] 路径过滤: 未找到有效段!")
            return np.empty((0, 3))
        
        # 过滤掉太短的段
        valid_segments = [(s, e) for s, e in segments if (e - s) >= min_segment_length]
        
        if len(valid_segments) == 0:
            # 回退：使用最长的段，即使它很短
            longest_seg = max(segments, key=lambda x: x[1] - x[0])
            valid_segments = [longest_seg]
            rospy.loginfo("[UrinalDetector] 使用最长段: [%d:%d] (%d点)", 
                         longest_seg[0], longest_seg[1], longest_seg[1] - longest_seg[0])
        
        # 选择最长的段作为主路径（假设开口形状有一个主要的连续弧段）
        longest_segment = max(valid_segments, key=lambda x: x[1] - x[0])
        start, end = longest_segment
        
        # ★★ 关键修复：检查是否为闭环且被错误截断
        # 如果首尾点距离很近（可能是闭环），且首尾都有效，则可能需要合并跨越边界的段
        if len(valid_segments) > 1:
            # 检查是否存在跨越数组边界的情况（第一段从0开始，最后一段到末尾）
            first_seg = min(valid_segments, key=lambda x: x[0])
            last_seg = max(valid_segments, key=lambda x: x[1])
            
            # 如果第一段从索引0开始，最后一段到数组末尾结束，且首尾点距离近
            if first_seg[0] == 0 and last_seg[1] == len(path_pts):
                closing_distance = np.linalg.norm(path_pts[0] - path_pts[-1])
                avg_segment_dist = np.mean(np.linalg.norm(np.diff(path_pts, axis=0), axis=1))
                
                # 如果首尾距离小于平均段长的2倍，说明是闭环，需要合并
                if closing_distance < avg_segment_dist * 2.0:
                    # 合并跨界的两段
                    combined_length = (first_seg[1] - first_seg[0]) + (last_seg[1] - last_seg[0])
                    
                    # 如果合并后比当前最长段更长，则使用合并段
                    if combined_length > (longest_segment[1] - longest_segment[0]):
                        rospy.loginfo("[UrinalDetector] 检测到闭环跨界情况，合并首尾段: [%d:%d]+[%d:%d] → %d点",
                                     last_seg[0], last_seg[1], first_seg[0], first_seg[1], combined_length)
                        
                        # 重新组合：从last_seg开始，到first_seg结束
                        filtered_pts = np.vstack([
                            path_pts[last_seg[0]:last_seg[1]],
                            path_pts[first_seg[0]:first_seg[1]]
                        ])
                        
                        # 统计信息
                        removed_count = len(path_pts) - combined_length
                        avg_dist = float(np.mean(distances))
                        max_dist = float(np.max(distances))
                        valid_count = np.sum(valid_mask)
                        
                        rospy.loginfo("[UrinalDetector] 路径过滤: 原始=%d点, 距离有效=%d点, 连续段数=%d, 合并跨界段=%d点, 删除=%d (%.1f%%)",
                                     len(path_pts), valid_count, len(valid_segments), 
                                     combined_length, removed_count,
                                     100.0 * removed_count / len(path_pts) if len(path_pts) > 0 else 0)
                        rospy.loginfo("[UrinalDetector] 距离统计: 平均=%.4fm, 最大=%.4fm", avg_dist, max_dist)
                        
                        return filtered_pts
        
        # 统计信息
        removed_count = len(path_pts) - (end - start)
        avg_dist = float(np.mean(distances))
        max_dist = float(np.max(distances))
        valid_count = np.sum(valid_mask)
        
        rospy.loginfo("[UrinalDetector] 路径过滤: 原始=%d点, 距离有效=%d点, 连续段数=%d, 选择最长段[%d:%d]=%d点, 删除=%d (%.1f%%)",
                     len(path_pts), valid_count, len(valid_segments), 
                     start, end, end - start, removed_count,
                     100.0 * removed_count / len(path_pts) if len(path_pts) > 0 else 0)
        rospy.loginfo("[UrinalDetector] 距离统计: 平均=%.4fm, 最大=%.4fm", avg_dist, max_dist)
        
        # 提取最长段
        filtered_pts = path_pts[start:end].copy()
        
        # ★★★ 二次过滤：检查连线是否穿过空白区域（关键改进！）
        # 两个点都离点云近，但连线可能穿过空白区域
        if len(filtered_pts) > 2:
            invalid_edges = []
            
            # 对每条边进行采样检查
            for i in range(len(filtered_pts) - 1):
                p1 = filtered_pts[i]
                p2 = filtered_pts[i + 1]
                
                # 计算边长
                edge_length = np.linalg.norm(p2 - p1)
                
                # 在边上采样多个中间点（每1cm一个点）
                n_samples = max(2, int(edge_length / 0.01))
                t_values = np.linspace(0, 1, n_samples)
                
                # 检查每个采样点到点云的距离
                edge_valid = True
                for t in t_values[1:-1]:  # 跳过端点（已经验证过）
                    sample_pt = p1 + t * (p2 - p1)
                    
                    # 查询最近点距离
                    k, nn_idx, nn_dist2 = kd_uniform.search_knn_vector_3d(sample_pt, 1)
                    if k > 0:
                        sample_dist = np.sqrt(nn_dist2[0])
                        if sample_dist > max_distance:
                            edge_valid = False
                            rospy.logdebug("[UrinalDetector] 边[%d→%d]的采样点t=%.2f距离点云%.4fm（超过阈值%.4fm）",
                                         i, i+1, t, sample_dist, max_distance)
                            break
                    else:
                        edge_valid = False
                        break
                
                if not edge_valid:
                    invalid_edges.append(i)
            
            # 如果发现无效边，在这些位置断开路径
            if len(invalid_edges) > 0:
                rospy.loginfo("[UrinalDetector] 检测到%d条边的连线穿过空白区域", len(invalid_edges))
                
                # 在所有无效边处断开，形成多个子段
                sub_segments = []
                segment_start = 0
                
                for idx in invalid_edges:
                    # idx是边的索引，在idx+1处断开
                    if idx + 1 - segment_start >= min_segment_length:
                        sub_segments.append((segment_start, idx + 1))
                    segment_start = idx + 1
                
                # 最后一个子段
                if len(filtered_pts) - segment_start >= min_segment_length:
                    sub_segments.append((segment_start, len(filtered_pts)))
                
                if len(sub_segments) > 0:
                    # 选择最长的子段
                    longest_sub = max(sub_segments, key=lambda x: x[1] - x[0])
                    rospy.loginfo("[UrinalDetector] 二次过滤（连线采样）：断开为%d个子段，选择最长子段[%d:%d]=%d点",
                                 len(sub_segments), longest_sub[0], longest_sub[1], 
                                 longest_sub[1] - longest_sub[0])
                    
                    filtered_pts = filtered_pts[longest_sub[0]:longest_sub[1]].copy()
                else:
                    rospy.logwarn("[UrinalDetector] 二次过滤后无有效子段，保留原路径")
        
        return filtered_pts
    
    def _calculate_layer_direction(self, layer_path):
        """
        Calculate rotation direction of layer path using signed area (shoelace formula)
        Returns: +1 for counter-clockwise, -1 for clockwise
        """
        if len(layer_path) < 3:
            return 1
        
        # Shoelace formula for signed area
        area = 0.0
        for i in range(len(layer_path)):
            j = (i + 1) % len(layer_path)
            area += layer_path[i, 0] * layer_path[j, 1]
            area -= layer_path[j, 0] * layer_path[i, 1]
        
        return 1 if area > 0 else -1
    
    def _find_normal_connection_point(self, prev_layer, curr_layer):
        """
        Find optimal connection point on curr_layer using normal direction from prev_layer endpoint.
        This ensures layer connections are perpendicular to path direction.
        
        Args:
            prev_layer: Previous layer path (Nx3)
            curr_layer: Current layer path (Mx3)
        
        Returns:
            best_idx: Index in curr_layer for optimal start point
        """
        if len(prev_layer) < 2 or len(curr_layer) < 2:
            return 0
        
        # Calculate tangent at end of previous layer (in XY plane)
        tangent_xy = prev_layer[-1, :2] - prev_layer[-2, :2]
        tangent_norm = np.linalg.norm(tangent_xy)
        
        if tangent_norm < 1e-6:
            return 0  # Degenerate case
        
        tangent_xy = tangent_xy / tangent_norm
        
        # Calculate normal directions (perpendicular to tangent in XY plane)
        # Two choices: rotate 90° left or right
        normal_left = np.array([-tangent_xy[1], tangent_xy[0]])
        normal_right = np.array([tangent_xy[1], -tangent_xy[0]])
        
        end_point_xy = prev_layer[-1, :2]
        
        # Find best connection point for both normal directions
        best_idx = 0
        min_score = float('inf')
        
        for normal in [normal_left, normal_right]:
            for j in range(len(curr_layer)):
                # Vector from prev endpoint to curr_layer point
                vec_to_point = curr_layer[j, :2] - end_point_xy
                
                # Project onto normal direction
                proj_along_normal = np.dot(vec_to_point, normal)
                
                # Only consider points in forward normal direction
                if proj_along_normal <= 0:
                    continue
                
                # Calculate perpendicular distance from normal ray
                perp_component = vec_to_point - proj_along_normal * normal
                perp_dist = np.linalg.norm(perp_component)
                
                # Score: minimize perpendicular distance, slight penalty for far points
                # This finds the point closest to the normal ray
                score = perp_dist + 0.1 * proj_along_normal
                
                if score < min_score:
                    min_score = score
                    best_idx = j
        
        rospy.logdebug(f"[UrinalDetector] Normal connection: best_idx={best_idx}, score={min_score:.4f}")
        return best_idx
    
    def _add_orientation_to_path(self, path_xyz):
        """
        Add orientation (Roll, Pitch, Yaw) to XYZ path points.
        Returns array of shape (N, 6) with [x, y, z, roll, pitch, yaw]
        """
        if len(path_xyz) == 0:
            return np.empty((0, 6))
        
        # Extract XYZ coordinates
        x = path_xyz[:, 0]
        y = path_xyz[:, 1]
        z = path_xyz[:, 2]
        
        # Calculate center of path for tool pointing target
        x0_min = np.min(x)
        x0_max = np.max(x)
        interpolated_x0 = x0_min + self.tool_pointing_x_offset_ratio * (x0_max - x0_min)
        
        # Center Y and Z for target points
        y0 = np.mean(y)
        
        # Target points for tool orientation (pointing inward + upward offset)
        target_points = np.array([
            np.full_like(x, interpolated_x0),
            np.full_like(y, y0),
            z + self.tool_pointing_height
        ])
        current_points = np.array([x, y, z])
        
        # Calculate Z-axis (pointing from current point to target)
        z_axis = target_points - current_points
        z_axis = z_axis / np.linalg.norm(z_axis, axis=0)
        
        # Base XY direction from predefined RPY
        xy_dir = np.array([np.cos(self.predefined_rpy[2]), np.sin(self.predefined_rpy[2]), 0.0])
        xy_dir_alt = np.array([0.0, 1.0, 0.0])
        
        # Gram-Schmidt orthonormalization to get Y-axis
        dot_y = xy_dir @ z_axis
        y_axis = xy_dir[:, np.newaxis] - dot_y * z_axis
        
        # Normalize per-point
        norms_y = np.linalg.norm(y_axis, axis=0)
        
        # Fallback for degenerate cases
        is_degenerate = norms_y < 1e-10
        if np.any(is_degenerate):
            dot_alt = xy_dir_alt @ z_axis
            y_axis_alt = xy_dir_alt[:, np.newaxis] - dot_alt * z_axis
            y_axis[:, is_degenerate] = y_axis_alt[:, is_degenerate]
            norms_y[is_degenerate] = np.linalg.norm(y_axis[:, is_degenerate], axis=0)
        
        y_axis = y_axis / norms_y
        
        # Compute X-axis = Y × Z (right-hand rule)
        x_axis = np.cross(y_axis, z_axis, axis=0)
        x_axis = x_axis / np.linalg.norm(x_axis, axis=0)
        
        # Extract RPY from rotation matrix
        yaw = np.arctan2(x_axis[1], x_axis[0])
        pitch = np.arcsin(x_axis[2])
        roll = np.arctan2(-y_axis[2], z_axis[2])
        
        # Combine XYZ and RPY
        return np.column_stack([x, y, z, roll, pitch, yaw])

    def generate_clean_path(self, pts3d):
        """Generate cleaning path for urinal in local coordinates"""
        try:
            if self.use_alpha_shape:
                rospy.loginfo("[UrinalDetector] Using Alpha Shape algorithm (urinal-optimized)...")
                return self._generate_path_alpha_shape(pts3d)
            else:
                rospy.loginfo("[UrinalDetector] Using legacy spiral algorithm...")
                return self._generate_path_legacy_spiral(pts3d)
                
        except Exception as e:
            rospy.logerr(f"Error generating cleaning path: {str(e)}")
            return None

    def _generate_path_legacy_spiral(self, pts3d):
        """Legacy spiral algorithm (geometry analysis)"""
        rospy.loginfo("Starting urinal geometry analysis...")
        
        # Analyze urinal geometry from point cloud
        geometry_params = self.analyze_urinal_geometry(pts3d)
        
        rospy.loginfo(f"Urinal geometry analyzed: "
                     f"Top D: {geometry_params['d_top']:.3f}m, "
                     f"Bottom D: {geometry_params['d_bottom']:.3f}m, "
                     f"Height: {geometry_params['total_height']:.3f}m, "
                     f"Opening: {geometry_params['opening_angle']:.1f}Â°")
        
        # Generate spiral cleaning path in local cylinder coordinates
        cleaning_path = self.generate_spiral_path(geometry_params)
        
        rospy.loginfo(f"Generated cleaning path with {len(cleaning_path)} points")
        
        return cleaning_path

    def cleanup(self):
        """Cleanup resources"""
        self.is_active = False
        if hasattr(self, 'proc_timer'):
            self.proc_timer.shutdown()
        if hasattr(self, 'repub_timer'):
            self.repub_timer.shutdown()
        rospy.loginfo("UrinalDetector cleaned up")


def main():
    """Main entry point"""
    rospy.init_node("urinal_detector_node")
    rospy.loginfo("[UrinalDetector] Node starting...")
    
    detector = UrinalDetector()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("[UrinalDetector] Shutting down...")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
