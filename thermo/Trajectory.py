import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline
from geometry_msgs.msg import Pose, Point, TransformStamped
from tf2_geometry_msgs import do_transform_pose
from typing import Tuple
from copy import deepcopy

def pose_to_scipy(pose: Pose) -> R:
    r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    return r

def pose_to_np(pose: Pose) -> R:
    p = np.array([pose.position.x, pose.position.y, pose.position.z])
    return p

def position_error(pose1: Pose, pose2: Pose, tf: TransformStamped=None) -> float:
    """
    Compute the position error between two poses
    """
    if tf is None:
        return np.linalg.norm(np.array([pose1.position.x, pose1.position.y, pose1.position.z]) - np.array([pose2.position.x, pose2.position.y, pose2.position.z]))
    pose1_tf = do_transform_pose(pose1, tf)
    return np.linalg.norm(np.array([pose1_tf.position.x, pose1_tf.position.y, pose1_tf.position.z]) - np.array([pose2.position.x, pose2.position.y, pose2.position.z]))


def orientation_error(pose1: Pose, pose2: Pose, tf: TransformStamped=None) -> float:
    """
    Compute the orientation error between two poses
    """
    if tf is None:
        r1 = R.from_quat(np.array([pose1.orientation.x, pose1.orientation.y, pose1.orientation.z, pose1.orientation.w]))
        r2 = R.from_quat(np.array([pose2.orientation.x, pose2.orientation.y, pose2.orientation.z, pose2.orientation.w]))
    else:
        pose1_tf = do_transform_pose(pose1, tf)
        r1 = R.from_quat(np.array([pose1_tf.orientation.x, pose1_tf.orientation.y, pose1_tf.orientation.z, pose1_tf.orientation.w]))
        r2 = R.from_quat(np.array([pose2.orientation.x, pose2.orientation.y, pose2.orientation.z, pose2.orientation.w]))
    return (r1 * r2.inv()).magnitude()

class Trajectory:
    """
    A class to represent a trajectory of points and normals. All data is in the robot base coordinate frame.
    """
    def __init__(self, points: np.ndarray, normals: np.ndarray, resample_distance: float = 0, reverse: bool = False, 
                 smooth_trajectory: bool = False, interpolation_factor: int = 1, retraction_position: float = 0, 
                 average_normals: bool = False, cut_depth_m: float = 0):
        """
        @param points: A numpy array of shape (n, 3) representing the points along the trajectory
        @param normals: A numpy array of shape (n, 3) representing the normals at each point
        @param resample_distance: The minimum distance between points in the resampled trajectory
        @param smooth_trajectory: run exponential smoothing on trajectory
        @param interpolation_factor: how many points to interpolate between given points.
        """

        self.points: list = [points[0]]
        self.normals: list = [normals[0]]
        self.interpolation_factor = interpolation_factor
        self.smooth_trajectory = smooth_trajectory
        self.cut_depth_m = cut_depth_m
        self.average_normals = average_normals
        self.retrajection_position = retraction_position
   
        for point, normal in zip(points, normals):
            if np.linalg.norm(point - self.points[-1]) > resample_distance:
                self.points.append(point)
                self.normals.append(normal)
    
        self.reversed = reverse  ## if true, cw, else ccw
        if reverse:
            self.points.reverse()
            self.normals.reverse()

        assert len(self.points) == len(self.normals), "Points and normals must have the same length"
        self.points = np.array(self.points)
        self.normals = np.array(self.normals)

        self.points = np.roll(self.points, int(self.retrajection_position / self.interpolation_factor), axis=0)
        self.normals = np.roll(self.normals, int(self.retrajection_position / self.interpolation_factor), axis=0)

        self.poses: list[Pose] = []
        for i in range(len(self.normals)):
            self.normals[i] /= -np.linalg.norm(self.normals[i])
        self._set_cut_depth(cut_depth_m)
        self._compute_pose(average_normals=self.average_normals)
        self.poses[0].orientation = deepcopy(self.poses[1].orientation)
        self.poses[-1].orientation = deepcopy(self.poses[-2].orientation)

    def change_starting_point(self, starting_point: float):
        self.points = np.roll(self.points, int(starting_point / self.interpolation_factor), axis=0)
        self.normals = np.roll(self.normals, int(starting_point / self.interpolation_factor), axis=0)

        self.poses: list[Pose] = []
        for i in range(len(self.normals)):
            self.normals[i] /= np.linalg.norm(self.normals[i])
        self._set_cut_depth(self.cut_depth_m)
        self._compute_pose(average_normals=self.average_normals)
        self.poses[0].orientation = deepcopy(self.poses[1].orientation)
        self.poses[-1].orientation = deepcopy(self.poses[-2].orientation)

    def _set_cut_depth(self, cut_depth_m: float) -> None:
        for point, normal in zip(self.points, self.normals):
            point -= cut_depth_m * normal
        self.cut_depth_m = cut_depth_m

    def reverse_trajectory(self):
        self.poses.reverse()
        self.reversed = not self.reversed

    def _compute_retraction_points(self, retraction_distance: float) -> Tuple[Pose, Pose]:
        entry_point = np.array([self.poses[0].position.x, self.poses[0].position.y, self.poses[0].position.z])
        entry_point += self.normals[0] * (retraction_distance + self.cut_depth_m)
        entry_pose = Pose()
        entry_pose.position.x = entry_point[0]
        entry_pose.position.y = entry_point[1]
        entry_pose.position.z = entry_point[2]
        entry_pose.orientation = deepcopy(self.poses[0].orientation)
        return entry_pose, deepcopy(entry_pose)

    def _compute_projected_tangents(self):
        # compute the tangent at each point, projected onto the tangent plane
        self.projected_tangents = np.zeros((len(self.points), 3))
        self.tangents = -np.diff(self.points, axis=0)
        self.tangents = np.vstack([self.tangents[0], self.tangents])
        self.tangents /= np.linalg.norm(self.tangents, axis=1)[:, None]

        for i in range(len(self.projected_tangents)):
            self.projected_tangents[i] = self.tangents[i] - np.dot(self.tangents[i], self.normals[i]) * self.normals[i]
            self.projected_tangents[i] /= np.linalg.norm(self.projected_tangents[i])

        # for i in range(1, len(self.projected_tangents)):
        #     self.projected_tangents[i] = 0.1 * self.projected_tangents[i] + 0.9 * self.projected_tangents[i-1]

    def _compute_pose(self, average_normals: bool):
        # for each point, compute the orientation of the end effector in the global frame that would align the z-axis with the normal and best align the x-axis with the tangent
        if average_normals:
            alpha = 0.1
            mean_orientation = np.mean(self.normals, axis=0)
            for i in range(len(self.normals)):
                self.normals[i] = alpha * self.normals[i] + (1-alpha) * mean_orientation
                self.normals[i] /= np.linalg.norm(self.normals[i])
        else:
            self._smooth_normals()
        self._compute_projected_tangents()
        for i in range(len(self.projected_tangents)):
            z = -self.normals[i]
            y = self.projected_tangents[i]
            x = np.cross(y, z)
            rot_mat = np.array([x, y, z]).T
            pose = Pose()
            pose.position = Point(x=float(self.points[i][0]), y=float(self.points[i][1]), z=float(self.points[i][2]))
            r = R.from_matrix(rot_mat)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = r.as_quat()
            self.poses.append(pose)
        self.poses.append(self.poses[0])
        self.poses.append(self.poses[1])


        # smooth entry and exit 
        # self.poses[0].orientation = self.poses[1].orientation
        # self.poses[-1].orientation = self.poses[-1].orientation
        if self.smooth_trajectory:
            self._smooth_poses()
        if self.interpolation_factor > 1:
            self._interpolate_poses()

        start_pose, end_pose = self._compute_retraction_points(3e-2)
        self.poses = [start_pose] + self.poses
        end_pose.orientation = deepcopy(self.poses[-1].orientation)
        self.poses.append(end_pose)

    def _interpolate_poses(self):
        # interpolate between poses
        interpolated_poses = []
        rotations = R.from_quat([[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] for pose in self.poses])
        rot_spline = RotationSpline(range(len(self.poses)), rotations)
        pose = Pose()
        pose.position.x = self.poses[0].position.x 
        pose.position.y = self.poses[0].position.y 
        pose.position.z = self.poses[0].position.z
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = rotations[0].as_quat()
        interpolated_poses.append(pose)        
        for i in range(len(self.poses)-1):
            t = np.linspace(i, i+1, self.interpolation_factor)
            for time in t:
                r = rot_spline(time)
                pose = Pose()
                pose.position.x = self.poses[i].position.x + (self.poses[i+1].position.x - self.poses[i].position.x) * (time - i)
                pose.position.y = self.poses[i].position.y + (self.poses[i+1].position.y - self.poses[i].position.y) * (time - i)
                pose.position.z = self.poses[i].position.z + (self.poses[i+1].position.z - self.poses[i].position.z) * (time - i)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = r.as_quat()
                interpolated_poses.append(pose)
        pose = Pose()
        pose.position.x = self.poses[-1].position.x 
        pose.position.y = self.poses[-1].position.y 
        pose.position.z = self.poses[-1].position.z
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = rotations[-1].as_quat()
        interpolated_poses.append(pose)        
        self.poses = interpolated_poses

    def _smooth_poses(self):
        window = 2
        alpha = 0.5
        for i in range(len(self.poses) - window + 1):
            poses = self.poses[i:i+window]
            rots = R.from_quat([[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in poses])
            mean_rot = rots.mean(weights=[alpha, 1-alpha])
            for pose in poses:
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = mean_rot.as_quat()

    def _smooth_normals(self):
        window = 2
        alpha = 0.5
        for i in range(len(self.normals) - window + 1):
            normals = self.normals[i:i+window]
            mean_normal = np.mean(normals, axis=0)
            mean_normal /= np.linalg.norm(mean_normal)
            for normal in normals:
                normal[:] = mean_normal[:]

    def find_closest_orientation(self, goal_pose: Pose) -> int:
        # print(f"Goal Pose: {pose_to_scipy(goal_pose).as_euler('zyx')}")
        min_error = np.inf
        min_error_pose = None
        for i, pose in enumerate(self.poses[1:-1]):  # dont iterate over retraction poses
            error = orientation_error(pose, goal_pose)
            if error < min_error:
                min_error = error
                min_error_pose = i + 1
                # print(f"Closest match: {pose_to_scipy(pose).as_euler('zyx')}")
        # print(f"Orientation Error: {min_error}")
        return min_error_pose

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, key) -> Pose:
        return self.poses[key]
    
    def __iter__(self):
        yield from self.poses
