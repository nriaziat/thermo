import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp, RotationSpline
from geometry_msgs.msg import Pose, Point, TransformStamped
from tf2_geometry_msgs import do_transform_pose

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
    def __init__(self, points: np.ndarray, normals: np.ndarray, resample_distance: float = 2e-3):
        """
        @param points: A numpy array of shape (n, 3) representing the points along the trajectory
        @param normals: A numpy array of shape (n, 3) representing the normals at each point
        @param resample_distance: The minimum distance between points in the resampled trajectory
        """

        self.points: list = [points[0]]
        self.normals: list = [normals[0]]

        for point, normal in zip(points, normals):
            if np.linalg.norm(point - self.points[-1]) > resample_distance:
                self.points.append(point)
                self.normals.append(normal)

        self.points = np.array(self.points)
        self.normals = np.array(self.normals)
        self.poses: list[Pose] = []
        assert len(self.points) == len(self.normals), "Points and normals must have the same length"
        for i in range(len(self.normals)):
            self.normals[i] /= np.linalg.norm(self.normals[i])
        self._compute_pose()

    def _compute_projected_tangents(self):
        # compute the tangent at each point, projected onto the tangent plane
        self.projected_tangents = np.zeros((len(self.points), 3))
        self.tangents = np.diff(self.points, axis=0)
        self.tangents = np.vstack([self.tangents, self.tangents[-1]])
        self.tangents /= np.linalg.norm(self.tangents, axis=1)[:, None]

        for i in range(len(self.poses)):
            self.projected_tangents[i] = self.tangents[i] - np.dot(self.projected_tangents[i], self.normals[i]) * self.normals[i]
            self.projected_tangents[i] /= np.linalg.norm(self.projected_tangents[i])

    def _compute_pose(self):
        # for each point, compute the orientation of the end effector in the global frame that would align the z-axis with the normal and best align the x-axis with the tangent
        self._compute_projected_tangents()
        for i in range(len(self.points)):
            z = -self.normals[i] 
            x = self.projected_tangents[i]
            y = np.cross(z, x)
            rot_mat = np.array([x, y, z]).T
            pose = Pose()
            pose.position = Point(x=float(self.points[i][0]), y=float(self.points[i][1]), z=float(self.points[i][2]))
            r = R.from_matrix(rot_mat)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = r.as_quat()
            self.poses.append(pose)

        # smooth entry and exit 
        self.poses[0].orientation = self.poses[1].orientation
        self.poses[-1].orientation = self.poses[-1].orientation
        self._smooth_poses()
        self._interpolate_poses()

    def _interpolate_poses(self):
        # interpolate between poses
        interpolated_poses = []
        rotations = R.from_quat([[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] for pose in self.poses])
        rot_spline = RotationSpline(range(len(self.poses)), rotations)
        for i in range(len(self.poses) - 1):
            t = np.linspace(i, i+1, 10)
            for time in t:
                r = rot_spline(time)
                pose = Pose()
                pose.position.x = self.poses[i].position.x + (self.poses[i+1].position.x - self.poses[i].position.x) * (time - i)
                pose.position.y = self.poses[i].position.y + (self.poses[i+1].position.y - self.poses[i].position.y) * (time - i)
                pose.position.z = self.poses[i].position.z + (self.poses[i+1].position.z - self.poses[i].position.z) * (time - i)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = r.as_quat()
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

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, key) -> Pose:
        return self.poses[key]
    
    def __iter__(self):
        yield from self.poses
