import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float

    def __array__(self):
        return np.array([self.x, self.y, self.z])
    
    def __getitem__(self, key):
        return np.array(self)[key]

@dataclass
class Pose:
    position: Point
    orientation: R

class Trajectory:
    """
    A class to represent a trajectory of points and normals. All data is in the robot base coordinate frame.
    """
    def __init__(self, points: np.ndarray, normals: np.ndarray):
        self.points: np.ndarray = points
        self.normals: np.ndarray = normals
        self.poses: list[Pose] = []
        assert len(self.points) == len(self.normals), "Points and normals must have the same length"
        for i in range(len(self.normals)):
            self.normals[i] /= np.linalg.norm(self.normals[i])
        self._len = len(self.points)
        self._compute_pose()

    def _compute_projected_tangents(self):
        # compute the tangent at each point, projected onto the tangent plane
        self.projected_tangents = np.zeros((self._len, 3))
        self.tangents = np.diff(self.points, axis=0)
        self.tangents = np.vstack([self.tangents, self.tangents[-1]])
        self.tangents /= np.linalg.norm(self.tangents, axis=1)[:, None]

        for i in range(self._len):
            self.projected_tangents[i] = self.tangents[i] - np.dot(self.projected_tangents[i], self.normals[i]) * self.normals[i]
            self.projected_tangents[i] /= np.linalg.norm(self.projected_tangents[i])

    def _compute_pose(self):
        # for each point, compute the orientation of the end effector in the global frame that would align the z-axis with the normal and best align the x-axis with the tangent
        self._compute_projected_tangents()
        for i in range(self._len):
            z = -self.normals[i] 
            x = self.projected_tangents[i]
            y = np.cross(z, x)
            rot_mat = np.array([x, y, z]).T
            # assert np.isclose(np.linalg.det(rot_mat), 1, atol=1e-5), f"Rotation matrix is not valid: {rot_mat}: Det = {np.linalg.det(rot_mat)}."
            # assert np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-5), f"Rotation matrix is not valid: {rot_mat}: R^T R =   {rot_mat @ rot_mat.T}."
            pose = Pose(Point(*self.points[i]), R.from_matrix(rot_mat))
            self.poses.append(pose)

    
    def __len__(self):
        return self._len
    
    def __getitem__(self, key) -> Pose:
        return self.poses[key]
    
    def __iter__(self):
        yield from self.poses
    
