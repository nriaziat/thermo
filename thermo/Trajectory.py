import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import make_interp_spline

class Trajectory:
    """
    A class to represent a trajectory of points and normals. All data is in the robot base coordinate frame.
    """
    def __init__(self, dir: str):
        self.points: np.ndarray = np.load(dir.strip("/") + "/points.npy")
        self.normals: np.ndarray = np.load(dir.strip("/") + "/normals.npy")
        assert len(self.points) == len(self.normals), "Points and normals must have the same length"
        for i in range(len(self.normals)):
            self.normals[i] /= np.linalg.norm(self.normals[i])
        self._len = len(self.points)
        self._orientations: list[R] = []
        self._compute_orientation()

    def _compute_projected_tangents(self):
        # compute the tangent at each point, projected onto the tangent plane
        self.projected_tangents = np.zeros((self._len, 3))
        self.spline = make_interp_spline(np.arange(self._len), self.points, k=3)
        self.tangent_spline = self.spline.derivative()
        self.tangents = self.tangent_spline(np.arange(self._len))
        for i in range(self._len):
            self.projected_tangents[i] = self.tangent_spline(i) / np.linalg.norm(self.tangent_spline(i))
            assert np.dot(self.projected_tangents[i], self.normals[i]) < 0.9, f"Tangent is not orthogonal to normal: {self.projected_tangents[i]} {self.normals[i]} = {np.dot(self.projected_tangents[i], self.normals[i])}"
            self.projected_tangents[i] -= np.dot(self.projected_tangents[i], self.normals[i]) * self.normals[i]
            self.projected_tangents[i] /= np.linalg.norm(self.projected_tangents[i])
            assert np.isclose(np.dot(self.projected_tangents[i], self.normals[i]), 0, atol=1e-5), f"Tangent is not orthogonal to normal: {self.projected_tangents[i]} {self.normals[i]} = {np.dot(self.projected_tangents[i], self.normals[i])}"


    def _compute_orientation(self):
        # for each point, compute the orientation of the end effector in the global frame that would align the z-axis with the normal and best align the x-axis with the tangent
        self._compute_projected_tangents()
        for i in range(self._len):
            z = -self.normals[i] 
            x = self.projected_tangents[i]
            y = np.cross(z, x)
            rot_mat = np.array([x, y, z])
            # assert np.isclose(np.linalg.det(rot_mat), 1, atol=1e-5), f"Rotation matrix is not valid: {rot_mat}: Det = {np.linalg.det(rot_mat)}."
            # assert np.allclose(rot_mat @ rot_mat.T, np.eye(3), atol=1e-5), f"Rotation matrix is not valid: {rot_mat}: R^T R =   {rot_mat @ rot_mat.T}."
            self._orientations.append(R.from_matrix(rot_mat))

    
    def __len__(self):
        return self._len
    
    def __getitem__(self, key) -> tuple[np.ndarray, R]:
        return self.points[key], self._orientations[key]
    
    def __iter__(self):
        yield from zip(self.points, self._orientations)
    
