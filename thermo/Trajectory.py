import numpy as np
from scipy.spatial.transform import Rotation as R

class Trajectory:
    """
    A class to represent a trajectory of points and normals. All data is in the robot coordinate frame.
    """
    def __init__(self, dir: str):
        self.points = np.load(dir.strip("/") + "/points.npy")
        self.normals = np.load(dir.strip("/") + "/normals.npy")
        assert len(self.points) == len(self.normals), "Points and normals must have the same length"
        for i in range(len(self.points)):
            self.normals[i] = R.align_vectors(np.array([0, 0, 1]), self.normals[i])[0].as_quat()
        self._len = len(self.points)
        self._compute_orientation()

    def _compute_orientation(self):
        # for each point, compute the orientation of the end effector in the global frame that would align the z-axis with the normal and best align the x-axis with the tangent
        self._compute_tangents()
        self._orientations = np.zeros((self._len, 4))
        for i in range(self._len):
            z = self.normals[i]
            x = self._tangents[i]
            y = np.cross(z, x)
            self._orientations[i] = R.from_matrix(np.array([x, y, z])).as_quat()

    def _compute_tangents(self):
        # compute the tangent at each point, projected onto the tangent plane
        self._tangents = np.zeros((self._len, 3))
        for i in range(0, self._len - 1):
            tangent = (self.points[i + 1] - self.points[i])
            tangent /= np.linalg.norm(tangent)
            self._tangents[i] = tangent - np.dot(tangent, self.normals[i]) * self.normals[i]
            self._tangents[i] /= np.linalg.norm(self._tangents[i])
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, key) -> tuple:
        return self.points[key], self._orientations[key]
    
    def __iter__(self):
        yield from zip(self.points, self._orientations)
    
