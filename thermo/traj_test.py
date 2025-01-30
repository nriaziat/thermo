from thermo.Trajectory import Trajectory
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

def plot_trajectory(traj: Trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for point, orientation in traj:
        point *= 1000
        orientation = R.from_quat(orientation).as_matrix()
        x_axis = orientation[:, 0] / np.linalg.norm(orientation[:, 0])
        y_axis = orientation[:, 1] / np.linalg.norm(orientation[:, 1])
        z_axis = orientation[:, 2] / np.linalg.norm(orientation[:, 2])
        ax.quiver(point[0], point[1], point[2], x_axis[0], x_axis[1], x_axis[2], color='r')
        ax.quiver(point[0], point[1], point[2], y_axis[0], y_axis[1], y_axis[2], color='g')
        ax.quiver(point[0], point[1], point[2], z_axis[0], z_axis[1], z_axis[2], color='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("trajectory.png", dpi=300)

if __name__ == "__main__":
    traj = Trajectory("./src/trajectory")
    plot_trajectory(traj)