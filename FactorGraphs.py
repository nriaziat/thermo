import argparse
from typing import List, Tuple
import pickle as pkl
import gtsam
import numpy as np
from gtsam import ISAM2, noiseModel
from gtsam.symbol_shorthand import L, V, X, P
from utils import find_tooltip

class CameraMeasurement:
    """An instance of a Camera tip position measurement."""
    def __init__(self, position: np.array):
        self.position = position


def loadCameraData(data_log: str) -> List[CameraMeasurement]:
    """Helper to load the GPS data."""
    # Read camera data
    # Time,X,Y,Z
    with open(data_log, 'rb') as f:
        data = pkl.load(f)
    camera_measurements = []
    frames = data['thermal_frames']
    for frame in frames:
        position = find_tooltip(frame, 70)
        position = np.asfortranarray(position).astype(np.float64)
        camera_measurements.append(CameraMeasurement(position))

    return camera_measurements



def optimize(camera_measurements: List[CameraMeasurement],
             sigma_init_x: gtsam.noiseModel.Diagonal,
             sigma_init_v: gtsam.noiseModel.Diagonal,
             sigma_init_b: gtsam.noiseModel.Diagonal,
             noise_model_gps: gtsam.noiseModel.Diagonal,
             first_camera_pose: int,
             camera_skip: int) -> gtsam.ISAM2:
    """Run ISAM2 optimization on the measurements."""
    # Set initial conditions for the estimated trajectory
    # initial pose is the reference frame (navigation frame)
    current_pose_global = camera_measurements[first_camera_pose].position

    # the vehicle is stationary at the beginning at position 0,0
    current_velocity_global = np.zeros(2)

    current_param = np.array([1, 1], dtype=np.float64, order='F')
    current_landmark = camera_measurements[first_camera_pose].position

    # Set ISAM2 parameters and create ISAM2 solver object
    isam_params = gtsam.ISAM2Params()
    isam_params.setFactorization("CHOLESKY")
    isam_params.relinearizeSkip = 10

    isam = gtsam.ISAM2(isam_params)

    # Create the factor graph and values object that will store new factors and
    # values to add to the incremental graph
    new_factors = gtsam.NonlinearFactorGraph()
    # values storing the initial estimates of new nodes in the factor graph
    new_values = gtsam.Values()

    # Main loop:
    # (1) we read the measurements
    # (2) we create the corresponding factors in the graph
    # (3) we solve the graph to obtain and optimal estimate of robot trajectory
    print("-- Starting main loop: inference is performed at each time step, "
          "but we plot trajectory every 10 steps")

    j = 0
    for i in range(first_camera_pose, len(camera_measurements)):
        # At each non=IMU measurement we initialize a new node in the graph
        current_pose_key = X(i)
        current_vel_key = V(i)
        current_param_key = P(i)
        current_landmark_key = L(i)

        if i == first_camera_pose:
            # Create initial estimate and prior on initial pose, velocity, and biases
            new_values.insert(current_pose_key, current_pose_global)
            new_values.insert(current_vel_key, current_velocity_global)
            new_values.insert(current_param_key, current_param)
            new_values.insert(current_landmark_key, current_landmark)

            new_factors.addPriorPoint2(current_pose_key, current_pose_global,
                                      sigma_init_x)
            new_factors.addPriorVector(current_vel_key,
                                       current_velocity_global, sigma_init_v)
            new_factors.addPriorConstantBias(current_param_key, current_param,
                                             sigma_init_b)
            new_factors.addPriorPoint2(current_landmark_key, current_landmark,
                                        sigma_init_x)
        else:

            # Create IMU factor
            previous_pose_key = X(i - 1)
            previous_vel_key = V(i - 1)
            previous_param_key = P(i - 1)
            previous_landmark_key = L(i - 1)

            new_factors.push_back(
                gtsam.ImuFactor(previous_pose_key, previous_vel_key,
                                current_pose_key, current_vel_key,
                                previous_bias_key,
                                current_summarized_measurement))

            # Bias evolution as given in the IMU metadata
            sigma_between_b = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray([
                    np.sqrt(included_imu_measurement_count) *
                    kitti_calibration.accelerometer_bias_sigma
                ] * 3 + [
                    np.sqrt(included_imu_measurement_count) *
                    kitti_calibration.gyroscope_bias_sigma
                ] * 3))

            new_factors.push_back(
                gtsam.BetweenFactorConstantBias(previous_bias_key,
                                                current_param_key,
                                                gtsam.imuBias.ConstantBias(),
                                                sigma_between_b))

            # Create GPS factor
            gps_pose = Pose3(current_pose_global.rotation(),
                             camera_measurements[i].position)
            if (i % camera_skip) == 0:
                new_factors.addPriorPose3(current_pose_key, gps_pose,
                                          noise_model_gps)
                new_values.insert(current_pose_key, gps_pose)

                print(f"############ POSE INCLUDED AT TIME {t} ############")
                print(gps_pose.translation(), "\n")
            else:
                new_values.insert(current_pose_key, current_pose_global)

            # Add initial values for velocity and bias based on the previous
            # estimates
            new_values.insert(current_vel_key, current_velocity_global)
            new_values.insert(current_param_key, current_bias)

            # Update solver
            # =======================================================================
            # We accumulate 2*GPSskip GPS measurements before updating the solver at
            # first so that the heading becomes observable.
            if i > (first_camera_pose + 2 * camera_skip):
                print(f"############ NEW FACTORS AT TIME {t:.6f} ############")
                new_factors.print()

                isam.update(new_factors, new_values)

                # Reset the newFactors and newValues list
                new_factors.resize(0)
                new_values.clear()

                # Extract the result/current estimates
                result = isam.calculateEstimate()

                current_pose_global = result.atPose3(current_pose_key)
                current_velocity_global = result.atVector(current_vel_key)
                current_bias = result.atConstantBias(current_param_key)

                print(f"############ POSE AT TIME {t} ############")
                current_pose_global.print()
                print("\n")

    return isam


def main():
    """Main runner."""

    # Configure different variables
    first_camera_pose = 1
    camera_skip = 10
    camera_measurements = loadCameraData("./logs/data_adaptive_2024-10-11-16:22.pkl")

    # Configure noise models
    noise_model_camera = noiseModel.Diagonal.Precisions(
        np.asarray([0, 0, 0] + [1.0 / 0.07] * 3))

    sigma_init_x = noiseModel.Diagonal.Precisions(
        np.asarray([0, 0]))
    sigma_init_v = noiseModel.Diagonal.Sigmas(np.ones(2) * 1000.0)
    sigma_init_b = noiseModel.Diagonal.Sigmas(
        np.asarray([0.1] * 3 + [5.00e-05] * 3))

    isam = optimize(camera_measurements, sigma_init_x,
                    sigma_init_v, sigma_init_b, noise_model_camera, first_camera_pose, camera_skip)



if __name__ == "__main__":
    main()