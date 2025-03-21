from .ROS import (
    ASTRCartesianCommandPublisher,
    ASTRFeedbackSubscriber,
    FrameListener,
    PointCloudPublisher,
    CommandTFPublisher,
    SpeedSubscriber,
)
from .Trajectory import position_error, orientation_error, Trajectory, TransformStamped
import warnings
from dataclasses import dataclass
import rclpy
import numpy as np

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Parameters:
    thermal_px_per_mm: float
    v_min: float
    v_max: float
    t_death: float
    t_amb: float
    frame_size_px: tuple
    deflection_max: float
    time_step: float


def main(
    params: Parameters,
    trajectory_path: str,
    smooth_traj: bool,
    interpolation_factor: int,
    cut_depth_m: float,
) -> None:
    """
    Main function to run the experiment
    :param params: Running parameters
    :param trajectory_path: Path to the trajectory
    :param reverse_traj: Flag to run the trajectory in reverse.        \
    :param smooth_trajectory: run exponential smoothing on trajectory
    :param interpolation_factor: how many points to interpolate between given points.

    """

    # TODO: just a workaround for some bad trajectory removing first and last point
    points = np.load(trajectory_path + "/points.npy")
    normals = np.load(trajectory_path + "/normals.npy")
    trajectory_world = Trajectory(
        points,
        normals,
        resample_distance=2e-3,
        reverse=False,
        smooth_trajectory=smooth_traj,
        interpolation_factor=interpolation_factor,
        retraction_position=0,
        average_normals=True,
        cut_depth_m=cut_depth_m,
    )

    #############################

    rclpy.init()
    node = rclpy.create_node("thermo_trajectory_node")
    astr_sub = ASTRFeedbackSubscriber(name="trajectory")
    astr_pub = ASTRCartesianCommandPublisher()
    speed_sub = SpeedSubscriber()
    tf_sub = FrameListener()
    pc_pub = PointCloudPublisher()
    tf_pub = CommandTFPublisher()
    node.get_logger().info("Waiting for transform")

    while tf_sub.transform is None:
        rclpy.spin_once(tf_sub)

    rclpy.spin_once(astr_sub)
    while astr_sub.pose is None:
        rclpy.spin_once(astr_sub)

    starting_pose = astr_sub.pose
    start_idx = trajectory_world.find_closest_orientation(starting_pose)
    trajectory_world = Trajectory(
        points,
        normals,
        resample_distance=2e-3,
        reverse=False,
        smooth_trajectory=smooth_traj,
        interpolation_factor=interpolation_factor,
        retraction_position=start_idx,
        average_normals=True,
        cut_depth_m=cut_depth_m,
    )

    node.get_logger().info("Transform found.")
    pc_pub.set_command(trajectory_world)
    pc_pub.publish_command()

    args = {
        "node": node,
        "params": params,
        "astr_pub": astr_pub,
        "astr_sub": astr_sub,
        "tf_pub": tf_pub,
        "speed_sub": speed_sub,
        "traj": trajectory_world,
        "arm_tf": tf_sub.transform,
    }

    loop(**args)

    node.get_logger().info("Experiment finished")
    astr_sub.destroy_node()
    astr_pub.destroy_node()
    rclpy.shutdown()


def loop(
    *,
    node: rclpy.node.Node,
    params: Parameters,
    astr_pub: ASTRCartesianCommandPublisher,
    astr_sub: ASTRFeedbackSubscriber,
    tf_pub: CommandTFPublisher,
    speed_sub: SpeedSubscriber,
    traj: Trajectory,
    arm_tf: TransformStamped,
) -> None:
    """
    Run the real experiment
    """

    u0 = (params.v_min + params.v_max) / 2
    entry_exit_speed = 5

    entry_pose_world = traj[0]
    tf_pub.set_command(
        entry_pose_world, arm_tf
    )  # this is for RVIZ showing the orientation of the requested pt
    astr_pub.set_command(
        entry_pose_world, entry_exit_speed, arm_tf, should_stop_here=True
    )
    tf_pub.publish_command()
    astr_pub.publish_command()
    while position_error(
        entry_pose_world, astr_sub.pose, arm_tf
    ) > 0.001 or orientation_error(
        entry_pose_world, astr_sub.pose, arm_tf
    ) > np.deg2rad(
        10
    ):
        rclpy.spin_once(astr_sub)
        rclpy.spin_once(speed_sub)
        wrist_angle = astr_sub.joints[-1]

    wrist_angle = astr_sub.joints[-1]
    if (traj.reversed and wrist_angle > 0) or (not traj.reversed and wrist_angle < 0):
        traj.reverse_trajectory()
        # traj.change_starting_point((traj.retrajection_position + len(traj)) % len(traj))
        entry_pose_world = traj[0]
        tf_pub.set_command(
            entry_pose_world, arm_tf
        )  # this is for RVIZ showing the orientation of the requested pt
        astr_pub.set_command(
            entry_pose_world, entry_exit_speed, arm_tf, should_stop_here=True
        )
        tf_pub.publish_command()
        astr_pub.publish_command()
        while position_error(
            entry_pose_world, astr_sub.pose, arm_tf
        ) > 0.001 or orientation_error(
            entry_pose_world, astr_sub.pose, arm_tf
        ) > np.deg2rad(
            10
        ):
            rclpy.spin_once(astr_sub)
            rclpy.spin_once(speed_sub)
        rclpy.spin_once(astr_pub)
        rclpy.spin_once(tf_pub)
    node.get_logger().info(
        f"Running {'clockwise' if traj.reversed else 'counter-clockwise'}"
    )
    node.get_logger().info("Waiting at insertion point...")
    input()

    for i, cmd_pose_world in enumerate(traj[1:-1]):
        wrist_angle = astr_sub.joints[-1]
        if not -2 * np.pi < wrist_angle < 2 * np.pi:
            node.get_logger().error(f"Wrist Angle Out of Limits: {wrist_angle:.3f}")

        tf_pub.set_command(
            cmd_pose_world, arm_tf
        )  # this is for RVIZ showing the orientation of the requested pt

        # TODO: Remove hardcode and replace with u0
        if i > 0 and i < len(traj) - 1:
            u0 = speed_sub.speed_mm_s
            if u0 < params.v_min:
                u0 = params.v_min
            elif u0 > params.v_max:
                u0 = params.v_max
        else:
            u0 = (params.v_min + params.v_max) / 2  ## retraction speed

        if i == 0:
            astr_pub.set_command(cmd_pose_world, u0, arm_tf, should_stop_here=True)
        else:
            astr_pub.set_command(cmd_pose_world, u0, arm_tf)

        astr_pub.publish_command()
        tf_pub.publish_command()
        thresh = 0.0015
        while position_error(
            cmd_pose_world, astr_sub.pose, arm_tf
        ) > thresh or orientation_error(
            cmd_pose_world, astr_sub.pose, arm_tf
        ) > np.deg2rad(
            15
        ):
            rclpy.spin_once(astr_sub)
            rclpy.spin_once(speed_sub)
            # astr_pub.set_command(cmd_pose_world, speed_sub.speed_mm_s, arm_tf)
            # astr_pub.publish_command()

    exit_pose_world = traj[-1]
    tf_pub.set_command(
        exit_pose_world, arm_tf
    )  # this is for RVIZ showing the orientation of the requested pt
    astr_pub.set_command(exit_pose_world, entry_exit_speed, arm_tf)
    tf_pub.publish_command()
    astr_pub.publish_command()
    while position_error(
        exit_pose_world, astr_sub.pose, arm_tf
    ) > 0.001 or orientation_error(exit_pose_world, astr_sub.pose, arm_tf) > np.deg2rad(
        10
    ):
        rclpy.spin_once(astr_sub)
        rclpy.spin_once(speed_sub)

    # DO NOT SET VELOCITY TO 0 OR IDLE MODE
    astr_pub.set_command(
        exit_pose_world, entry_exit_speed, arm_tf, should_stop_here=True
    )
    node.get_logger().info("Trajectory planner shutting down.")
    astr_pub.publish_command()