import rclpy.logging
import rclpy.time
from .utils import *
from .ROS import ASTRCartesianCommandPublisher, ASTRFeedbackSubscriber, FrameListener, PointCloudPublisher, CommandTFPublisher, SpeedSubscriber
from .Trajectory import *
import warnings
from dataclasses import dataclass
import rclpy

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
        trajectory_path: str) -> None:
    """
    Main function to run the experiment
    :param params: Running parameters
    :param trajectory_path: Path to the trajectory
    """

    # TODO: just a workaround for some bad trajectory removing first and last point
    points = np.load(trajectory_path + '/points.npy')
    normals = np.load(trajectory_path + '/normals.npy')
    trajectory = Trajectory(points, normals, resample_distance=2e-3)

    #############################

    rclpy.init()
    node = rclpy.create_node('thermo_trajectory_node')
    astr_sub = ASTRFeedbackSubscriber(name="trajectory")
    astr_pub = ASTRCartesianCommandPublisher()
    speed_sub = SpeedSubscriber()
    tf_sub = FrameListener()
    pc_pub = PointCloudPublisher()
    tf_pub = CommandTFPublisher()
    node.get_logger().info('Waiting for transform')
    while tf_sub.transform is None:
        rclpy.spin_once(tf_sub)
    node.get_logger().info("Transform found.")
    pc_pub.set_command(trajectory, tf_sub.transform)
    pc_pub.publish_command()

    args = {
            'params': params,
            'astr_pub': astr_pub,
            'astr_sub': astr_sub,
            'pc_pub': pc_pub,
            'tf_pub': tf_pub,
            'speed_sub': speed_sub,
            'traj': trajectory,
            'arm_tf': tf_sub.transform,
            }

    loop(**args)
    
    node.get_logger().info('Experiment finished')
    astr_sub.destroy_node()
    astr_pub.destroy_node()
    rclpy.shutdown()

def loop(*,
         params: Parameters,
         astr_pub: ASTRCartesianCommandPublisher,
         astr_sub: ASTRFeedbackSubscriber,
         pc_pub: PointCloudPublisher,
         tf_pub: CommandTFPublisher,
         speed_sub: SpeedSubscriber,
         traj: Trajectory,
         arm_tf: TransformStamped) -> None:
    """
    Run the real experiment
    """

    u0 = (params.v_min + params.v_max) / 2
    
    for i, cmd_pose in enumerate(traj):
        if i == 1:
            input("Waiting at insertion point...")
        tf_pub.set_command(cmd_pose, arm_tf)  # this is for RVIZ showing the orientation of the requested pt 

        # TODO: Remove hardcode and replace with u0
        if i > 0 and i < len(traj) - 1:
            u0 = speed_sub.speed_mm_s
            if u0 < params.v_min:
                u0 = params.v_min
            elif u0 > params.v_max:
                u0 = params.v_max
        else:
            u0 = (params.v_min + params.v_max) / 2  ## retraction speed

        astr_pub.set_command(cmd_pose, u0, arm_tf)
        astr_pub.publish_command()  
        tf_pub.publish_command()            
        # rclpy.spin_once(astr_pub)
        # rclpy.spin_once()
        # rclpy.spin_once(pc_pub)
        # rclpy.spin_once(tf_pub)
        while position_error(cmd_pose, astr_sub.pose, arm_tf) > 0.001 or orientation_error(cmd_pose, astr_sub.pose, arm_tf) > np.deg2rad(10):
            rclpy.spin_once(astr_sub)
            rclpy.spin_once(speed_sub)


    # DO NOT SET VELOCITY TO 0 OR IDLE MODE
    astr_pub.set_command(cmd_pose, 3, arm_tf)
    astr_pub.command.motion_mode.should_stop_here = True
    astr_pub.publish_command()

