from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension, Int16
from geometry_msgs.msg import Pose, Point32, PointStamped # for position and velocity of ASTR
from sensor_msgs.msg import PointCloud  # for trajectory visualization
from astr_msgs.msg import AstrCartesianCommand, AstrFeedback, AstrCartesianMotionMode
from thermo_msgs.msg import LoggingData
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_geometry_msgs import do_transform_pose, TransformStamped, do_transform_point
from .Trajectory import Trajectory
import numpy as np
import rclpy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class LogReplayer(Node):
    def __init__(self):
        super().__init__('logging_data_player')
        self.publisher_ = self.create_subscription(LoggingData, '/thermo/logging_data', self.replay_callback, 10)
        self.therm_arr = None
        self.robot_pose = Pose()
        self.meas_speed_mm_s = 0
        self.data_ready = False

    def replay_callback(self, msg: LoggingData):
        self.data_ready = True
        self.therm_arr = np.array(msg.thermal_arr.data).reshape((msg.thermal_arr.layout.dim[0].size, msg.thermal_arr.layout.dim[1].size))
        self.robot_pose = msg.pose
        self.meas_speed_mm_s = msg.meas_speed_mm_s

    def get_data(self):
        self.data_ready = False
        return self.therm_arr, self.robot_pose, self.meas_speed_mm_s

class DirectionPublisher(Node):
    def __init__(self):
        super().__init__('direction_publisher')
        self.publisher_ = self.create_publisher(Int16, '/thermo/direction', 10)
        self.direction = None  # 1 for ccw, -1 for cw or True for ccw, False for cw

    def set_direction(self, direction: int):
        self.direction = direction

    def publish_direction(self):
        msg = Int16()
        msg.data = self.direction
        self.publisher_.publish(msg)

class DirectionSubscriber(Node):
    def __init__(self):
        super().__init__('direction_subscriber')
        self.subscription = self.create_subscription(Int16, '/thermo/direction', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.direction = None

    def listener_callback(self, msg: Int16):
        self.direction = msg.data  # 1 for ccw, -1 for cw, 0 for stop

class LoggingDataPublisher(Node):
    def __init__(self):
        super().__init__('logging_data_publisher')
        self.publisher_ = self.create_publisher(LoggingData, '/thermo/logging_data', 10)
        self.logging_data: LoggingData = LoggingData()

    def set_logging_data(self, *,
                         pose: Pose, 
                         width_mm: float,
                         cmd_speed_mm_s: float,
                         meas_speed_mm_s: float,
                         vstar_mm_s: float,
                         defl_mm: float,
                         c_defl: float,
                         q: float,
                         cp: float,
                         lambda_thermal: float,
                         rho: float, 
                         thermal_arr: np.array):
        
        multiarr = Float32MultiArray()
        multiarr.data = thermal_arr.flatten().tolist()
        multiarr.layout.dim.append(MultiArrayDimension())    
        multiarr.layout.dim.append(MultiArrayDimension())
        multiarr.layout.dim[0].label  = "height"
        multiarr.layout.dim[0].size   = thermal_arr.shape[0]
        multiarr.layout.dim[0].stride = thermal_arr.size
        multiarr.layout.dim[1].label  = "width"
        multiarr.layout.dim[1].size   = thermal_arr.shape[1]
        multiarr.layout.dim[1].stride = thermal_arr.shape[1]

        self.logging_data.header.stamp = self.get_clock().now().to_msg()
        self.logging_data.pose = pose
        self.logging_data.width_mm = float(width_mm)
        self.logging_data.cmd_speed_mm_s = float(cmd_speed_mm_s)
        self.logging_data.meas_speed_mm_s = float(meas_speed_mm_s)
        self.logging_data.vstar_mm_s = float(vstar_mm_s)
        self.logging_data.deflection_mm = float(defl_mm)
        self.logging_data.c_defl = float(c_defl)
        self.logging_data.q = float(q)
        self.logging_data.cp = float(cp)
        self.logging_data.lambda_thermal = float(lambda_thermal)
        self.logging_data.rho = float(rho)
        self.logging_data.thermal_arr = multiarr

    def publish_logging_data(self):
        self.publisher_.publish(self.logging_data)

class ParameterEstiamtePublisher(Node):
    def __init__(self):
        super().__init__('parameter_estimate_publisher')
        self.publisher_ = self.create_publisher(Float32, '/thermo/parameter_estimate', 10)
        self.parameter_estimate = 0

    def set_parameter_estimate(self, estimate: float):
        self.parameter_estimate = estimate

    def publish_parameter_estimate(self):
        msg = Float32()
        msg.data = self.parameter_estimate
        self.publisher_.publish(msg)

class SpeedPublisher(Node):
    def __init__(self):
        super().__init__('speed_publisher')
        self.publisher_ = self.create_publisher(Float32, '/thermo/speed', 10)
        self.speed_mm_s = 0
    
    def set_speed(self, speed: float):
        self.speed_mm_s = speed

    def publish_speed(self):
        msg = Float32()
        msg.data = self.speed_mm_s / 1000  # Convert mm/s to m/s
        self.publisher_.publish(msg)

class SpeedSubscriber(Node):
    def __init__(self):
        super().__init__('speed_subscriber')
        self.subscription = self.create_subscription(Float32, '/thermo/speed', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.speed_mm_s = 0

    def listener_callback(self, msg: Float32):
        self.speed_mm_s = msg.data * 1000  # Convert m/s to mm/s

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('thermo_pc_pub')
        self.publisher_ = self.create_publisher(PointCloud, '/thermo/traj_pc', 10)
        self.pc = PointCloud()
        self.pc.header.frame_id = 'world'
        self.loop_rate = self.create_rate(100)

    def set_command(self, traj: Trajectory):
        for i in range(len(traj)):
            pt = PointStamped()
            pt.point.x = float(traj.poses[i].position.x)
            pt.point.y = float(traj.poses[i].position.y)
            pt.point.z = float(traj.poses[i].position.z)
            add_pt = Point32()
            add_pt.x = pt.point.x
            add_pt.y = pt.point.y
            add_pt.z = pt.point.z
            self.pc.points.append(add_pt)

    def publish_command(self):
        self.publisher_.publish(self.pc)

class CommandTFPublisher(Node):
    def __init__(self):
        super().__init__('thermo_tf_pub')
        self.name = self.declare_parameter(
          'framename', 'target').get_parameter_value().string_value
        self.publisher_ = self.create_publisher(TransformStamped, '/thermo/target_tf', 10)
        self.tf = TransformStamped()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf.header.frame_id = 'electrocautery_arm_base_link'
        self.tf.child_frame_id = self.name
        self.loop_rate = self.create_rate(100)

    def set_command(self, pose: Pose, tf: TransformStamped):
        self.tf.header.stamp = self.get_clock().now().to_msg()
        world_pose = do_transform_pose(pose, tf)
        self.tf.transform.translation.x = world_pose.position.x
        self.tf.transform.translation.y = world_pose.position.y
        self.tf.transform.translation.z = world_pose.position.z
        self.tf.transform.rotation = world_pose.orientation

    def publish_command(self):
        self.tf_broadcaster.sendTransform(self.tf)


class ASTRCartesianCommandPublisher(Node):
    def __init__(self):
        super().__init__('thermo_pub')
        self.publisher_ = self.create_publisher(AstrCartesianCommand, '/electrocautery_arm/target_pt', 10)
        self.command = AstrCartesianCommand()
        self.loop_rate = self.create_rate(100)

    def set_command(self, pose: Pose, velocity: float, tf: TransformStamped, should_stop_here: bool = False):
        """
        @param velocity: velocity in mm/s
        """
        self.command.target_pose = do_transform_pose(pose, tf)
        if velocity > 1e-3:
            self.command.motion_mode.mode_enum = AstrCartesianMotionMode.CUSTOM
        else:
            self.command.motion_mode.mode_enum = AstrCartesianMotionMode.IDLE
        self.command.motion_mode.requested_lin_vel_m_s = float(velocity * 1e-3)
        self.command.motion_mode.requested_ang_vel_deg_s = 60.
        self.command.motion_mode.should_stop_here = should_stop_here

    def publish_command(self):
        self.publisher_.publish(self.command)

class ASTRFeedbackSubscriber(Node):
    def __init__(self, name: str):
        super().__init__(f"astr_feedback_subscriber_{name}")
        self.subscription = self.create_subscription(AstrFeedback, '/electrocautery_arm/state_feedback', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.pose = None
        self.twist = np.array([0, 0, 0])
        self.dt = 1/25
        self.kf_init = False
        self.kf = KalmanFilter(dim_x=6, dim_z=3)  # constant acceleration model
        self.kf.x = np.array([0, 0, 0, 0, 0, 0])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0], 
                              [0, 0, 1, 0, 0, 0], 
                              [0, 0, 0, 0, 1, 0]]) 
        self.kf.F = np.array([[1, self.dt, 0, 0, 0, 0], 
                              [0, 1, 0, 0, 0, 0],
                              [0, 0, 1, self.dt, 0, 0], 
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, self.dt],
                              [0, 0, 0, 0, 0, 1]])
        self.kf.P  = np.diag([0.01, 9, 0.01, 9, 0.01, 9])
        self.kf.R  = np.diag([0.01**2, 0.01**2, 0.01**2])
        self.kf.Q  = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.2**2, block_size=3)
        self.joints = [] * 6

    def listener_callback(self, msg: AstrFeedback):
        self.pose = msg.actual_cartesian_position
        z = np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z])
        if not self.kf_init:
            self.kf.x = np.array([z[0],0, z[1], 0, z[2], 0])
            self.kf_init = True
        else:
            self.kf.predict()
            self.kf.update(1000 * z)
        self.joints = msg.actual_joint_position

    def get_speed_m_s(self):
        return np.linalg.norm([self.kf.x[1], self.kf.x[3], self.kf.x[5]]) / 1000
        # return np.linalg.norm(self.twist)

class FrameListener(Node):

    def __init__(self):
        super().__init__('astr_tf')

        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
          'target_frame', 'world').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.transform = None
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Call on_timer function every 0.1 second
        self.timer = self.create_timer(0.1, self.on_timer)

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = self.target_frame
        to_frame_rel = 'electrocautery_arm_base_link'

        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
        
        self.transform = t