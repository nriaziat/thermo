from rclpy.node import Node
from geometry_msgs.msg import Pose, Point32, PointStamped # for position and velocity of ASTR
from sensor_msgs.msg import Image, PointCloud  # for thermal image
from astr_msgs.msg import AstrCartesianCommand, AstrFeedback, AstrCartesianMotionMode
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_geometry_msgs import do_transform_pose, TransformStamped, do_transform_point
from .Trajectory import Trajectory
import numpy as np
import rclpy

class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__('thermo_pc_pub')
        self.publisher_ = self.create_publisher(PointCloud, '/traj_pc', 10)
        self.pc = PointCloud()
        self.pc.header.frame_id = 'electrocautery_arm_base_link'
        self.loop_rate = self.create_rate(100)

    def set_command(self, traj: Trajectory, tf: TransformStamped):
        for i in range(len(traj)):
            pt = PointStamped()
            pt.point.x = float(traj.poses[i].position.x)
            pt.point.y = float(traj.poses[i].position.y)
            pt.point.z = float(traj.poses[i].position.z)
            pt = do_transform_point(pt, tf)
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
        self.publisher_ = self.create_publisher(TransformStamped, '/target_tf', 10)
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


class AstrCartesianCommandPublisher(Node):
    def __init__(self):
        super().__init__('thermo_pub')
        self.publisher_ = self.create_publisher(AstrCartesianCommand, '/electrocautery_arm/target_pt', 10)
        self.command = AstrCartesianCommand()
        self.loop_rate = self.create_rate(100)

    def set_command(self, pose: Pose, velocity: float, tf: TransformStamped):
        """
        @param velocity: velocity in mm/s
        """
        self.command.target_pose = do_transform_pose(pose, tf)
        self.command.motion_mode.mode_enum = AstrCartesianMotionMode.CUSTOM
        self.command.motion_mode.requested_lin_vel_m_s = float(velocity * 1e-3)
        self.command.motion_mode.requested_ang_vel_deg_s = 100.

    def publish_command(self):
        self.publisher_.publish(self.command)

class FeedbackSubscriber(Node):
    def __init__(self):
        super().__init__('thermo_sub')
        self.subscription = self.create_subscription(AstrFeedback, '/electrocautery_arm/state_feedback', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.pose = Pose()
        self.twist = np.array([0, 0, 0])

    def listener_callback(self, msg: AstrFeedback):
        self.pose = msg.actual_cartesian_position
        self.twist = np.array([msg.actual_cartesian_velocity.linear.x, msg.actual_cartesian_velocity.linear.y, msg.actual_cartesian_velocity.linear.z])

class FrameListener(Node):

    def __init__(self):
        super().__init__('astr_tf')

        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
          'target_frame', 'electrocautery_arm_base_link').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.transform = None
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Call on_timer function every second
        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = self.target_frame
        to_frame_rel = 'world'

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

class ColorImagePublisher(Node):
    def __init__(self):
        super().__init__('color_image_publisher')
        self.publisher_ = self.create_publisher(Image, 'color_image', 10)
        self.image = None

    def set_image(self, image):
        self.image = image

    def publish_image(self):
        msg = Image()
        msg.data = self.image
        self.publisher_.publish(msg)

class ThermalImagePublisher(Node):
    def __init__(self):
        super().__init__('thermal_image_publisher')
        self.publisher_ = self.create_publisher(Image, 'thermal_image', 10)
        self.image = None

    def set_image(self, image):
        self.image = image

    def publish_image(self):
        msg = Image()
        msg.data = self.image
        self.publisher_.publish(msg)