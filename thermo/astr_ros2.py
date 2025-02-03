import cv2
import rclpy.logging
import rclpy.time
from .T3pro import T3pro
from .ParameterEstimation import *
from .utils import *
from .DataLogger import DataLogger
from .ROS import *
import warnings
from dataclasses import dataclass
from copy import deepcopy
import pygame
from enum import Enum
from thermo.Trajectory import Trajectory
from scipy.spatial.transform import Rotation as R
import rclpy
from tf2_geometry_msgs import do_transform_pose

warnings.filterwarnings("ignore")

# Define constants
THERMAL_PX_PER_MM = 35./5.  # px/mm
V_MIN = 0  # minimum velocity [mm/s]
V_MAX = 15.0  # maximum velocity [mm/s]
T_DEATH = 60
Ta = 20
FRAME_SIZE_PX = (384, 288)
DEFLECTION_MAX = 10
TIME_STEP = 1 / 24

def position_error(pose1, pose2):
    """
    Compute the position error between two poses
    """
    return np.linalg.norm(np.array([pose1.position.x, pose1.position.y, pose1.position.z]) - np.array([pose2.position.x, pose2.position.y, pose2.position.z]))

class ControlMode(str, Enum):
    AUTONOMOUS = 'AUTONOMOUS'
    CONSTANT_VELOCITY = 'CONSTANT_VELOCITY'
    TELEOPERATED = 'TELEOPERATED'
    SHARED_CONTROL = 'SHARED_CONTROL'

@dataclass(frozen=True)
class Devices:
    t3: T3pro
    joystick: pygame.joystick.Joystick

@dataclass(frozen=True)
class Parameters:
    thermal_px_per_mm: float = THERMAL_PX_PER_MM
    v_min: float = V_MIN
    v_max: float = V_MAX
    t_death: float = T_DEATH
    t_amb: float = Ta
    frame_size_px: tuple = FRAME_SIZE_PX
    deflection_max: float = DEFLECTION_MAX
    time_step: float = TIME_STEP

@dataclass(frozen=True)
class RunConfig:
    control_mode: ControlMode
    adaptive_velocity: bool
    log_save_dir: str
    material: MaterialProperties

def update_kf(model, material: MaterialProperties, defl_adaptation: DeflectionAdaptation, therm_adaptation: ThermalAdaptation,
              tip_mm: np.ndarray, w_mm: float, u0: float):
    """
    Update the adaptive parameters
    :param model: Model object
    :param material: Material properties
    :param defl_adaptation: Deflection adaptation object
    :param therm_adaptation: Thermal adaptation object
    :param tip_mm: Tip position mm [mm]updated_kf
    :param w_mm: Width [mm]
    :param u0: Velocity [mm/s]
    """
    # deflection_adaptation.update(defl_mm, np.exp(-model.c_defl / u0))
    dT = model.t_death - model.Ta
    if defl_adaptation.init:
        if tip_mm is not None:
            defl_adaptation.update(np.array([tip_mm[0], tip_mm[1]]), v=u0)
        else:
            defl_adaptation.update(None, v=u0)
    therm_adaptation.update(np.array([w_mm]), dT=dT, v=u0)
    material.Cp = therm_adaptation.Cp
    material.k = therm_adaptation.k
    material.rho = therm_adaptation.rho

def update(params: Parameters,
           model,
           material: MaterialProperties,
           tip_candidates_px: list[np.ndarray],
           defl_adaptation: DeflectionAdaptation,
           therm_adaptation: ThermalAdaptation,
           u0: float, frame: np.ndarray,
           vstar: float):
    """
    Update velocity and adaptive parameters
    :param params: Parameters object
    :param model: Model object
    :param material: Material properties
    :param tip_candidates_px: Tool tip resting position
    :param defl_adaptation: Deflection adaptation object
    :param therm_adaptation: Thermal adaptation object
    :param u0: Current velocity
    :param vstar: Desired velocity
    :param frame: Thermal frame
    :return: Updated velocity, deflection, width, MPC initialization flag
    """
    tip_px = find_tooltip(frame, params.t_death + 10)
    if tip_px is not None:
        tip_px = np.array(tip_px)
        if len(tip_candidates_px) < 1:
            tip_candidates_px.append(tip_px)
        elif not defl_adaptation.init:
            tip_neutral_px = np.median(tip_candidates_px, axis=0)
            defl_adaptation.kf.x[4:6] = tip_neutral_px / params.thermal_px_per_mm
            defl_adaptation.kf.x[0:2] = tip_neutral_px / params.thermal_px_per_mm
            defl_adaptation.init = True

    tip_mm = np.array(tip_px) / params.thermal_px_per_mm if tip_px is not None else None
    w_px, _ = cv_isotherm_width(frame, model.t_death)
    w_mm = w_px / params.thermal_px_per_mm
    if u0 > 0:
        update_kf(model, material, defl_adaptation, therm_adaptation, tip_mm, w_mm, u0)
    else:
        if tip_mm is None:
            return u0, 0, w_mm, tip_candidates_px
        return u0, np.linalg.norm(tip_mm - defl_adaptation.neutral_tip_mm), w_mm, tip_candidates_px

    u0 = model.find_optimal_velocity(material, defl_adaptation.c_defl, therm_adaptation.q, vstar)
    return u0, defl_adaptation.defl_mm, therm_adaptation.w_mm, tip_candidates_px

def main(model,
         run_conf: RunConfig,
         devices: Devices,
         trajectory_path: str) -> None:
    """
    Main function to run the experiment
    :param model: Model object
    :param run_conf: Running parameters
    :param devices: Devices object
    """
    params = Parameters()

    points = np.load(trajectory_path + '/points.npy')
    normals = np.load(trajectory_path + '/normals.npy')
    trajectory = Trajectory(points, normals)

    ## Initialize the parameter adaptation
    material = deepcopy(run_conf.material)

    thermal_adaptation = ThermalAdaptation(np.array([0, 40, material.Cp, material.rho * 1e9, material.k * 1e3]), labels=['w', 'P', 'Cp', 'rho', 'k'],
                                           material=material)
    deflection_adaptation = DeflectionAdaptation(np.array([0, 0, 0, 0, 0, 0, 1]),
                                                 labels=['x', 'y', 'x_dot', 'y_dot', 'x_rest', 'y_rest', 'c_defl'],
                                                 px_per_mm=params.thermal_px_per_mm, frame_size_px=params.frame_size_px)
    deflection_adaptation.init = False

    #############################

    rclpy.init()
    node = rclpy.create_node('thermo')
    astr_sub = FeedbackSubscriber()
    astr_pub = AstrCartesianCommandPublisher()
    color_image_publisher = ColorImagePublisher()
    thermal_image_publisher = ThermalImagePublisher()
    tf_sub = FrameListener()
    pc_pub = PointCloudPublisher()
    tf_pub = CommandTFPublisher()
    pc_pub.set_command(trajectory)


    model.vmin = params.v_min
    model.vmax = params.v_max

    loop(run_conf, model, material, devices, params, deflection_adaptation, thermal_adaptation,
        astr_pub, astr_sub, tf_sub, pc_pub, tf_pub,
        color_image_publisher, thermal_image_publisher, trajectory)
    node.get_logger().info('Experiment finished')
    for device in devices.__dict__.values():
        if hasattr(device, 'close'):
            device.close()
    astr_sub.destroy_node()
    astr_pub.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

def loop(run_conf: RunConfig,
         model, material: MaterialProperties,
         devices,
         params: Parameters,
         defl_adaptation: DeflectionAdaptation,
         therm_adaptation: ThermalAdaptation,
         astr_pub: AstrCartesianCommandPublisher,
         astr_sub: FeedbackSubscriber,
         tf_sub: FrameListener,
         pc_pub: PointCloudPublisher,
         tf_pub: CommandTFPublisher,
         color_image_publisher: ColorImagePublisher,
         thermal_image_publisher: ThermalImagePublisher, 
         traj: Trajectory) -> None:
    """
    Run the real experiment
    """
    t3 = devices.t3
    joy = devices.joystick

    u0 = (params.v_min + params.v_max) / 2
    data_logger = DataLogger(run_conf.log_save_dir, run_conf.adaptive_velocity, None)
    
    ret = True
    if t3 is not None:
        ret, raw_frame = t3.read()
        info, lut = t3.info()
        thermal_arr = lut[raw_frame]
        thermal_frame_to_color(thermal_arr)
        tip_neutral_px = []
        
    vstar = 0
    i = 0
    while (tf_sub.transform is None):
        rclpy.spin_once(tf_sub)
    while ret and i < len(traj):
        cmd_pose = traj[i]
        tf_pub.set_command(cmd_pose, tf_sub.transform)
        if position_error(cmd_pose, astr_sub.pose) < 0.01:
            i += 1
            if i >= len(traj):
                break
        
        if t3 is not None:
            ret, raw_frame = t3.read()
            _, lut = t3.info()
            thermal_arr = lut[raw_frame]
            color_frame = thermal_frame_to_color(thermal_arr)

        if run_conf.control_mode is ControlMode.AUTONOMOUS:
            vstar = 7
        elif run_conf.control_mode is ControlMode.SHARED_CONTROL or run_conf.control_mode is ControlMode.TELEOPERATED:
            pygame.event.get()
            if joy.get_button(0):
                vstar += 2 * ((joy.get_axis(5) + 1) / 2 - (joy.get_axis(2) + 1) / 2) - 0.05 * vstar
                vstar = np.clip(vstar, params.v_min, params.v_max)
            else:
                vstar = 0
            if joy.get_button(1):
                break
        
        if t3 is not None:
            try:
                u0 = np.linalg.norm(astr_sub.twist)
                u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(params, model, material,
                                                                    tip_neutral_px, defl_adaptation,
                                                                    therm_adaptation, u0, thermal_arr,
                                                                    vstar=vstar)
            except ValueError:
                print("Covariance error, ending.")
                break

            new_tip_pos = defl_adaptation.kf.x[:2] * params.thermal_px_per_mm
            neutral_tip_pos = defl_adaptation.neutral_tip_mm * params.thermal_px_per_mm
            color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos, neutral_tip_pos)

            if defl_mm > params.deflection_max:
                break

        if vstar < 0.1:
            u0 = 0
        elif run_conf.control_mode is ControlMode.TELEOPERATED:
            u0 = vstar

        data_logger.log_data(cmd_pose.position, w_mm, u0, vstar, defl_mm, thermal_arr, defl_adaptation, therm_adaptation, None)

        astr_pub.set_command(cmd_pose, u0, tf_sub.transform)
        astr_pub.publish_command()
        pc_pub.publish_command()
        tf_pub.publish_command()

        if t3 is not None:
            color_image_publisher.set_image(color_frame)
            color_image_publisher.publish_image()
            thermal_image_publisher.set_image(thermal_arr)
            thermal_image_publisher.publish_image()
            rclpy.spin_once(color_image_publisher)
            rclpy.spin_once(thermal_image_publisher)

        rclpy.spin_once(astr_pub)
        astr_pub.loop_rate.sleep()
        rclpy.spin_once(astr_sub)
        rclpy.spin_once(tf_sub)
        rclpy.spin_once(pc_pub)
        rclpy.spin_once(tf_pub)

    astr_pub.set_command(cmd_pose.position, cmd_pose.orientation, 0)
    astr_pub.command.motion_mode.should_stop_here = True
    astr_pub.publish_command()
    data_logger.save_log()

