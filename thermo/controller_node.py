from scipy.spatial.transform import Rotation as R
from .ParameterEstimation import *
from thermo.thermo.trajectory_planner import Parameters, RunConfig
from .ROS import SpeedPublisher, ASTRFeedbackSubscriber, LoggingDataPublisher
from .T3pro import T3pro
from .models import hydrogelPhantom, humanTissue, SteadyStateMinimizationModel
from .utils import thermal_frame_to_color, cv_isotherm_width, find_tooltip, draw_info_on_frame
from copy import deepcopy
import rclpy
from dataclasses import dataclass
from enum import Enum
import json 
import click

class ControlMode(str, Enum):
    AUTONOMOUS = 'AUTONOMOUS'
    CONSTANT_VELOCITY = 'CONSTANT_VELOCITY'

@dataclass(frozen=True)
class RunConfig:
    control_mode: ControlMode
    adaptive_velocity: bool
    log_save_dir: str


def update_kf(model,
              material: MaterialProperties, 
              defl_adaptation: DeflectionAdaptation, 
              therm_adaptation: ThermalAdaptation,
              tip_mm: np.ndarray, 
              w_mm: float, 
              u0: float):
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
    material.k = therm_adaptation.lambda_therm
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


@click.command()
@click.argument('params_path', type=str)
def main(params_path: str):
    """
    Main function to run the experiment
    PARAMS_PATH: Path to the parameters file
    """

    rclpy.init()
    node = rclpy.create_node('thermo_controller_node')

    with open(params_path, 'r') as f:
        params = json.load(f)
        params = Parameters(**params)

    material = hydrogelPhantom
    model = SteadyStateMinimizationModel(qw=1, qd=1, r=1)

    run_conf = RunConfig(
        control_mode=ControlMode.CONSTANT_VELOCITY,
        adaptive_velocity=True,
        log_save_dir='./thermo_logs',
        material=material,
    )

    t3 = T3pro()
    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = lut[raw_frame]
    # thermal_frame_to_color(thermal_arr)
    tip_neutral_px = []

    # Initialize the ROS2 nodes
    astr_sub = ASTRFeedbackSubscriber(name="estimator")
    speed_pub = SpeedPublisher()
    log_pub = LoggingDataPublisher()

    ## Initialize the parameter adaptation
    material = deepcopy(run_conf.material)

    therm_adaptation = ThermalAdaptation(np.array([0, 40, material.Cp, material.rho * 1e9, material.k * 1e3]), labels=['w', 'P', 'Cp', 'rho', 'k'],
                                           material=material)
    defl_adaptation = DeflectionAdaptation(np.array([0, 0, 0, 0, 0, 0, 1]),
                                                 labels=['x', 'y', 'x_dot', 'y_dot', 'x_rest', 'y_rest', 'c_defl'],
                                                 px_per_mm=params.thermal_px_per_mm, frame_size_px=params.frame_size_px)
    defl_adaptation.init = False
    vstar = 7  # nominal velocity [mm/s]

    while True:
        ur_speed_mm_s = astr_sub.get_speed_m_s() * 1000
        ret, raw_frame = t3.read()
        if not ret:
            break
        _, lut = t3.info()
        thermal_arr = lut[raw_frame]
        # color_frame = thermal_frame_to_color(thermal_arr)   

        try:
            u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(params, model, material,
                                                                tip_neutral_px, defl_adaptation,
                                                                therm_adaptation, ur_speed_mm_s, thermal_arr,
                                                                vstar=vstar)
        except ValueError:
            print("Covariance error, ending.")
            break

        speed_pub.set_speed(u0)
        speed_pub.publish_speed()

        # new_tip_pos = defl_adaptation.kf.x[:2] * params.thermal_px_per_mm
        # neutral_tip_pos = defl_adaptation.neutral_tip_mm * params.thermal_px_per_mm
        # color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos, neutral_tip_pos)

        if defl_mm > params.deflection_max:
            node.get_logger().warn("Deflection too high, stopping.")
            break
        
        pose = astr_sub.pose
        log_pub.publish_data(pose=pose,
                             width_mm=w_mm,
                             cmd_speed_mm_s=u0,
                             meas_speed_mm_s=ur_speed_mm_s,
                             vstar_mm_s=vstar,
                             defl_mm=defl_mm,
                             c_defl=defl_adaptation.c_defl,
                             q=therm_adaptation.q,
                             cp=therm_adaptation.Cp,
                             lambda_thermal=therm_adaptation.lambda_therm,
                             rho=therm_adaptation.rho)
        log_pub.publish_logging_data()
        rclpy.spin_once(log_pub)
        rclpy.spin_once(astr_sub)
        rclpy.spin_once(speed_pub)
    
    astr_sub.destroy_node()
