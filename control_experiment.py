import cv2
import do_mpc.controller
from helper_code.aruco_tracker import ArucoTracker
from T3pro import T3pro
from testbed import Testbed, TestbedState
from ParameterEstimation import *
from utils import *
from Plotters import *
from DataLogger import DataLogger
import warnings
from typing import Optional
from dataclasses import dataclass
import tqdm
from tkinter import messagebox
from datetime import datetime
from copy import deepcopy
import pygame
from enum import Enum, auto, StrEnum

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
N_STEPS = 1000

# ExperimentType = StrEnum('ExperimentType', 'REAL PRERECORDED SIMULATED')
class ExperimentType(StrEnum):
    REAL = 'REAL'
    PRERECORDED = "PRERECORDED"
    SIMULATED = "SIMULATED"

# ControlMode = StrEnum('ControlMode', 'AUTONOMOUS CONSTANT_VELOCITY TELEOPERATED SHARED_CONTROL')
class ControlMode(StrEnum):
    AUTONOMOUS = 'AUTONOMOUS'
    CONSTANT_VELOCITY = 'CONSTANT_VELOCITY'
    TELEOPERATED = 'TELEOPERATED'
    SHARED_CONTROL = 'SHARED_CONTROL'

@dataclass(frozen=True)
class Devices:
    t3: T3pro
    testbed: Testbed
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
    n_steps: int = N_STEPS

@dataclass(frozen=True)
class RunConfig:
    exp_type: ExperimentType
    control_mode: ControlMode
    adaptive_velocity: bool
    constant_velocity: float
    log_save_dir: str
    log_file_to_load: str
    mpc: bool
    home: bool
    plot_adaptive_params: bool
    material: MaterialProperties

@dataclass
class PlotConfig:
    run_conf: RunConfig
    plotter: GenericPlotter
    w_mm: float
    defl_mm: float
    u0: float
    deflection_plotter: Optional[AdaptiveParameterPlotter]
    thermal_plotter: Optional[AdaptiveParameterPlotter]
    defl_hist_mm: Optional[list]


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

def update(run_conf: RunConfig,
           params: Parameters,
           model,
           material: MaterialProperties,
           tip_candidates_px: list[np.ndarray],
           defl_adaptation: DeflectionAdaptation,
           therm_adaptation: ThermalAdaptation,
           u0: float, frame: np.ndarray,
           vstar: float,
           init_mpc: bool,
           mpc: Optional[do_mpc.controller.MPC]):
    """
    Update velocity and adaptive parameters
    :param run_conf: Running parameters
    :param params: Parameters object
    :param model: Model object
    :param material: Material properties
    :param tip_candidates_px: Tool tip resting position
    :param defl_adaptation: Deflection adaptation object
    :param therm_adaptation: Thermal adaptation object
    :param u0: Current velocity
    :param vstar: Desired velocity
    :param frame: Thermal frame
    :param init_mpc: MPC initialization flag
    :param mpc: Optional MPC object
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
    tip_lead_distance = 0
    w_px, ellipse = cv_isotherm_width(frame, model.t_death)
    w_mm = w_px / params.thermal_px_per_mm
    if u0 > 0:
        update_kf(model, material, defl_adaptation, therm_adaptation, tip_mm, w_mm, u0)
    else:
        if tip_mm is None:
            return u0, 0, w_mm, init_mpc, tip_candidates_px
        return u0, np.linalg.norm(tip_mm - defl_adaptation.neutral_tip_mm), w_mm, init_mpc, tip_candidates_px
    if not init_mpc and run_conf.mpc:
        mpc.x0['width_0'] = w_mm
        mpc.x0['tip_lead_dist'] = tip_lead_distance
        mpc.set_initial_guess()
        init_mpc = True
    if run_conf.mpc:
        u0 = mpc.make_step(np.array([w_mm, tip_lead_distance])).item()
    else:
        u0 = model.find_optimal_velocity(material, defl_adaptation.c_defl, therm_adaptation.q, vstar)
    return u0, defl_adaptation.defl_mm, therm_adaptation.w_mm, init_mpc, tip_candidates_px

def setup_mpc(model, mpc, defl_adapt, therm_adapt, r, params: Parameters):
    mpc.set_objective(mterm=model.aux['mterm'], lterm=model.aux['lterm'])
    mpc.set_rterm(u=r)
    mpc.bounds['lower', '_u', 'u'] = params.v_min
    mpc.bounds['upper', '_u', 'u'] = params.v_max
    for i in range(model.n_isotherms):
        mpc.bounds['lower', '_x', f'width_{i}'] = 0
    mpc.bounds['lower', '_z', 'deflection'] = 0

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        tvp_template['_tvp'] = np.array([model.deflection_mm, defl_adapt.b, therm_adapt.b])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.u0 = (params.v_max + params.v_min) / 2
    mpc.setup()

    return MPCPlotter(mpc.data, isotherm_temps=model.isotherm_temps)

def main(model,
         run_conf: RunConfig,
         devices: Devices,
         mpc: Optional[do_mpc.controller.MPC] = None) -> None:
    """
    Main function to run the experiment
    :param model: Model object
    :param run_conf: Running parameters
    :param devices: Devices object
    :param mpc: Optional MPC object
    """
    params = Parameters()
    if run_conf.mpc:
        assert mpc is not None, "MPC object must be provided if MPC is enabled"

    ## Initialize the parameter adaptation
    material = deepcopy(run_conf.material)

    thermal_adaptation = ThermalAdaptation(np.array([0, 40, material.Cp, material.rho * 1e9, material.k * 1e3]), labels=['w', 'P', 'Cp', 'rho', 'k'],
                                           material=material)
    deflection_adaptation = DeflectionAdaptation(np.array([0, 0, 0, 0, 0, 0, 1]),
                                                 labels=['x', 'y', 'x_dot', 'y_dot', 'x_rest', 'y_rest', 'c_defl'],
                                                 px_per_mm=params.thermal_px_per_mm, frame_size_px=params.frame_size_px)
    deflection_adaptation.init = False

    #############################

    if run_conf.mpc:
        plotter = setup_mpc(model, mpc, deflection_adaptation, thermal_adaptation, 0.1, params)
    else:
        plotter = GenericPlotter(n_plots=3,
                                 labels=['Widths', 'Velocities', 'Deflections'],
                                 x_label='Time [s]',
                                 y_labels=[r'$\text{Widths [mm]}$',
                                           r'$\text{Velocities [mm/s]}$',
                                           r'$\delta \text{[mm]}$'])
    if run_conf.plot_adaptive_params:
        deflection_plotter = AdaptiveParameterPlotter(deflection_adaptation, plot_indices=[0, 0, 0, 0, 0, 0, 1])
        thermal_plotter = AdaptiveParameterPlotter(thermal_adaptation, plot_indices=[0, 1, 1, 1, 1])
    else:
        deflection_plotter = None
        thermal_plotter = None

    model.vmin = params.v_min
    model.vmax = params.v_max

    plotconfig = PlotConfig(run_conf, plotter, 0, 0, 0, deflection_plotter, thermal_plotter, None)

    if run_conf.exp_type is ExperimentType.PRERECORDED:
        prerecorded_experiment(run_conf, plotconfig, model, material, mpc, params, deflection_adaptation, thermal_adaptation)

    elif run_conf.exp_type is ExperimentType.SIMULATED:
        simulated_experiment(run_conf, plotconfig, model, material, mpc, params, deflection_adaptation, thermal_adaptation)

    elif run_conf.exp_type is ExperimentType.REAL:
        real_experiment(run_conf, plotconfig, model, material, devices, mpc, params, deflection_adaptation, thermal_adaptation)

    plt.show()
    for device in devices.__dict__.values():
        if hasattr(device, 'close'):
            device.close()
    cv2.destroyAllWindows()


def plot(plotconfig: PlotConfig) -> bool:
    """
    :param plotconfig: Plot configuration
    """
    if plotconfig.run_conf.mpc:
        plotconfig.plotter.plot()
    else:
        plotconfig.plotter.plot([plotconfig.w_mm,
                                 plotconfig.u0, plotconfig.defl_mm])
    if plotconfig.deflection_plotter is not None:
        plotconfig.deflection_plotter.plot()
    if plotconfig.thermal_plotter is not None:
        plotconfig.thermal_plotter.plot()
    fignums = [p.fig.number for p in [plotconfig.plotter, plotconfig.deflection_plotter, plotconfig.thermal_plotter] if p is not None]
    if not all([plt.fignum_exists(f) for f in fignums]):
        return False
    return True


def real_experiment(run_conf: RunConfig, pltconfig: PlotConfig,
                    model, material: MaterialProperties, devices, mpc,
                    params: Parameters,
                    defl_adaptation: DeflectionAdaptation,
                    therm_adaptation: ThermalAdaptation):
    """
    Run the real experiment
    """
    tb = devices.testbed
    t3 = devices.t3
    joy = devices.joystick

    u0 = (params.v_min + params.v_max) / 2
    data_logger = DataLogger(run_conf.log_save_dir, run_conf.adaptive_velocity, run_conf.constant_velocity)
    aruco_cam = cv2.VideoCapture(0)
    aruco_tracker = ArucoTracker()
    aruco_tag_width_mm = 8

    if run_conf.home:
        tb.home()

    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = lut[raw_frame]
    thermal_frame_to_color(thermal_arr)
    init_mpc = False

    start_quit = messagebox.askyesno("Start Experiment?", message="Press yes to continue, or no to cancel.")
    if not start_quit:
        return
    tip_neutral_px = []
    time0 = datetime.now()
    vstar = 0
    while ret:
        ret, aruco_frame = aruco_cam.read()
        aruco_pos, ids = aruco_tracker.detect(aruco_frame)
        if ids is not None:
            aruco_tag_width_px = np.linalg.norm(aruco_pos[0][0][0] - aruco_pos[0][0][1])
            aruco_position = (aruco_pos[0][0][0] * aruco_tag_width_mm / aruco_tag_width_px)
        else:
            aruco_position = None
        ret, raw_frame = t3.read()
        info, lut = t3.info()
        pos = tb.pos
        thermal_arr = lut[raw_frame]
        color_frame = thermal_frame_to_color(thermal_arr)
        if run_conf.control_mode is ControlMode.AUTONOMOUS:
            vstar = 7
        elif run_conf.control_mode is ControlMode.CONSTANT_VELOCITY:
            vstar = run_conf.constant_velocity
        elif run_conf.control_mode is ControlMode.SHARED_CONTROL or run_conf.control_mode is ControlMode.TELEOPERATED:
            pygame.event.get()
            if joy.get_button(0):
                vstar += 2 * ((joy.get_axis(5) + 1) / 2 - (joy.get_axis(2) + 1) / 2) - 0.05 * vstar
                vstar = np.clip(vstar, params.v_min, params.v_max)
            else:
                vstar = 0
            if joy.get_button(1):
                break

        try:
            u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(run_conf, params, model, material,
                                                                 tip_neutral_px, defl_adaptation,
                                                                 therm_adaptation, u0, thermal_arr,
                                                                 init_mpc=init_mpc, mpc=mpc, vstar=vstar)
        except ValueError:
            print("Covariance error, ending.")
            break

        new_tip_pos = defl_adaptation.kf.x[:2] * params.thermal_px_per_mm
        neutral_tip_pos = defl_adaptation.neutral_tip_mm * params.thermal_px_per_mm
        color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos, neutral_tip_pos)
        pltconfig.w_mm = w_mm
        pltconfig.defl_mm = defl_mm
        if len(data_logger.data_log) > 0:
            if data_logger.data_log["aruco_pos_mm"][0] is not None:
                defl_aruco = np.linalg.norm(aruco_position - data_logger.data_log['aruco_pos_mm'][0]) if aruco_position is not None else 0
            else:
                defl_aruco = 0
        else:
            defl_aruco = 0
        if defl_aruco > params.deflection_max:
            messagebox.showerror("Error", "Plastic Deformation, stopping the experiment")
            break
        # elif defl_mm > 0.8 * params.deflection_max:
        #     u0 = u0 - 2 * (defl_mm - (0.8 * params.deflection_max / 2))
        #     u0 = np.clip(u0, -params.v_max, params.v_max)


        cv.imshow("Frame", color_frame)

        if vstar < 0.1:
            u0 = 0
        elif run_conf.control_mode is ControlMode.CONSTANT_VELOCITY:
            u0 = run_conf.constant_velocity
        elif run_conf.control_mode is ControlMode.TELEOPERATED:
            u0 = vstar

        data_logger.log_data(pos, w_mm, u0, vstar, defl_mm, thermal_arr, defl_adaptation, therm_adaptation,
                             aruco_position)

        pltconfig.u0 = u0
        if not plot(pltconfig):
            break

        ret = tb.set_speed(u0)
        if ret is TestbedState.HOMED or ret is TestbedState.RIGHT:
            break
        dt = (datetime.now() - time0).total_seconds()
        if cv.waitKey(1) & 0xFF == ord('q') or pos == -1:
            break

    tb.set_speed(0)
    # update deflections with final neutral tip position
    print(f"Total time: {dt:.2f} seconds")
    print(f"7mm/s equivalent time: {pos / 7:.2f} seconds")
    data_logger.save_log()

def simulated_experiment(run_conf: RunConfig, pltconfig: PlotConfig,
                         model, material, mpc,
                         params: Parameters,
                         deflection_adaptation: DeflectionAdaptation,
                         thermal_adaptation: ThermalAdaptation):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=params.time_step, integration_tool='idas')
    sim_tvp = simulator.get_tvp_template()

    def sim_tvp_fun(t_now):
        sim_tvp['d'] = 0.5
        sim_tvp['defl_meas'] = model.deflection_mm
        sim_tvp['P'] = 20
        return sim_tvp

    simulator.set_tvp_fun(sim_tvp_fun)
    simulator.setup()
    x0 = np.linspace(2, 10, model.n_isotherms)
    simulator.x0 = x0
    mpc.x0 = x0
    simulator.set_initial_guess()
    mpc.set_initial_guess()
    for _ in tqdm.trange(params.n_steps):
        u0 = mpc.make_step(x0)
        defl_mm = deflection_adaptation.b * np.exp(-model._c_defl / u0.item()) * np.random.normal(1, 0.05)
        if defl_mm < 0:
            defl_mm = 0
        deflection_adaptation.update(defl_mm, u0.item())
        u_prime = (2 * material.alpha / u0) * np.sqrt(
            u0 / (4 * np.pi * material.k * material.alpha * (model.t_death - model.Ta)))
        thermal_adaptation.update(x0.item(), u_prime.item())
        model.deflection_mm = defl_mm
        x0 = simulator.make_step(u0)
        pltconfig.w_mm = x0.item()
        pltconfig.defl_mm = defl_mm
        pltconfig.u0 = u0.item()
        if not plot(pltconfig):
            break
        pltconfig.deflection_plotter.plot()
        pltconfig.thermal_plotter.plot()
        plt.pause(0.0001)

def prerecorded_experiment(run_conf: RunConfig, pltconfig: PlotConfig,
                           model, material: MaterialProperties, mpc,
                           params: Parameters,
                           defl_adaptation: DeflectionAdaptation,
                           therm_adaptation: ThermalAdaptation):

    with open(run_conf.log_file_to_load, 'rb') as f:
        data = pkl.load(f)
    if isinstance(data, LoggingData):
        thermal_frames = data.thermal_frames
        bs = data.damping_estimates
        vs = data.velocities
    else:
        thermal_frames = data['thermal_frames']
        try:
            bs = data['damping_estimates']
        except KeyError:
            bs = data['deflection_estimates']
        try:
            vstars = data['vstars']
        except KeyError:
            vstars = [None] * len(thermal_frames)
        vs = data['velocities']

    init_mpc = False
    tip_neutral_px = []
    defl_covs = []
    defls = []
    fill = None
    for i, (frame, b, v, vstar) in enumerate(zip(tqdm.tqdm(thermal_frames), bs, vs, vstars)):
        color_frame = thermal_frame_to_color(frame)
        vstar = 7 if vstar is None else vstar
        u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(run_conf, params, model, material, tip_neutral_px, defl_adaptation, therm_adaptation, v, vstar,
                                                             frame, init_mpc, mpc)
        new_tip_pos = defl_adaptation.kf.x[:2] * params.thermal_px_per_mm
        color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos, defl_adaptation.neutral_tip_mm * params.thermal_px_per_mm)
        cv.imshow("Frame", color_frame)
        pltconfig.w_mm = w_mm
        pltconfig.defl_mm = defl_mm
        pltconfig.u0 = u0
        if not plot(pltconfig):
            break
        defl_covs.append(3 * defl_adaptation.defl_std)
        defls.append(defl_mm)
        # if fill is not None:
        #     fill.remove()
        # fill = plotter.axs[-1].fill_between(range(len(defl_covs)), np.array(defls) - defl_covs, np.array(defls) + defl_covs, alpha=0.25, color='blue')

        cv.waitKey(1)
        plt.pause(0.0001)