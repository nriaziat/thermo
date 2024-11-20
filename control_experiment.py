import cv2
import do_mpc.controller
from T3pro import T3pro
from models import humanTissue, hydrogelPhantom
from testbed import Testbed
from ParameterEstimation import *
from utils import *
from Plotters import *
from enum import StrEnum
import pandas as pd
import warnings
from typing import Optional
from dataclasses import dataclass
import tqdm
from tkinter import messagebox
from datetime import datetime
from copy import deepcopy
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

class ExperimentType(StrEnum):
    REAL = "r"
    PRERECORDED = "p"
    SIMULATED = "s"

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
    adaptive_velocity: bool
    constant_velocity: float
    log_save_dir: str
    log_file_to_load: str
    mpc: bool
    home: bool
    plot_adaptive_params: bool
    material: MaterialProperties

class DataLogger:
    def __init__(self, log_save_dir: str, adaptive_velocity: bool, constant_velocity: float):
        self.log_save_dir = log_save_dir
        self.adaptive_velocity = adaptive_velocity
        self.constant_velocity = constant_velocity
        self.data_log = pd.DataFrame(
            columns=['time_sec', 'position_mm', 'widths_mm', 'velocities', 'deflections_mm', 'thermal_frames', 'deflection_estimates', 'thermal_estimates']
        )
        self.start_time = datetime.now()

    def log_data(self, pos, w_mm, u0, defl_mm, thermal_arr, defl_adaptation, therm_adaptation):
        dt = (datetime.now() - self.start_time).total_seconds()
        self.data_log.loc[len(self.data_log)] = [
            dt, pos, w_mm, u0, defl_mm, thermal_arr,
            {'c_defl': defl_adaptation.c_defl},
            {'q': therm_adaptation.q, 'Cp': therm_adaptation.Cp}
        ]

    def save_log(self):
        if len(self.data_log['widths_mm']) > 0 and self.log_save_dir != "":
            date = datetime.now()
            mode = "adaptive" if self.adaptive_velocity else f"{self.constant_velocity:.0f}mm_s"
            fname = f"{self.log_save_dir}/data_{mode}_{date.strftime('%Y-%m-%d-%H:%M')}.pkl"
            self.data_log.to_pickle(fname)

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
           u0: float, frame: np.ndarray, init_mpc: bool, mpc: Optional[do_mpc.controller.MPC]):
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
    update_kf(model, material, defl_adaptation, therm_adaptation, tip_mm, w_mm, u0)
    if not init_mpc and run_conf.mpc:
        mpc.x0['width_0'] = w_mm
        mpc.x0['tip_lead_dist'] = tip_lead_distance
        mpc.set_initial_guess()
        init_mpc = True
    if run_conf.mpc:
        u0 = mpc.make_step(np.array([w_mm, tip_lead_distance])).item()
    else:
        u0 = model.find_optimal_velocity(material, defl_adaptation.c_defl, therm_adaptation.q)
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
         mpc: Optional[do_mpc.controller.MPC] = None,
         t3: Optional[T3pro] = None,
         tb: Optional[Testbed] = None,
         r: float = 1e-3) -> None:
    """
    Main function to run the experiment
    :param model: Model object
    :param run_conf: Running parameters
    :param mpc: Optional MPC object
    :param t3: T3pro object
    :param tb: Testbed object
    :param r: Input change penalty
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
        plotter = setup_mpc(model, mpc, deflection_adaptation, thermal_adaptation, r, params)
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

    if run_conf.exp_type == ExperimentType.PRERECORDED:
        prerecorded_experiment(run_conf, model, material, plotter, mpc, params, deflection_adaptation, thermal_adaptation, deflection_plotter, thermal_plotter)

    elif run_conf.exp_type == ExperimentType.SIMULATED:
        simulated_experiment(run_conf, model, material, plotter, mpc, params, deflection_adaptation, thermal_adaptation, deflection_plotter, thermal_plotter)

    elif run_conf.exp_type == ExperimentType.REAL:
        real_experiment(run_conf, model, material, tb, t3, plotter, mpc, params, deflection_adaptation, thermal_adaptation, deflection_plotter, thermal_plotter)

    plt.show()
    if tb is not None:
        tb.stop()
    if t3 is not None:
        t3.release()
    cv2.destroyAllWindows()


def plot(run_conf: RunConfig, plotter: GenericPlotter | MPCPlotter, w_mm: float, defl_mm: float, u0: float,
         deflection_plotter:Optional[AdaptiveParameterPlotter]=None, thermal_plotter:Optional[AdaptiveParameterPlotter]=None,
         defl_hist_mm: Optional[list] = None) -> bool:
    """
    :param run_conf: Running parameters
    :param plotter: Plotter object
    :param w_mm: Width [mm]
    :param defl_mm: Deflection [mm]
    :param u0: Velocity [mm/s]
    :param deflection_plotter: Deflection plotter object
    :param thermal_plotter: Thermal plotter object
    :param defl_hist_mm: Deflection history [mm] (Optional)
    """
    if run_conf.mpc:
        plotter.plot()
    else:
        plotter.plot([w_mm, u0, defl_mm], defl_hist=defl_hist_mm)
    if deflection_plotter is not None:
        deflection_plotter.plot()
    if thermal_plotter is not None:
        thermal_plotter.plot()
    fignums = [p.fig.number for p in [plotter, deflection_plotter, thermal_plotter] if p is not None]
    if not all([plt.fignum_exists(f) for f in fignums]):
        return False
    return True


def real_experiment(run_conf: RunConfig, model, material: MaterialProperties, tb, t3, plotter, mpc,
                    params: Parameters,
                    defl_adaptation: DeflectionAdaptation,
                    therm_adaptation: ThermalAdaptation,
                    deflection_plotter, thermal_plotter):
    """
    Run the real experiment
    """

    u0 = (params.v_min + params.v_max) / 2
    data_logger = DataLogger(run_conf.log_save_dir, run_conf.adaptive_velocity, run_conf.constant_velocity)

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
    while ret:
        ret, raw_frame = t3.read()
        info, lut = t3.info()
        thermal_arr = lut[raw_frame]
        color_frame = thermal_frame_to_color(thermal_arr)

        u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(run_conf, params, model, material,
                                                             tip_neutral_px, defl_adaptation, therm_adaptation, u0, thermal_arr, init_mpc, mpc)
        new_tip_pos = defl_adaptation.kf.x[:2] * params.thermal_px_per_mm
        neutral_tip_pos = defl_adaptation.neutral_tip_mm * params.thermal_px_per_mm
        color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos,  neutral_tip_pos)
        cv.imshow("Frame", color_frame)
        if not run_conf.adaptive_velocity:
            u0 = run_conf.constant_velocity
        if not plot(run_conf, plotter, w_mm, defl_mm, u0, deflection_plotter, thermal_plotter, defl_hist_mm=defl_adaptation.defl_hist_mm):
            break
        if defl_mm > params.deflection_max:
            messagebox.showerror("Error", "Deflection too high, stopping the experiment")
            break


        tb.set_speed(u0)
        pos = tb.get_position()
        dt = (datetime.now() - time0).total_seconds()
        data_logger.log_data(pos, w_mm, u0, defl_mm, thermal_arr, defl_adaptation, therm_adaptation)
        if cv.waitKey(1) & 0xFF == ord('q') or pos == -1:
            break

    tb.set_speed(0)
    # data_logger.data_log['deflections_mm'] = defl_adaptation.defl_hist_mm
    # update deflections with final neutral tip position
    print(f"Total time: {dt:.2f} seconds")
    print(f"7mm/s equivalent time: {pos / 7:.2f} seconds")
    data_logger.save_log()

def simulated_experiment(run_conf: RunConfig, model, material, plotter, mpc,
                         params: Parameters,
                         deflection_adaptation: DeflectionAdaptation,
                         thermal_adaptation: ThermalAdaptation,
                         deflection_plotter, thermal_plotter):
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
        if not plot(run_conf, plotter, x0, defl_mm, u0, deflection_plotter, thermal_plotter):
            break
        deflection_plotter.plot()
        thermal_plotter.plot()
        plt.pause(0.0001)

def prerecorded_experiment(run_conf: RunConfig, model, material: MaterialProperties, plotter, mpc,
                           params: Parameters,
                           defl_adaptation: DeflectionAdaptation,
                           therm_adaptation: ThermalAdaptation,
                           deflection_plotter, thermal_plotter):

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
        vs = data['velocities']

    init_mpc = False
    tip_neutral_px = []
    defl_covs = []
    defls = []
    fill = None
    for i, (frame, b, v) in enumerate(zip(tqdm.tqdm(thermal_frames), bs, vs)):
        color_frame = thermal_frame_to_color(frame)
        u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(run_conf, params, model, material, tip_neutral_px, defl_adaptation, therm_adaptation, v,
                                                             frame, init_mpc, mpc)
        new_tip_pos = defl_adaptation.kf.x[:2] * params.thermal_px_per_mm
        color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos, defl_adaptation.neutral_tip_mm * params.thermal_px_per_mm)
        cv.imshow("Frame", color_frame)
        if not plot(run_conf, plotter, w_mm, defl_mm, u0, deflection_plotter, thermal_plotter, defl_hist_mm=defl_adaptation.defl_hist_mm):
            break
        defl_covs.append(3 * defl_adaptation.defl_std)
        defls.append(defl_mm)
        # if fill is not None:
        #     fill.remove()
        # fill = plotter.axs[-1].fill_between(range(len(defl_covs)), np.array(defls) - defl_covs, np.array(defls) + defl_covs, alpha=0.25, color='blue')

        cv.waitKey(1)
        plt.pause(0.0001)