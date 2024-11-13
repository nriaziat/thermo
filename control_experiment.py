import cv2
import do_mpc.controller
from T3pro import T3pro
from testbed import Testbed
from AdaptiveID import *
from utils import *
from Plotters import *
from datetime import datetime
from enum import StrEnum
import pandas as pd
import warnings
from typing import Optional
from dataclasses import dataclass
import tqdm
from tkinter import messagebox
from datetime import datetime
warnings.filterwarnings("ignore")

class ExperimentType(StrEnum):
    REAL = "r"
    PRERECORDED = "p"
    SIMULATED = "s"

@dataclass
class RunningParams:
    exp_type: ExperimentType
    adaptive_velocity: bool
    constant_velocity: float
    log_save_dir: str
    log_file_to_load: str
    mpc: bool
    home: bool
    plot_adaptive_params: bool

thermal_px_per_mm = 5.1337 # px/mm
v_min = 0.  # minimum velocity [mm/s]
v_max = 25  # maximum velocity [mm/s]


def update_adaptive_params(model, defl_adaptation: DeflectionAdaptation, therm_adaptation: ThermalAdaptation,
                           tip_mm: np.ndarray, w_mm: float, u0: float):
    """
    Update the adaptive parameters
    :param model: Model object
    :param defl_adaptation: Deflection adaptation object
    :param therm_adaptation: Thermal adaptation object
    :param tip_mm: Tip position mm [mm]
    :param w_mm: Width [mm]
    :param u0: Velocity [mm/s]
    """
    # deflection_adaptation.update(defl_mm, np.exp(-model.c_defl / u0))
    dT = model.t_death - model.Ta
    if defl_adaptation.init:
        if tip_mm is not None:
            defl_adaptation.update(np.array([tip_mm[0], tip_mm[1], tip_mm[1]]), v=u0)
        else:
            defl_adaptation.update(None, v=u0)
    therm_adaptation.update(np.array([w_mm]), dT=dT, v=u0)
    model.material.k = therm_adaptation.k_w_mmK
    model.material.rho = therm_adaptation.rho_kg_mm3
    model.material.Cp = therm_adaptation.Cp
    model._P = therm_adaptation.q
    model._c_defl = defl_adaptation.c_defl


def update(params: RunningParams, model,
           tip_candidates_px: list[np.ndarray],
           defl_adaptation: DeflectionAdaptation,
           therm_adaptation: ThermalAdaptation,
           u0: float, frame: np.ndarray, init_mpc: bool, mpc: Optional[do_mpc.controller.MPC]):
    """
    Update velocity and adaptive parameters
    :param params: Running parameters
    :param model: Model object
    :param tip_candidates_px: Tool tip resting position
    :param defl_adaptation: Deflection adaptation object
    :param therm_adaptation: Thermal adaptation object
    :param u0: Current velocity
    :param frame: Thermal frame
    :param init_mpc: MPC initialization flag
    :param mpc: Optional MPC object
    :return: Updated velocity, deflection, width, MPC initialization flag
    """
    tip_px = find_tooltip(frame, 70)
    if tip_px is not None:
        tip_px = np.array(tip_px)
        if len(tip_candidates_px) < 1:
            tip_candidates_px.append(tip_px)
        elif not defl_adaptation.init:
            tip_neutral_px = np.median(tip_candidates_px, axis=0)
            defl_adaptation.kf.x[4:6] = tip_neutral_px / thermal_px_per_mm
            defl_adaptation.kf.x[0:2] = tip_neutral_px / thermal_px_per_mm
            defl_adaptation.init = True

    tip_mm = np.array(tip_px) / thermal_px_per_mm if tip_px is not None else None
    tip_lead_distance = 0
    w_px, ellipse = cv_isotherm_width(frame, model.isotherm_temps[0])
    w_mm = w_px / thermal_px_per_mm
    update_adaptive_params(model, defl_adaptation, therm_adaptation, tip_mm, w_mm, u0)
    model.deflection_mm = defl_adaptation.defl_mm
    model.width_mm = therm_adaptation.w_mm
    if not init_mpc and params.mpc:
        mpc.x0['width_0'] = w_mm
        mpc.x0['tip_lead_dist'] = tip_lead_distance
        mpc.set_initial_guess()
        init_mpc = True
    if params.mpc:
        u0 = mpc.make_step(np.array([w_mm, tip_lead_distance])).item()
    else:
        u0 = model.find_optimal_velocity()
    return u0, defl_adaptation.defl_mm, therm_adaptation.w_mm, init_mpc, tip_candidates_px

def setup_mpc(model, mpc, defl_adapt, therm_adapt, r):
    mpc.set_objective(mterm=model.aux['mterm'], lterm=model.aux['lterm'])
    mpc.set_rterm(u=r)
    mpc.bounds['lower', '_u', 'u'] = v_min
    mpc.bounds['upper', '_u', 'u'] = v_max
    for i in range(model.n_isotherms):
        mpc.bounds['lower', '_x', f'width_{i}'] = 0
    mpc.bounds['lower', '_z', 'deflection'] = 0

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        tvp_template['_tvp'] = np.array([model.deflection_mm, defl_adapt.b, therm_adapt.b])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.u0 = (v_min + v_max) / 2
    mpc.setup()

    return MPCPlotter(mpc.data, isotherm_temps=model.isotherm_temps)

def main(model,
         params: RunningParams,
         mpc: Optional[do_mpc.controller.MPC] = None,
         t3: Optional[T3pro] = None,
         tb: Optional[Testbed] = None,
         r = 1e-3):
    """
    Main function to run the experiment
    :param model: Model object
    :param params: Running parameters
    :param mpc: Optional MPC object
    :param t3: T3pro object
    :param tb: Testbed object
    :param r: Input change penalty
    """
    if params.mpc:
        assert mpc is not None, "MPC object must be provided if MPC is enabled"

    ## Initialize the parameter adaptation
    # thermal_adaptation = ScalarLinearAlgabraicAdaptation(b=model.material.alpha, gamma=0.01)
    thermal_adaptation = ThermalAdaptation(np.array([0, model._P, model.material.Cp]), labels=['w', 'P', 'Cp'])
    # deflection_adaptation = ScalarLinearAlgabraicAdaptation(b=0.1, gamma=0.5)
    deflection_adaptation = DeflectionAdaptation(np.array([0, 0, 0, 0, 0, 0, 1, 1]), labels=['x', 'y', 'x_dot', 'y_dot', 'x_rest', 'y_rest', 'c_defl', 'k'],
                                                 px_per_mm=thermal_px_per_mm, frame_size_px=(384, 288))
    # adaptation = ComboAdaptation(np.array([0, 0, 0, 0, 0, 0, 5, 5, 0, model._P, model.material.Cp]),
    #                              labels=['x', 'y', 'x_dot', 'y_dot', 'x_rest', 'y_rest', 'b', 'c_defl', 'w', 'q', 'Cp'])
    deflection_adaptation.init = False

    #############################

    if params.mpc:
        plotter = setup_mpc(model, mpc, deflection_adaptation, thermal_adaptation, r)

    else:
        plotter = GenericPlotter(n_plots=3,
                                 labels=['Widths', 'Velocities', 'Deflections'],
                                 x_label='Time [s]',
                                 y_labels=[r'$\text{Widths [mm]}$',
                                           r'$\text{Velocities [mm/s]}$',
                                           r'$\delta \text{[mm]}$'])
    if params.plot_adaptive_params:
        deflection_plotter = AdaptiveParameterPlotter(deflection_adaptation, plot_indices=[0, 0, 0, 0, 0, 0, 1, 1])
        thermal_plotter = AdaptiveParameterPlotter(thermal_adaptation, plot_indices=[0, 1, 1])
    else:
        deflection_plotter = None
        thermal_plotter = None


    if params.exp_type == ExperimentType.PRERECORDED:
        prerecorded_experiment(params, model, plotter, mpc, deflection_adaptation, thermal_adaptation, deflection_plotter, thermal_plotter)

    elif params.exp_type == ExperimentType.SIMULATED:
        simulated_experiment(params, model, plotter, mpc, deflection_adaptation, thermal_adaptation, deflection_plotter, thermal_plotter)

    elif params.exp_type == ExperimentType.REAL:
        real_experiment(params, model, tb, t3, plotter, mpc, deflection_adaptation, thermal_adaptation, deflection_plotter, thermal_plotter)

    plt.show()
    if tb is not None:
        tb.stop()
    if t3 is not None:
        t3.release()
    cv2.destroyAllWindows()


def plot(params: RunningParams, plotter: GenericPlotter | MPCPlotter, w_mm: float, defl_mm: float, u0: float,
         deflection_plotter:Optional[AdaptiveParameterPlotter]=None, thermal_plotter:Optional[AdaptiveParameterPlotter]=None,
         defl_hist_mm: Optional[list] = None) -> bool:
    """
    :param params: Running parameters
    :param plotter: Plotter object
    :param w_mm: Width [mm]
    :param defl_mm: Deflection [mm]
    :param u0: Velocity [mm/s]
    :param deflection_plotter: Deflection plotter object
    :param thermal_plotter: Thermal plotter object
    :param defl_hist_mm: Deflection history [mm] (Optional)
    """
    if params.mpc:
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


def real_experiment(params: RunningParams, model, tb, t3, plotter, mpc,
                    defl_adaptation: DeflectionAdaptation,
                    therm_adaptation: ThermalAdaptation,
                    deflection_plotter, thermal_plotter):
    u0 = (v_min + v_max) / 2
    data_log = pd.DataFrame(
        columns=['time_sec', 'position_mm', 'widths_mm', 'velocities', 'deflections_mm', 'thermal_frames', 'deflection_estimates', 'thermal_estimates'])
    if params.home:
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


        u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(params, model, tip_neutral_px, defl_adaptation, therm_adaptation, u0, thermal_arr, init_mpc, mpc)
        new_tip_pos = defl_adaptation.kf.x[:2] * thermal_px_per_mm
        neutral_tip_pos = defl_adaptation.neutral_tip_mm * thermal_px_per_mm
        color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos,  neutral_tip_pos)
        cv.imshow("Frame", color_frame)
        if not params.adaptive_velocity:
            u0 = params.constant_velocity
        if not plot(params, plotter, w_mm, defl_mm, u0, deflection_plotter, thermal_plotter, defl_hist_mm=defl_adaptation.defl_hist_mm):
            break
        # if defl_mm > 5:
        #     u0 = np.clip(u0 - v_max / 5 * defl_mm, -v_max, v_max)
        if defl_mm > 10:
            messagebox.showerror("Error", "Deflection too high, stopping the experiment")
            break


        tb.set_speed(u0)
        pos = tb.get_position()
        dt = (datetime.now() - time0).total_seconds()
        data_log.loc[len(data_log)] = [dt, pos, w_mm, u0,
                                       defl_mm,
                                       thermal_arr, {'c_defl': defl_adaptation.c_defl},
                                       {'q': therm_adaptation.q, 'Cp': therm_adaptation.Cp}]
        if cv.waitKey(1) & 0xFF == ord('q') or pos == -1:
            break

    # update deflections with final neutral tip position
    data_log['deflections_mm'] = defl_adaptation.defl_hist_mm
    print(f"Total time: {dt:.2f} seconds")
    print(f"7mm/s equivalent time: {pos / 7:.2f} seconds")

    tb.set_speed(0)
    if len(data_log['widths_mm']) > 0 and params.log_save_dir != "":
        date = datetime.now()
        mode = "adaptive" if params.adaptive_velocity else f"{params.constant_velocity:.0f}mm_s"    
        fname = f"{params.log_save_dir}/data_{mode}_{date.strftime('%Y-%m-%d-%H:%M')}.pkl"
        data_log.to_pickle(fname)

def simulated_experiment(params: RunningParams, model, plotter, mpc,
                         deflection_adaptation: ScalarLinearAlgabraicAdaptation | DeflectionAdaptation,
                         thermal_adaptation: ScalarLinearAlgabraicAdaptation | ThermalAdaptation,
                         deflection_plotter, thermal_plotter):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=1 / 24, integration_tool='idas')
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
    n_steps = 1000
    simulator.set_initial_guess()
    mpc.set_initial_guess()
    for _ in tqdm.trange(n_steps):
        u0 = mpc.make_step(x0)
        defl_mm = deflection_adaptation.b * np.exp(-model._c_defl / u0.item()) * np.random.normal(1, 0.05)
        if defl_mm < 0:
            defl_mm = 0
        deflection_adaptation.update(defl_mm, u0.item())
        u_prime = (2 * model.material.alpha / u0) * np.sqrt(
            u0 / (4 * np.pi * model.material.k * model.material.alpha * (model.t_death - model.Ta)))
        thermal_adaptation.update(x0.item(), u_prime.item())
        model.deflection_mm = defl_mm
        x0 = simulator.make_step(u0)
        if not plot(params, plotter, x0, defl_mm, u0, deflection_plotter, thermal_plotter):
            break
        deflection_plotter.plot()
        thermal_plotter.plot()
        plt.pause(0.0001)

def prerecorded_experiment(params: RunningParams, model, plotter, mpc,
                           defl_adaptation: DeflectionAdaptation,
                           therm_adaptation: ThermalAdaptation,
                           deflection_plotter, thermal_plotter):

    with open(params.log_file_to_load, 'rb') as f:
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
    u0 = (v_min + v_max) / 2
    init_mpc = False
    tip_neutral_px = []
    defl_covs = []
    defls = []
    fill = None
    for i, (frame, b, v) in enumerate(zip(tqdm.tqdm(thermal_frames), bs, vs)):
        color_frame = thermal_frame_to_color(frame)
        u0, defl_mm, w_mm, init_mpc, tip_neutral_px = update(params, model, tip_neutral_px, defl_adaptation, therm_adaptation, v,
                                             frame, init_mpc, mpc)
        new_tip_pos = defl_adaptation.kf.x[:2] * thermal_px_per_mm
        color_frame = draw_info_on_frame(color_frame, defl_mm, w_mm, u0, new_tip_pos, defl_adaptation.neutral_tip_mm * thermal_px_per_mm)
        cv.imshow("Frame", color_frame)
        if not plot(params, plotter, w_mm, defl_mm, u0, deflection_plotter, thermal_plotter, defl_hist_mm=defl_adaptation.defl_hist_mm):
            break
        defl_covs.append(3 * defl_adaptation.defl_std)
        defls.append(defl_mm)
        if fill is not None:
            fill.remove()
        fill = plotter.axs[-1].fill_between(range(len(defl_covs)), np.array(defls) - defl_covs, np.array(defls) + defl_covs, alpha=0.25, color='blue')

        cv.waitKey(1)
        plt.pause(0.0001)