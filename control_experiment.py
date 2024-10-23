import do_mpc.controller
from T3pro import T3pro
from testbed import Testbed
from models import *
from AdaptiveID import *
from utils import *
from datetime import datetime
from enum import StrEnum
import pandas as pd
import warnings
from typing import Optional
from dataclasses import dataclass
import tqdm
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

thermal_px_per_mm = 5.1337 # px/mm
v_min = 0.  # minimum velocity [mm/s]
v_max = 10  # maximum velocity [mm/s]


def update_adaptive_params(model, deflection_adaptation: ScalarLinearAlgabraicAdaptation,
                           thermal_adaptation: ScalarLinearAlgabraicAdaptation,
                           defl_mm: float, w_mm: float, u0: float):
    """
    Update the adaptive parameters
    :param model: Model object
    :param deflection_adaptation: Deflection adaptation object
    :param thermal_adaptation: Thermal adaptation object
    :param defl_mm: Deflection [mm]
    :param w_mm: Width [mm]
    :param u0: Velocity [mm/s]
    """
    deflection_adaptation.update(defl_mm, np.exp(-model.c_defl / u0))
    thermal_adaptation.update(w_mm, ymax(1, u0, Tc(model.t_death - model.Ta, model.material, model._P)))
    model.material.alpha = thermal_adaptation.b
    model.isotherm_widths_mm = w_mm
    model.deflection_mm = defl_mm
    model._d = deflection_adaptation.b
    return deflection_adaptation.b, thermal_adaptation.b


def update(params: RunningParams, model,
           tipPos: ToolTipKF,
           deflection_adaptation: ScalarLinearAlgabraicAdaptation,
           thermal_adaptation: ScalarLinearAlgabraicAdaptation,
           u0: float, frame: np.ndarray, init_mpc: bool, mpc: Optional[do_mpc.controller.MPC]):
    """
    Update velocity and adaptive parameters
    :param params: Running parameters
    :param model: Model object
    :param tipPos: ToolTipKF object
    :param deflection_adaptation: Deflection adaptation object
    :param thermal_adaptation: Thermal adaptation object
    :param u0: Current velocity
    :param frame: Thermal frame
    :param init_mpc: MPC initialization flag
    :param mpc: Optional MPC object
    :return: Updated velocity, deflection, width, MPC initialization flag
    """
    defl_px, _ = tipPos.update_with_measurement(frame)
    defl_mm = defl_px / thermal_px_per_mm
    tip_lead_distance = 0
    w_px, ellipse = cv_isotherm_width(frame, model.isotherm_temps[0])
    w_mm = w_px / thermal_px_per_mm
    if ellipse is not None and tipPos.pos_px is not None:
        tip_lead_distance = find_wavefront_distance(ellipse, tipPos.pos_px) / thermal_px_per_mm
    update_adaptive_params(model, deflection_adaptation, thermal_adaptation, defl_mm, w_mm, u0)
    if not init_mpc and params.mpc:
        mpc.x0['width_0'] = w_mm
        mpc.x0['tip_lead_dist'] = tip_lead_distance
        mpc.set_initial_guess()
        init_mpc = True
    if params.mpc:
        u0 = mpc.make_step(np.array([w_mm, tip_lead_distance])).item()
    else:
        u0 = model.find_optimal_velocity()
    return u0, defl_mm, w_mm, init_mpc

def Tc(dT: float, material: MaterialProperties, P: float):
    """
    Calculate the dimensionless temperature
    :param dT: Temperature difference [K]
    :param material: Material properties
    :param P: Power [W]
    """
    return 2 * np.pi * material.k * dT / P

def setup_mpc(model, mpc, thermal_adaptation, deflection_adaptation, r):
    mpc.set_objective(mterm=model.aux['mterm'], lterm=model.aux['lterm'])
    mpc.set_rterm(u=r)
    mpc.bounds['lower', '_u', 'u'] = v_min
    mpc.bounds['upper', '_u', 'u'] = v_max
    for i in range(model.n_isotherms):
        mpc.bounds['lower', '_x', f'width_{i}'] = 0
    mpc.bounds['lower', '_z', 'deflection'] = 0

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        tvp_template['_tvp'] = np.array([model.deflection_mm, deflection_adaptation.b, thermal_adaptation.b])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.u0 = (v_min + v_max) / 2
    init_mpc = False
    mpc.setup()

    ## Setup plotting
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

    # Initialize the tool tip kalman filter
    tipPos = ToolTipKF(0.7)

    ## Initialize the parameter adaptation
    thermal_adaptation = ScalarLinearAlgabraicAdaptation(b=model.material.alpha, gamma=0.01)
    deflection_adaptation = ScalarLinearAlgabraicAdaptation(b=0.1, gamma=0.5)

    #############################

    if params.mpc:
        plotter = setup_mpc(model, mpc, thermal_adaptation, deflection_adaptation, r)

    else:
        plotter = GenericPlotter(n_plots=5,
                                 labels=['Widths', 'Velocities', 'Deflections', 'Deflection Adaptation', 'Thermal Adaptation'],
                                 x_label='Time [s]',
                                 y_labels=[r'$\text{Widths [mm]}$',
                                           r'$\text{Velocities [mm/s]}$',
                                           r'$\delta \text{[mm]}$',
                                           r'$d$',
                                           r'$\alpha$'])

    if params.exp_type == ExperimentType.PRERECORDED:
        prerecorded_experiment(params, model, tipPos, tb, t3, plotter, mpc, thermal_adaptation, deflection_adaptation)

    elif params.exp_type == ExperimentType.SIMULATED:
        simulated_experiment(params, model, tipPos, tb, t3, plotter, mpc, thermal_adaptation, deflection_adaptation)

    elif params.exp_type == ExperimentType.REAL:
        real_experiment(params, model, tipPos, tb, t3, plotter, mpc, thermal_adaptation, deflection_adaptation)

    plt.show()
    if tb is not None:
        tb.stop()
    if t3 is not None:
        t3.release()


def real_experiment(params: RunningParams, model, tipPos, tb, t3, plotter, mpc, thermal_adaptation: ScalarLinearAlgabraicAdaptation, deflection_adaptation: ScalarLinearAlgabraicAdaptation):
    u0 = (v_min + v_max) / 2
    data_log = pd.DataFrame(
        columns=['widths_mm', 'velocities', 'deflections_mm', 'thermal_frames', 'damping_estimates', 'thermal_diffusivity_estimates'])

    if params.home:
        tb.home()

    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = lut[raw_frame]
    thermal_frame_to_color(thermal_arr)
    init_mpc = False
    start_quit = input("Press enter to start the experiment, q to quit")
    if start_quit == 'q':
        return
    while ret:
        ret, raw_frame = t3.read()
        info, lut = t3.info()
        thermal_arr = lut[raw_frame]
        color_frame = thermal_frame_to_color(thermal_arr)
        cv.imshow("Frame", color_frame)

        u0, defl_mm, w_mm, init_mpc = update(params, model, tipPos, deflection_adaptation, thermal_adaptation, u0, thermal_arr, init_mpc, mpc)
        if not params.adaptive_velocity:
            u0 = params.constant_velocity
        else:
            if params.mpc:
                plotter.plot()
            else:
                plotter.plot([w_mm, u0, defl_mm, thermal_adaptation.b, deflection_adaptation.b])

        tb.set_speed(u0)
        data_log.loc[len(data_log)] = [w_mm, mpc.u0 if params.adaptive_velocity else params.constant_velocity,
                                       defl_mm,
                                       thermal_arr, deflection_adaptation.b, thermal_adaptation.b]
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    tb.set_speed(0)
    if len(data_log['widths_mm']) > 0:
        date = datetime.now()
        mode = "adaptive" if params.adaptive_velocity else f"{params.constant_velocity:.0f}mm_s"
        fname = f"{params.log_save_dir}/data_{mode}_{date.strftime('%Y-%m-%d-%H:%M')}.pkl"
        data_log.to_pickle(fname)

def simulated_experiment(params: RunningParams, model, tipPos, tb, t3, plotter, mpc, thermal_adaptation: ScalarLinearAlgabraicAdaptation, deflection_adaptation: ScalarLinearAlgabraicAdaptation):
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
        defl = deflection_adaptation.b * np.exp(-model.c_defl / u0.item()) * np.random.normal(1, 0.05)
        if defl < 0:
            defl = 0
        deflection_adaptation.update(defl, u0.item())
        u_prime = (2 * model.material.alpha / u0) * np.sqrt(
            u0 / (4 * np.pi * model.material.k * model.material.alpha * (model.t_death - model.Ta)))
        thermal_adaptation.update(x0.item(), u_prime.item())
        model.deflection_mm = defl
        x0 = simulator.make_step(u0)
        if params.mpc:
            plotter.plot()
        else:
            plotter.plot([x0, u0, defl, thermal_adaptation.b, deflection_adaptation.b])
        plt.pause(0.0001)

def prerecorded_experiment(params: RunningParams, model, tipPos, tb, t3, plotter, mpc, thermal_adaptation: ScalarLinearAlgabraicAdaptation, deflection_adaptation: ScalarLinearAlgabraicAdaptation):
    data = pkl.load(open(params.log_file_to_load, 'rb'))
    if isinstance(data, LoggingData):
        thermal_frames = data.thermal_frames
        bs = data.damping_estimates
    else:
        thermal_frames = data['thermal_frames']
        bs = data['damping_estimates']
    u0 = (v_min + v_max) / 2
    init_mpc = False
    for i, (frame, b) in enumerate(zip(tqdm.tqdm(thermal_frames), bs)):
        u0, defl_mm, w_mm, init_mpc = update(params, model, tipPos, deflection_adaptation, thermal_adaptation, u0,
                                             frame, init_mpc, mpc)
        if params.mpc:
            plotter.plot()
        else:
            plotter.plot([w_mm, u0, defl_mm, thermal_adaptation.b, deflection_adaptation.b])
        plt.pause(0.0001)