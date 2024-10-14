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
warnings.filterwarnings("ignore")

class ExperimentType(StrEnum):
    REAL = "r"
    PRERECORDED = "p"
    SIMULATED = "s"

class SpeedMode(StrEnum):
    ADAPTIVE = "adaptive"
    CONSTANT = "constant"

thermal_px_per_mm = 5.1337 # px/mm
qw = 1  # width cost
qd = 20 # deflection cost
r = 0.001  # control change cost
v_min = 0  # minimum velocity [mm/s]
v_max = 10  # maximum velocity [mm/s]

def main(model,
         mpc,
         exp_type:ExperimentType,
         adaptive_velocity:bool,
         constant_velocity:float,
         log_save_dir:str,
         log_file_to_load:str,
         t3:Optional[T3pro]=None,
         tb:Optional[Testbed]=None):

    def add_to_data_log(data_log: pd.DataFrame, thermal_arr: np.array,
                        widths_mm: np.array,
                        deflection_mm: float,
                        adaptive_deflection_model: ScalarLinearAlgabraicAdaptation,
                        adaptive_velocity: bool = False,
                        const_velocity: float = None):

        data_log.loc[len(data_log)] = [widths_mm, mpc.u0 if adaptive_velocity else const_velocity, deflection_mm,
                                       thermal_arr, adaptive_deflection_model.b]

    def mpc_loop(mpc, u0, frame, init_mpc, b=None):
        defl_px, _ = tipPos.update_with_measurement(frame)
        defl_mm = defl_px / thermal_px_per_mm
        w_mm = np.array([(cv_isotherm_width(frame, temp)[0] / thermal_px_per_mm) for temp in model.isotherm_temps])
        model._isotherm_width_measurement = w_mm
        deflection_adaptation.update(defl_mm, np.exp(-model.c_defl / u0))
        u_prime = (2 * model._material.alpha / u0) * np.sqrt(
            u0 / (4 * np.pi * model._material.k * model._material.alpha * (model.t_death - model.Ta)))
        thermal_adaptation.update(w_mm[0], u_prime)
        if b is not None:
            deflection_adaptation.b = b
        if all(w_mm > 0) and len(w_mm) == len(set(w_mm)):
            if not init_mpc:
                mpc.x0 = w_mm
                mpc.set_initial_guess()
                init_mpc = True
            u0 = mpc.make_step(w_mm).item()
            # lb_violation = mpc.opt_x_num.cat <= mpc._lb_opt_x.cat
            # ub_violation = mpc.opt_x_num.cat >= mpc._ub_opt_x.cat
            # opt_labels = mpc.opt_x.labels()
            # labels_lb_viol = np.array(opt_labels)[np.where(lb_violation)[0]]
            # labels_ub_viol = np.array(opt_labels)[np.where(ub_violation)[0]]
            # print(f"Lower bound violation: {labels_lb_viol}")
            # print(f"Upper bound violation: {labels_ub_viol}")
            # if len(labels_lb_viol) > 0 and len(labels_ub_viol) > 0:
            #     input()
            plotter.plot()
            plt.pause(0.0001)
        return u0, init_mpc, defl_mm

    # Initialize the physical models
    # model = MultiIsothermModel(n_isotherms=3,
    #                            material=hydrogelPhantom)
    # model = PseudoStaticModel(material=hydrogelPhantom)
    # model.set_cost_function(qw, qd)
    # model.setup()
    tipPos = ToolTipKF(0.7)

    ## Initialize the parameter adaptation
    thermal_adaptation = ScalarLinearAlgabraicAdaptation(b=45, gamma=0.5)
    deflection_adaptation = ScalarLinearAlgabraicAdaptation(b=0.1, gamma=0.5)


    #############################


    mpc.set_objective(mterm=model.aux['mterm'], lterm=model.aux['lterm'])
    mpc.set_rterm(u=r)
    mpc.bounds['lower', '_u', 'u'] = v_min
    mpc.bounds['upper', '_u', 'u'] = v_max
    for i in range(model.n_isotherms):
        # mpc.scaling['_x', f'width_{i}'] = 1
        mpc.bounds['lower', '_x', f'width_{i}'] = 0
        # mpc.set_nl_cons(f'width_{i}', model.isotherm_widths_mm[i], ub=25, soft_constraint=True)
        # mpc.bounds['upper', '_x', f'width_{i}'] = 25
    mpc.bounds['lower', '_z', 'deflection'] = 0
    # mpc.scaling['_z', 'deflection'] = 0.1
    # for i in range(model.n_isotherms - 1):
    #     mpc.set_nl_cons(f'width_{i}_ordering_constr', model.isotherm_widths_mm[i] - model.isotherm_widths_mm[i+1], ub=0, soft_constraint=False)

    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        tvp_template['_tvp'] = np.array([model.deflection_mm, deflection_adaptation.b, thermal_adaptation.b])
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)
    mpc.u0 = (v_min + v_max) / 2
    init_mpc = False
    mpc.setup()

    ## Setup plotting
    plotter = Plotter(mpc.data, isotherm_temps=model.isotherm_temps)

    # exp_type = input("Real, Pre-recorded, or Simulated? (r/p/s): ").lower().strip()
    exp_type = ExperimentType(exp_type)
    if exp_type == ExperimentType.REAL:
        if not adaptive_velocity:
            # constant_velocity =
            # constant_velocity = float(input("Enter constant velocity: "))
            print(f"Constant velocity: {constant_velocity} mm/s")
            speed_mode = SpeedMode.CONSTANT
        else:
            speed_mode = SpeedMode.ADAPTIVE

    if exp_type == ExperimentType.PRERECORDED:
        data = pkl.load(open(log_file_to_load, 'rb'))
        if isinstance(data, LoggingData):
            thermal_frames = data.thermal_frames
            bs = data.damping_estimates
        else:
            thermal_frames = data['thermal_frames']
            bs = data['damping_estimates']
        u0 = (v_min + v_max) / 2
        for frame, b in zip(thermal_frames, bs):
            u0, init_mpc, defl_mm = mpc_loop(mpc, u0, frame, init_mpc, b)
            model.deflection_mm = defl_mm

    elif exp_type == ExperimentType.SIMULATED:
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=1/24, integration_tool='idas')
        sim_tvp = simulator.get_tvp_template()
        def sim_tvp_fun(t_now):
            sim_tvp['d'] = 0.5
            sim_tvp['defl_meas'] = model.deflection_mm
            sim_tvp['P'] = 20
            return sim_tvp
        simulator.set_tvp_fun(sim_tvp_fun)
        simulator.setup()
        x0 = np.linspace(2, 10, model.n_isotherms)
        print(f"Initial Widths: {x0}")
        simulator.x0 = x0
        mpc.x0 = x0
        n_steps = 1000
        u0 = (v_min + v_max) / 2
        simulator.set_initial_guess()
        mpc.set_initial_guess()
        for i in range(n_steps):
            u0 = mpc.make_step(x0)
            defl = deflection_adaptation.b * np.exp(-model.c_defl / u0.item()) * np.random.normal(1, 0.05)
            if defl < 0:
                defl = 0
            deflection_adaptation.update(defl, u0.item())
            u_prime = (2 * model._material.alpha / u0) * np.sqrt(
                u0 / (4 * np.pi * model._material.k * model._material.alpha * (model.t_death - model.Ta)))
            thermal_adaptation.update(x0.item(), u_prime.item())
            model.deflection_mm = defl
            x0 = simulator.make_step(u0)
            plotter.plot()
            plt.pause(0.0001)


    elif exp_type == ExperimentType.REAL:
        u0 = (v_min + v_max) / 2
        data_log = pd.DataFrame(columns=['widths_mm', 'velocities', 'deflections_mm', 'thermal_frames', 'damping_estimates'])
        home_input = input("Press Enter to home the testbed or 's' to skip: ")
        if home_input != 's':
            print("Homing testbed...")
            tb.home()
            print("Testbed homed.")
        else:
            print("Skipping homing.")

        ret, raw_frame = t3.read()
        info, lut = t3.info()
        thermal_arr = lut[raw_frame]
        thermal_frame_to_color(thermal_arr)

        start_input = input(
            "Ensure tool is on and touching tissue. Press Enter to start the experiment or 'q' to quit: ")
        if start_input == 'q':
            print("Quitting...")
        else:
            while ret:
                ret, raw_frame = t3.read()
                info, lut = t3.info()
                thermal_arr = lut[raw_frame]
                color_frame = thermal_frame_to_color(thermal_arr)
                cv.imshow("Frame", color_frame)
                u0, init_mpc, defl_mm = mpc_loop(mpc, u0, thermal_arr, init_mpc)
                model.deflection_mm = defl_mm
                add_to_data_log(data_log, thermal_arr, model.isotherm_widths_mm,
                                tipPos.update_with_measurement(thermal_arr)[0] / thermal_px_per_mm,
                                deflection_adaptation,
                                adaptive_velocity,
                                0 if adaptive_velocity else constant_velocity)
                tb.set_speed(u0)
                if len(mpc.data['_x', 'width_0']) > 0:
                    plotter.plot()
                    plt.pause(0.0001)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            tb.set_speed(0)

    plt.show()
    if tb is not None:
        tb.stop()
    if t3 is not None:
        t3.release()
    if exp_type == ExperimentType.REAL and len(data_log['widths_mm']) > 0:
        date = datetime.now()
        fname = f"{log_save_dir}/data_{str(speed_mode)}_{date.strftime('%Y-%m-%d-%H:%M')}.pkl"
        data_log.to_pickle(fname)