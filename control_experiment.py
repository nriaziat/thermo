import matplotlib.pyplot as plt
from T3pro import T3pro
from testbed import Testbed
from models import *
from AdaptiveID import *
from matplotlib import rcParams
from utils import *
from datetime import datetime

thermal_px_per_mm = 5.1337 # px/mm
qw = 1  # width cost
qd = 1 # deflection cost
r = 0.01  # control change cost
v_min = 1  # minimum velocity [mm/s]
v_max = 10  # maximum velocity [mm/s]

def mpc_loop(mpc, u0, frame, init_mpc):
    defl_px, _ = tipPos.update_with_measurement(frame)
    defl_mm = defl_px / thermal_px_per_mm
    w = np.array([cv_isotherm_width(frame, temp)[0] / thermal_px_per_mm for temp in model.isotherm_temps])
    deflection_adaptation.update(defl_mm, u0)
    if any(w < 1) or len(w) != len(set(w)):
        return u0, init_mpc
    if not init_mpc:
        mpc.x0 = w
        mpc.set_initial_guess()
        init_mpc = True
    u0 = mpc.make_step(w).item()
    plotter.plot()
    plt.pause(0.0001)
    return u0, init_mpc

def add_to_data_log(data_log: dict, thermal_arr: np.array,
                    widths_mm: np.array,
                    deflection_mm: float,
                    adaptive_deflection_model: ScalarLinearAlgabraicAdaptation,
                    adaptive_velocity: bool = False,
                    const_velocity: float = None):
    data_log['widths_mm'].append(widths_mm)
    if adaptive_velocity:
        data_log['velocities'].append(mpc.u0)
    else:
        data_log['velocities'].append(const_velocity)
    data_log['deflections_mm'].append(deflection_mm)
    data_log['thermal_frames'].append(thermal_arr)
    data_log['damping_estimates'].append(adaptive_deflection_model.b)


if __name__ == "__main__":
    plt.ion()
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'
    rcParams['axes.grid'] = True
    rcParams['lines.linewidth'] = 2.0
    rcParams['axes.labelsize'] = 'xx-large'
    rcParams['xtick.labelsize'] = 'xx-large'
    rcParams['ytick.labelsize'] = 'xx-large'

    ## Initialize the physical models
    model = MultiIsothermModel(n_isotherms=3)
    model.set_cost_function(qw, qd)
    model.setup()
    tipPos = ToolTipKF(0.7)

    ## Initialize the parameter adaptation
    # thermal_adaptation = ScalarFirstOrderAdaptation(x0=0, a0=0.1, b0=0.1)
    deflection_adaptation = ScalarLinearAlgabraicAdaptation(b=0.1)

    ## Initialize the MPC controller
    mpc = do_mpc.controller.MPC(model=model)

    ##############################
    mpc.settings.n_horizon = 20
    mpc.settings.n_robust = 0
    mpc.settings.open_loop = 0
    mpc.settings.t_step = 1/24
    mpc.settings.state_discretization = 'collocation'
    mpc.settings.collocation_type = 'radau'
    mpc.settings.collocation_deg = 2
    mpc.settings.collocation_ni = 2
    mpc.settings.store_full_solution = True
    mpc.settings.nlpsol_opts = {'ipopt.linear_solver': 'MA57',
                                'ipopt.check_derivatives_for_naninf': 'yes',
                                'ipopt.honor_original_bounds': 'yes',
                                'ipopt.linear_system_scaling': 'mc19',
                                # 'monitor': 'nlp_g'
                                }
    # mpc.settings.supress_ipopt_output()
    #############################


    mpc.set_objective(mterm=model.aux['mterm'], lterm=model.aux['lterm'])
    mpc.set_rterm(u=0.1)
    mpc.bounds['lower', '_u', 'u'] = v_min
    mpc.bounds['upper', '_u', 'u'] = v_max
    for i in range(model.n_isotherms):
        mpc.scaling['_x', f'width_{i}'] = 0.1
        mpc.bounds['lower', '_x', f'width_{i}'] = 0
        mpc.set_nl_cons(f'width_{i}_constr', model.isotherm_widths_mm[i], ub=50, soft_constraint=True)
    mpc.bounds['lower', '_z', 'deflection'] = 0
    mpc.scaling['_z', 'deflection'] = 0.1
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        tvp_template['_tvp'] = deflection_adaptation.b
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)
    mpc.u0 = v_min
    init_mpc = False
    for i in range(model.n_isotherms - 1):
        mpc.set_nl_cons(f'width_{i}_ordering_constr', model.isotherm_widths_mm[i] - model.isotherm_widths_mm[i+1], ub=0)
    mpc.setup()

    ## Setup plotting
    plotter = Plotter(mpc.data, isotherm_temps=model.isotherm_temps)

    real_or_virtual = input("Real or Virtual? (r/v): ").lower().strip()
    if real_or_virtual == "r":
        adaptive_velocity = input("Adaptive Velocity? (y/n): ").lower().strip()
        constant_velocity = None
        if adaptive_velocity == "n":
            constant_velocity = float(input("Enter constant velocity: "))
            print(f"Constant velocity: {constant_velocity} mm/s")
            experiment_type = "adaptive" if adaptive_velocity else f"{constant_velocity}mm-s"

        t3 = T3pro(port=0)
        tb = Testbed()
    else:
        t3 = None
        tb = None

    u0 = v_min
    if real_or_virtual == "v":
        data = pkl.load(open('logs/data_adaptive_2024-09-17-14:50.pkl', 'rb'))
        if isinstance(data, LoggingData):
            thermal_frames = data.thermal_frames
        else:
            thermal_frames = data['thermal_frames']

        for frame in thermal_frames:
            u0, init_mpc = mpc_loop(mpc, u0, frame, init_mpc)

    elif real_or_virtual == 'r':
        data_log = dict(widths_mm=[], velocities=[], deflections_mm=[], thermal_frames=[], damping_estimates=[])
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
                thermal_frame_to_color(thermal_arr)
                u0, init_mpc = mpc_loop(mpc, u0, thermal_arr, init_mpc)
                add_to_data_log(data_log, thermal_arr, model.isotherm_widths_mm,
                                tipPos.update_with_measurement(thermal_arr)[0] / thermal_px_per_mm,
                                deflection_adaptation,
                                adaptive_velocity,
                                0 if adaptive_velocity else constant_velocity)
                tb.set_speed(u0)

    plt.show()
    if tb is not None:
        tb.stop()
    if t3 is not None:
        t3.release()
    plt.ioff()
    cv.destroyAllWindows()
    if real_or_virtual == "r":
        date = datetime.now()
        with open(f"logs/data_{experiment_type}_{date.strftime('%Y-%m-%d-%H:%M')}.pkl", "wb") as f:
            pkl.dump(data_log, f)



