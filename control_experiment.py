import matplotlib.pyplot as plt
from T3pro import T3pro
from testbed import Testbed
from models import *
from AdaptiveID import *
from matplotlib import rcParams
from utils import *

thermal_px_per_mm = 5.1337 # px/mm
qw = 1  # width cost
qd = 1 # deflection cost
r = 0.01  # control change cost
v_min = 1  # minimum velocity [mm/s]
v_max = 10  # maximum velocity [mm/s]

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
    model = MultiIsothermModel(n_isotherms=5)
    model.set_cost_function(qw, qd)
    model.setup()
    tipPos = ToolTipKF(0.7)

    ## Initialize the parameter adaptation
    thermal_adaptation = ScalarFirstOrderAdaptation(x0=0, a0=0.1, b0=0.1)
    deflection_adaptation = ScalarLinearAlgabraicAdaptation(b=0.1)

    ## Initialize the MPC controller
    mpc = do_mpc.controller.MPC(model=model)

    ##############################
    mpc.settings.n_horizon = 10
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
                                # 'monitor': 'nlp_g'
                                }
    # mpc.settings.supress_ipopt_output()
    #############################


    mpc.set_objective(mterm=model.aux['mterm'], lterm=model.aux['lterm'])
    mpc.set_rterm(u=0.)
    mpc.bounds['lower', '_u', 'u'] = v_min
    mpc.bounds['upper', '_u', 'u'] = v_max
    for i in range(model.n_isotherms):
        mpc.scaling['_x', f'width_{i}'] = 1
        mpc.bounds['lower', '_x', f'width_{i}'] = 2
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
        mpc.set_nl_cons(f'width_{i}_constr', model.isotherm_widths[i] - model.isotherm_widths[i+1], ub=0)
    mpc.setup()

    ## Setup plotting
    plotter = Plotter(mpc.data)

    real_or_virtual = input("Real or Virtual? (r/v): ").lower().strip()
    if real_or_virtual == "r":
        adaptive_velocity = input("Adaptive Velocity? (y/n): ").lower().strip()
        constant_velocity = None
        if adaptive_velocity == "n":
            constant_velocity = float(input("Enter constant velocity: "))
            print(f"Constant velocity: {constant_velocity} mm/s")


        t3 = T3pro(port=0)
        tb = Testbed()
    else:
        t3 = None
        tb = None

    if real_or_virtual == "v":
        data = pkl.load(open('logs/data_adaptive_2024-09-17-14:43.pkl', 'rb'))
        if isinstance(data, LoggingData):
            thermal_frames = data.thermal_frames
            velocities = data.velocities
        else:
            thermal_frames = data['thermal_frames']
            velocities = data['velocities']

        u0 = v_min
        for frame, v in zip(thermal_frames, velocities):
            defl_px, _ = tipPos.update_with_measurement(frame)
            defl_mm = defl_px / thermal_px_per_mm
            w = np.array([cv_isotherm_width(frame, temp)[0] / thermal_px_per_mm for temp in model.isotherm_temps])
            deflection_adaptation.update(defl_mm, u0)
            if any(w < 1) or len(w) != len(set(w)):
                continue
            if not init_mpc:
                mpc.x0 = w
                mpc.set_initial_guess()
                init_mpc = True
            u0 = mpc.make_step(w).item()
            plotter.plot()
            plt.pause(0.0001)

    plt.show()
    if tb is not None:
        tb.stop()
    plt.ioff()