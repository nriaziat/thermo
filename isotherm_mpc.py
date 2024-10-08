import do_mpc
import numpy as np

A = 1
P = 10
rho = 1090e-9 # kg/mm^3
Cp = 3421 # J/kgK
dT = 10
Ta = 20
k = 0.49e-3 # W/mmK
alpha = k / (rho * Cp) # mm^2/s
print(alpha)

n=9
model = do_mpc.model.Model('continuous')
x = []
for i in range(n):
    x.append(model.set_variable(var_type='_x', var_name=f'x_{i}'))
ref = model.set_variable(var_type='_tvp', var_name='ref')
u = model.set_variable(var_type='_u', var_name='u')
L = 2 * alpha / u
S = 4 * alpha / u**2
T = np.arange(Ta + n * dT, Ta + dT, -dT)
print(f"Isotherm Levels: {T}")
model.set_rhs('x_0', (-2 * alpha / x[0]) * (x[1] / (x[1] - x[0])) + A * P * np.exp(-x[0]/L) / (np.pi * rho * Cp * dT * x[0]**2) * (1 + x[0]/L) \
        - (T[0] - Ta)/(S*dT) * (x[1] - x[0]))
for i in range(1, n-1):
    model.set_rhs(f'x_{i}', (-alpha / x[i]) * ((x[i + 1] / (x[i + 1] - x[i])) - (x[i - 1] / (x[i] - x[i - 1]))) - (T[i] - Ta) / (
                2 * S * dT) * (x[i + 1] - x[i - 1]))
model.set_rhs(f'x_{n-1}', (- alpha / x[-1]) * (x[-1] - 2*x[-2])/(x[-1] - x[-2]) - (x[-1] - x[-2]) / S)

model.set_expression(expr_name='cost', expr=(x[3])**2)
model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 10,
    'n_control': 10,
    'n_robust': 0,
    'open_loop': 0,
    't_step': 1/24,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)
mpc.set_objective(mterm=model.aux['cost'], lterm=model.aux['cost'])
mpc.set_rterm(u=0.1)
mpc.bounds['lower','_u', 'u'] = 1
mpc.bounds['upper','_u', 'u'] = 10
for i in range(n):
    mpc.scaling['_x', f'x_{i}'] = 1
    mpc.bounds['lower','_x', f'x_{i}'] = 1e-1

tvp_template = mpc.get_tvp_template()

def tvp_fun(t_now):
    tvp_template['_tvp'] = 0.25 * np.sin(2 * np.pi*t_now) + 1.5
    return tvp_template

mpc.set_tvp_fun(tvp_fun)

mpc.setup()


simulator = do_mpc.simulator.Simulator(model)
params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-8,
    'reltol': 1e-8,
    't_step': 1/24,
}
simulator.set_param(**params_simulator)
sim_tvp_template = simulator.get_tvp_template()
def tvp_fun(t_now):
    sim_tvp_template['ref'] = 0.25 * np.sin(2 * np.pi*t_now) + 1.5
    return sim_tvp_template
simulator.set_tvp_fun(tvp_fun)
simulator.setup()


mpc.x0 = np.array([(i+1) for i in range(n)])
mpc.set_initial_guess()

simulator.x0 = mpc.x0
x0 = simulator.x0
for k in range(50):
    u0 = mpc.make_step(x0)
    # print(u0)
    x0 = simulator.make_step(u0)

# plot results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(mpc.data['_x', f'x_{3}'], label='Isotherm')
plt.plot(mpc.data['_u', 'u'], label='u')
# plt.hlines(1, 0, 50, 'r', label='setpoint')
# plt.plot(0.25 * np.sin(2 * np.pi*np.arange(0, 50/24, 1/24)) + 1.5, 'r--', label='ref')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Isotherm Width [mm]')
plt.show()