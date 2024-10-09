import do_mpc
import numpy as np
import pickle as pkl
from utils import cv_isotherm_width as isotherm_width
import matplotlib.pyplot as plt

A = 1
P = 10
rho = 1090e-9 # kg/mm^3
Cp = 3421 # J/kgK
dT = 5
Ta = 20
k = 0.49e-3 # W/mmK
alpha = k / (rho * Cp) # mm^2/s

n=9
model = do_mpc.model.Model('continuous')
x = []
for i in range(n):
    x.append(model.set_variable(var_type='_x', var_name=f'x_{i}'))
u = model.set_variable(var_type='_u', var_name='u')
L = 2 * alpha / u
S = 4 * alpha / u**2
T = np.linspace(Ta + n * dT, Ta + dT, n)
print(f"Isotherm Levels: {T}")
model.set_rhs('x_0', (-2 * alpha / x[0]) * (x[1] / (x[1] - x[0])) + A * P * np.exp(-x[0]/L) / (np.pi * rho * Cp * dT * x[0]**2) * (1 + x[0]/L) \
        - (T[0] - Ta)/(S*dT) * (x[1] - x[0]))
for i in range(1, n-1):
    model.set_rhs(f'x_{i}', (-alpha / x[i]) * ((x[i + 1] / (x[i + 1] - x[i])) - (x[i - 1] / (x[i] - x[i - 1]))) - (T[i] - Ta) / (
                2 * S * dT) * (x[i + 1] - x[i - 1]))
model.set_rhs(f'x_{n-1}', (- alpha / x[-1]) * (x[-1] - 2*x[-2])/(x[-1] - x[-2]) - (x[-1] - x[-2]) / S)
model.set_expression(expr_name='cost', expr=(x[n//2])**2)
model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 10,
    'n_robust': 0,
    'open_loop': 0,
    't_step': 1/24,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA57 linear solver in ipopt for faster calculations:
    'nlpsol_opts': {'ipopt.linear_solver': 'MA57'}

}

mpc.set_param(**setup_mpc)
mpc.set_objective(mterm=model.aux['cost'], lterm=model.aux['cost'])
mpc.set_rterm(u=0.)
mpc.bounds['lower','_u', 'u'] = 1
mpc.bounds['upper','_u', 'u'] = 10
for i in range(n):
    mpc.scaling['_x', f'x_{i}'] = 1
    mpc.bounds['lower','_x', f'x_{i}'] = 1e-1

mpc.setup()

with open('../logs/data_adaptive_2024-09-17-14:43.pkl', 'rb') as f:
    data = pkl.load(f)

thermal_px_per_mm = 5.1337 # px/mm
thermal_frames = data.thermal_frames
init_state = False
for frame in thermal_frames:
    w = np.array([isotherm_width(frame, temp)[0]/thermal_px_per_mm for temp in T])
    if any(w == 0):
        continue
    if not init_state:
        mpc.x0 = np.array([(i + 1) for i in range(n)])
        mpc.set_initial_guess()
        init_state = True
    u0 = mpc.make_step(w)
    # plt.cla()
    # plt.contourf(frame, levels=T[::-1])
    # plt.pause(0.1)

# plot results
plt.figure()
plt.plot(mpc.data['_x', f'x_{n//2}'], label='Isotherm')
plt.plot(mpc.data['_u', 'u'], label='u')
# plt.hlines(1, 0, 50, 'r', label='setpoint')
# plt.plot(0.25 * np.sin(2 * np.pi*np.arange(0, 50/24, 1/24)) + 1.5, 'r--', label='ref')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Isotherm Width [mm]')
plt.show()