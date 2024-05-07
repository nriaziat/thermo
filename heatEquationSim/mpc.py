import importlib.util
import do_mpc
from casadi import *
import numpy as np
from copy import deepcopy

model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# Parameters
alpha = 0.14
k = 0.19
Q = 100
dx = dy = 0.25
dt = 0.1
Fo = alpha * dt / dx ** 2
W = 200
H = 1
Hd = 0.1

if 1 - 4 * Fo < 0:
    print('Unstable')
    exit(1)

Cv = alpha * dt / k
Kv = dt / (2 * dx)

temps = model.set_variable(var_type='_x', var_name='node_temps', shape=(W * H, 1))
inputs = model.set_variable(var_type='_u', var_name='inputs', shape=(W*H, 1))
measurements = model.set_variable(var_type='_x', var_name='measurements', shape=(W * H, 1))


a = k / dx ** 2
b = -2 * k / dx ** 2

A = np.zeros((W * H, W * H))
row = np.zeros((1, W * H))
row[0, 0] = a
row[0, 1] = b
row[0, 2] = a
for i in range(1, W * H - 1):
    A[i,:] = np.roll(row, i-1)
A[0, :] = np.roll(row, -1)
A[0, -1] = 0
A[-1, :] = np.roll(row, W*H-2)
A[-1, 0] = 0

B = np.eye(W * H)
C = np.eye(W * H)

model.set_rhs('node_temps', A @ temps + B @ inputs)
model.set_expression('cost', sum1((temps - 100) ** 2))
model.set_rhs('measurements', C @ temps)
model.setup()

xss = np.ones((W * H, 1)) * 25
uss = np.zeros((1, 1))
# model = do_mpc.model.linearize(model, x0=xss, u0=uss)

mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_robust': 0,
    'n_horizon': 5,
    't_step': dt,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 3,
    'collocation_ni': 1,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    'nlpsol_opts': {'ipopt.linear_solver': 'mumps'}
}
mpc.set_param(**setup_mpc)

mterm = model.aux['cost']
lterm = model.aux['cost']
mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(inputs=1e-4)

# mpc.bounds['lower', '_u', 'speed'] = -1
# mpc.bounds['upper', '_u', 'speed'] = 1
# mpc.bounds['upper', '_x', 'node_temps'] = 125
# mpc.bounds['lower', '_x', 'node_temps'] = 0
# mpc.bounds['lower', '_x', 'position'] = 0
# mpc.bounds['upper', '_x', 'position'] = W

mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=dt)
simulator.setup()

# init
T_0 = 25
x0 = np.ones((2*W * H, 1)) * T_0

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0
mpc.set_initial_guess()

import matplotlib.pyplot as plt

plt.ion()
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = "\n".join([r'\usepackage{amsmath}', r'\usepackage{siunitx}'])
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

fig, ax = plt.subplots(2, sharex=True, figsize=(16, 9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot the state on axis 1 to 4:
    g.add_line(var_type='_x', var_name='node_temps', axis=ax[0], color='#1f77b4')
    # g.add_line(var_type='_x', var_name='peak_temps', axis=ax[1], color='#ff7f0e')
    # g.add_line(var_type='_x', var_name='position', axis=ax[1], color='#1f77b4')
    # Plot the control input on axis 5:
    g.add_line(var_type='_u', var_name='inputs', axis=ax[1], color='#1f77b4')

ax[0].set_ylabel(r'$T$')
# ax[1].set_ylabel(r'$T_{\text{peak}}$')
# ax[1].set_ylabel(r'$x$')
ax[1].set_ylabel(r'$u$')
ax[1].set_xlabel(r'$t~[\si[per-mode=fraction]{\minute}]$')

n_steps = 100
for k in range(n_steps):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter


# The function describing the gif:
def update(t_ind):
    sim_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()


if True:
    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
    gif_writer = ImageMagickWriter(fps=10)
    anim.save('anim_heat_mpc.gif', writer=gif_writer)
