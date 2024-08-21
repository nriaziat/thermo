from ThermalProcessing import AdaptiveMPC
import do_mpc
import matplotlib.pyplot as plt

plt.ion()
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'

# Create a ThermalController object
tc = AdaptiveMPC()
mpc_graphics = do_mpc.graphics.Graphics(tc.mpc.data)
sim_graphics = do_mpc.graphics.Graphics(tc.simulator.data)

fig, ax = plt.subplots(2, sharex=True, figsize=(16, 9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot the state on axis 1 to 4:
    g.add_line(var_type='_x', var_name='width', axis=ax[0], color='#1f77b4')

    # Plot the control input on axis 5:
    g.add_line(var_type='_u', var_name='u', axis=ax[4], color='#1f77b4')

ax[0].set_ylabel('Width')
ax[1].set_ylabel('u')
ax[1].set_xlabel(r'$t~[\si[per-mode=fraction]{\minute}]$')

n_steps = 100
x_next = tc.x0
for i in range(n_steps):
    # Simulate the MPC controller
    u0 = tc.mpc.make_step(x_next)
    x_next = tc.simulator.make_step(u0)

from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
# The function describing the gif:
def update(t_ind):
    sim_graphics.plot_results(t_ind)
    mpc_graphics.plot_predictions(t_ind)
    mpc_graphics.reset_axes()


if True:
    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
    gif_writer = ImageMagickWriter(fps=10)
    anim.save('simulated_electrocautery.gif', writer=gif_writer)
