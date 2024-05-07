import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import control as ct
import control.flatsys as fs

# Parameters
k = 0.19  # thermal conductivity
dx = 0.1  # distance between nodes
dt = 0.01  # time step
N = 2000  # number of nodes
h = 10  # electrocautery heat
sigma = 0.000001  # cautery heat spread
q = lambda x, xo: np.exp(-(x - xo) ** 2 / (sigma ** 2))
Kp = 1  # proportional gain

# State Space Form
a = k / (dx ** 2)
b = -2 * k / (dx ** 2)

A = np.zeros((N, N))  # state matrix
row = np.zeros((1, N))
row[0, 0] = a
row[0, 1] = b
row[0, 2] = a
for i in range(1, N - 1):
    A[i, :] = np.roll(row, i - 1)
A[0, :] = np.roll(row, -1)
A[0, -1] = 0
A[-1, :] = np.roll(row, N - 2)
A[-1, 0] = 0

B = np.eye(N)  # input matrix

# Initial Conditions
y0 = np.ones((N, 1)) * 25
yd = np.ones((N, 1)) * 100
dyd = np.zeros((N, 1))
laser_pos = 0
u = np.zeros((N, 1))

# boundary conditions
phi0 = phiN = 25

fig, axs = plt.subplots(2, 1)
vels = []
laser_traj = []


# def forward(x, u, params=None):
#     if params is None:
#         params = {}
#     laser_pos = params.get('laser_pos')
#     zflag = np.zeros((N, 2))
#     zflag[:, 0] = x
#     v = h * q(np.arange(N), 0) + u
#     v[0] += a * phi0
#     v[-1] += a * phiN
#     zflag[:, 1] = A @ x + B @ v
#     return zflag
#
#
# def reverse(z, params=None):
#     if params is None:
#         params = {}
#     z = np.array(z)
#     x = z[:, 0]
#     u = z[:, 1] - h * q(np.arange(N), 0)
#     u[0] -= a * phi0
#     u[-1] -= a * phiN
#     return x, u
#
#
# flat_sys = fs.FlatSystem(forward, reverse, inputs=[f'u{i}' for i in range(N)],
#                          states=[f'x{i}' for i in range(N)],
#                          outputs=[f'x{i}' for i in range(N)])
# traj = fs.point_to_point(flat_sys, 10, y0, u, yd, np.zeros((N,1)))
# t = np.linspace(0, 10, 100)
# y, u = traj.eval(t)
#
# axs[0].plot(y[:, -1])
# #
# # axs[1].plot(t, u)
# plt.show()

# Simulation
y = y0
while np.max(y) < 99:
    axs[0].cla()
    axs[1].cla()
    # axs[2].cla()
    axs[0].set_ylim(0, 120)

    # control
    v = np.zeros((N, 1))
    dphidx = np.diff(y, axis=0, prepend=0) / dx
    v[0] = dyd[0] - b * y[0] - a * y[1] - Kp * (y[0] - yd[0])
    v[-1] = dyd[-1] - a * y[-2] - b * y[-1] - Kp * (y[-1] - yd[-1])
    u[0] = v[0] - a * phi0 - h * q(0, laser_pos)
    u[-1] = v[-1] - a * phiN - h * q(N - 1, laser_pos)
    for i in range(1, N - 1):
        v[i] = dyd[i] - a * y[i - 1] - b * y[i] - a * y[i + 1] - Kp * (y[i] - yd[i])
        u[i] = v[i] - h * q(i, laser_pos)

    vel = np.linalg.lstsq(dphidx, u, rcond=None)[0].item()
    vels.append(vel)
    laser_pos += vel * dt
    laser_traj.append(laser_pos)

    # simulate
    y += dt * (A @ y + B @ v)

    axs[0].plot(y)
    axs[0].vlines(laser_pos, 0, 120, colors='r')
    axs[1].plot(vels)
    plt.pause(0.01)

plt.show()
