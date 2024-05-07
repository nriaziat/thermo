import numpy as np
import importlib.util
import do_mpc
from casadi import *

# Parameters
Tmin = 273.15  # [K]
Tmax = 423.15  # [K]
L = 1
k = 0.59  # thermal conductivity [W/(kg*K)]
alpha = 0.14
cp = 4  # specific heat capacity [kJ/(kg*K)]
dHc = 250  # heat of denaturation [kJ/kg]
rho = 700  # density [kg/m^3]
beta = cp * (Tmax - Tmin) / dHc  # phase change dynamics
th_0 = 0.5  # initial non-dimensional temperature
s_0 = 0.5  # initial non-dimensional phase change boundary
Tc = 355.15  # phase change temperature [K]
th_c = (Tc - Tmin) / (Tmax - Tmin)  # non-dimensional phase change temperature
r_c = 10e-3  # cutoff radius [m]

assert dHc / cp < Tc, 'Phase change temperature is too low'


def qpp(r, P=15):
    eta = 1  # fraction of power absorbed
    a = 0.005  # spread of power [m]
    return eta * P / (2 * np.pi * a) * np.exp(-(1 / (4 * a ** 2) * r ** 2))


# MPC

model_type = 'continuous'
model = do_mpc.model.Model(model_type)

# Variables
theta = model.set_variable(var_type='_x', var_name='theta')
s = model.set_variable(var_type='_x', var_name='s')
u = model.set_variable(var_type='_u', var_name='u')

model.set_rhs('theta', jacobian(theta, ))
