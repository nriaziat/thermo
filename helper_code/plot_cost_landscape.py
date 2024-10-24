import matplotlib.pyplot as plt
import numpy as np
from models import ymax, Tc,humanTissue, hydrogelPhantom
from scipy.optimize import minimize_scalar

def cost_function(v, P, c):
    qw = 1
    qd = 1
    return qw * ymax(humanTissue.alpha, v, Tc(40, humanTissue, P))**2 + qd + (25 * np.exp(-c / v))**2

def plot_cost_landscape():
    p = np.linspace(0.0, 100, 100)
    c = np.linspace(0.0, 1, 100)
    P, C = np.meshgrid(p, c)

    Z = np.zeros_like(P)
    for i in range(len(p)):
        for j in range(len(c)):
            Z[j, i] = minimize_scalar(cost_function, args=(p[i], c[j]), bounds=[0.0, 10], method='bounded').x
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, C, Z, cmap='viridis')
    ax.set_xlabel('P')
    ax.set_ylabel('d')
    ax.set_zlabel('Velocity')
    plt.show()

plot_cost_landscape()


