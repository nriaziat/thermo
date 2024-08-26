import numpy as np
import cv2 as cv
from casadi import fabs
from functools import lru_cache
import cmapy
from dataclasses import dataclass, field
import do_mpc
import pickle as pkl
import matplotlib.pyplot as plt

def thermal_frame_to_color(thermal_frame):
    norm_frame = cv.normalize(thermal_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return cv.applyColorMap(norm_frame, cmapy.cmap('hot'))

def point_in_ellipse(x, y, ellipse) -> bool:
    """
    Check if a point is inside the ellipse
    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :param ellipse: Ellipse parameters (center, axes, angle)
    :return: True if the point is inside the ellipse
    """
    a, b = ellipse[1]
    cx, cy = ellipse[0]
    theta = np.deg2rad(ellipse[2])
    x = x - cx
    y = y - cy
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    return (x1 / a) ** 2 + (y1 / b) ** 2 <= 1


def isotherm_width(t_frame: np.array, t_death: float) -> int:
    """
    Calculate the width of the isotherm
    :param t_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :return: Width of the isotherm [px]
    """
    return np.max(np.sum(t_frame > t_death, axis=0))


def cv_isotherm_width(t_frame: np.ndarray, t_death: float) -> (float, tuple | None):
    """
    Calculate the width of the isotherm using ellipse fitting
    :param t_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :return: Width of the isotherm [px] and the ellipse
    """
    binary_frame = (t_frame > t_death).astype(np.uint8)
    blur_frame = cv.medianBlur(binary_frame, 5)
    contours = cv.findContours(blur_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = contours[0]
    list_of_pts = []
    for ctr in contours:
        if cv.contourArea(ctr) > 100:
            list_of_pts += [pt[0] for pt in ctr]

    ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
    hull = cv.convexHull(ctr)
    if hull is None or len(hull) < 5:
        return 0, None
    # if cv.contourArea(hull) < 1:
    #     return 0, None
    ellipse = cv.fitEllipse(hull)
    w = ellipse[1][0] / 2
    return w, ellipse


def find_tooltip(therm_frame: np.ndarray, t_death, last_tool_tip, ellipse=None) -> tuple | None:
    """
    Find the location of the tooltip
    :param therm_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :param last_tool_tip: Last known tool tip location
    :param ellipse: Isotherm ellipse (optional)
    :return: x, y location of the tooltip [px]
    """

    if (therm_frame > t_death).any():
        return np.unravel_index(np.argmax(therm_frame), therm_frame.shape)
        # top_temps = therm_frame > t_death
        # corners = cv.cornerHarris(top_temps.astype(np.uint8), 2, 3, 0.04)
        # corners = cv.dilate(corners, None)
        # corners = corners > 0.01 * corners.max()
        # # return coordinate of right-most true value
        # coordinates = np.where(corners)
        # right_most = np.argmax(coordinates[1])
        # tip = (coordinates[0][right_most], coordinates[1][right_most])
        # return tip
    else:
        return None

exp_gamma = np.exp(0.5772)
@lru_cache
def F(Tc):
    return np.exp(-Tc) * (1 + (1.477 * Tc) ** -1.407) ** 0.7107

# @lru_cache
def ymax(alpha, u, Tc):
    return fabs(4 * alpha / (u * exp_gamma) * F(Tc))

class Plotter:
    def __init__(self, mpc_data: do_mpc.data.Data):
        self.fig, self.axs = plt.subplots(6, sharex=False, figsize=(16, 9))
        for i in range(1, len(self.axs)-1):
            self.axs[i].sharex(self.axs[0])
            self.axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.line_plots = []

        self.graphics = do_mpc.graphics.Graphics(mpc_data)
        self.graphics.add_line(var_type='_x', var_name='width', axis=self.axs[0])
        self.graphics.add_line(var_type='_tvp', var_name='width_estimate', axis=self.axs[0])
        self.graphics.add_line(var_type='_tvp', var_name='deflection_measurement', axis=self.axs[1])
        self.graphics.add_line(var_type='_z', var_name='deflection', axis=self.axs[1])
        self.graphics.add_line(var_type='_u', var_name='u', axis=self.axs[2])
        self.graphics.add_line(var_type='_tvp', var_name='alpha', axis=self.axs[3])
        self.graphics.add_line(var_type='_tvp', var_name='a', axis=self.axs[4])
        self.graphics.add_line(var_type='_tvp', var_name='d', axis=self.axs[5])

        self.axs[0].set_ylabel(r'$w~[\si[per-mode=fraction]{\milli\meter}]$')
        self.axs[1].set_ylabel(r"$\delta~[\si[per-mode=fraction]{\milli\meter}]$")
        self.axs[2].set_ylabel(r"$u~[\si[per-mode=fraction]{\milli\meter\per\second}]$")
        self.axs[3].set_ylabel(r'$\hat{\alpha}$')
        self.axs[4].set_ylabel(r'$\hat{a}$')
        self.axs[5].set_ylabel(r'$\hat{d}$')
        self.axs[5].set_xlabel('Time Step')
        self.fig.align_ylabels()


    def plot(self, t_ind=None):
        if t_ind is None:
            self.graphics.plot_results()
            self.graphics.plot_predictions()
            self.graphics.reset_axes()
        else:
            self.graphics.plot_results(t_ind)
            self.graphics.plot_predictions(t_ind)
            self.graphics.reset_axes()


@dataclass
class LoggingData:
    widths_mm: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    deflections_mm: list[float] = field(default_factory=list)
    thermal_frames: list[np.ndarray] = field(default_factory=list)
    positions_mm: list[float] = field(default_factory=list)
    damping_estimates: list[float] = field(default_factory=list)
    a_hats: list[float] = field(default_factory=list)
    alpha_hats: list[float] = field(default_factory=list)
    width_estimates: list[float] = field(default_factory=list)
    deformations: list[float] = field(default_factory=list)

    def __iter__(self):
        return iter(zip(self.velocities, self.widths_mm, self.deflections_mm, self.thermal_frames, self.positions_mm, self.damping_estimates))

    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

