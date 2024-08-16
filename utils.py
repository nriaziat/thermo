import numpy as np
import cv2 as cv
from casadi import fabs
from functools import lru_cache

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
        top_temps = therm_frame > t_death
        corners = cv.cornerHarris(top_temps.astype(np.uint8), 2, 3, 0.04)
        corners = cv.dilate(corners, None)
        corners = corners > 0.01 * corners.max()
        # return coordinate of right-most true value
        coordinates = np.where(corners)
        right_most = np.argmax(coordinates[1])
        tip = (coordinates[0][right_most], coordinates[1][right_most])
        if ellipse is not None:
            # find the closest point on the ellipse to the tip
            if point_in_ellipse(tip[1], tip[0], ellipse):
                dist = np.linalg.norm(np.array(tip) - np.array(ellipse[0]))
                if dist < 0.25:
                    print("tip near center")

        return tip
    else:
        return None

exp_gamma = np.exp(0.5772)

@lru_cache
def F(Tc):
    return np.exp(-Tc) * (1 + (1.477 * Tc) ** -1.407) ** 0.7107

# @lru_cache
def ymax(alpha, u, Tc):
    return fabs(4 * alpha / (u * exp_gamma) * F(Tc))



