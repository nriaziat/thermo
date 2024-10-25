import numpy as np
import cv2 as cv
import cmapy
from dataclasses import dataclass, field
import pickle as pkl


def thermal_frame_to_color(thermal_frame):
    norm_frame = cv.normalize(thermal_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return cv.applyColorMap(norm_frame, cmapy.cmap('hot'))

def draw_info_on_frame(frame, deflection, width, velocity, tool_tip_pos):
    cv.putText(frame, f"Deflection: {deflection:.2f} mm",
           (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(frame, f"Width: {width:.2f} mm",
               (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(frame, f"Velocity: {velocity:.2f} mm/s",
               (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if tool_tip_pos is not None:
        cv.circle(frame, (int(tool_tip_pos[0]), int(tool_tip_pos[1])), 3, (0, 255, 0), -1)
    return frame

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
    blur_frame = cv.GaussianBlur(t_frame, (5, 5), 0)
    binary_frame = (blur_frame > t_death).astype(np.uint8)
    contours, hierarchy = cv.findContours(binary_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0, None
    ctr = max(contours, key=cv.contourArea)

    hull = cv.convexHull(ctr)
    if hull is None or len(hull) < 5:
        return 0, None
    ellipse = cv.fitEllipse(hull)
    w = min(*ellipse[1]) / 2
    return w, ellipse

def find_tooltip(therm_frame: np.ndarray, t_death) -> tuple | None:
    """
    Find the location of the tooltip
    :param therm_frame: Temperature field [C]
    :param t_death: Iotherm temperature [C]
    :return: x, y location of the tooltip [px]
    """

    if (therm_frame > t_death).any():
        # return np.unravel_index(np.argmax(therm_frame), therm_frame.shape)
        top_temps = therm_frame > t_death
        corners = cv.cornerHarris(top_temps.astype(np.uint8), 2, 3, 0.04)
        corners = cv.dilate(corners, None)
        corners = corners > 0.01 * corners.max()
        # return coordinate of corner-most true value
        coordinates = np.where(corners)
        left_most = np.argmax(coordinates[1])
        tip = (coordinates[0][left_most], coordinates[1][left_most])
        return tip
    else:
        return None

def find_wavefront_distance(ellipse: tuple, tool_tip: tuple) -> float:
    """
    Find the distance from the tool tip to the wavefront
    :param ellipse: Isotherm ellipse parameters (center, axes, angle)
    :param tool_tip: Tool tip location
    :return: Distance from the tool tip to the wavefront [px]
    """
    # find leading edge (furthest along the major axis) of the rotated ellipse
    tool_tip = np.array([tool_tip[1], tool_tip[0]])
    angle = ellipse[2] * np.pi / 180
    if angle > np.pi / 2:
        angle += np.pi / 2
    else:
        angle -= np.pi / 2
    r_major = max(ellipse[1]) / 2
    ellipse_tip = (ellipse[0] - np.array([np.cos(angle) * r_major, np.sin(angle) * r_major]),
                   ellipse[0] + np.array([np.cos(angle) * r_major, np.sin(angle) * r_major]))
    ellipse_tip = min(ellipse_tip, key=lambda x: np.linalg.norm(tool_tip - x))
    x_dir = np.sign(np.dot(ellipse_tip - tool_tip, np.array([1, 0])))
    return x_dir * np.linalg.norm(tool_tip - ellipse_tip)



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

