import numpy as np
import cv2 as cv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


gaus_kernel = cv.getGaussianKernel(3, 0)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
    return (x1 / a) ** 2 + (y1 / b) ** 2

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
    w = ellipse[1][0]
    return w, ellipse


def find_tooltip(therm_frame: np.ndarray, t_death, neutral_tip_pos, ellipse=None) -> tuple | None:
    """
    Find the location of the tooltip
    :param therm_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :param neutral_tip_pos: Neutral position of the tool tip [px]
    :param ellipse: Isotherm ellipse (optional)
    :return: x, y location of the tooltip [px]
    """
    # tip_mask = np.ones_like(therm_frame).astype(np.uint8)
    # tip_mask[neutral_tip_pos] = 0

    if (therm_frame > t_death).any():
        # gaus = cv.filter2D(therm_frame, cv.CV_64F, gaus_kernel)
        norm16 = cv.normalize(therm_frame, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)
        cl1 = clahe.apply(norm16)
        norm_frame = cv.normalize(cl1, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # weight_mask = np.ones_like(therm_frame)
        # weighted_frame = cv.multiply(norm_frame, weight_mask, dtype=cv.CV_8U)
        tip = np.unravel_index(np.argmax(norm_frame), norm_frame.shape)
        if ellipse is not None:
            # find the closest point on the ellipse to the tip
            dist = point_in_ellipse(tip[1], tip[0], ellipse)
            if dist < 0.05:
                print("Tip near center of ellipse")

        return tip
    else:
        return None


class ThermalController:
    Kp = 0.2
    # Ki = 0.0005
    Ki = 0
    Kd = 0.0
    Kf = 0.4
    K_adaptive_deflection = 0.005
    K_adaptive_width = 0.01
    importance_weight = 0.5
    width_scale = 15

    def __init__(self,
                    des_width: float,
                    v_min: float = 1,
                    v_max: float = 10,
                    max_accel: float = 10,
                    v0: float = 1):
        """

        """
        self._v_min = v_min
        self._v_max = v_max
        self._max_accel = max_accel
        self.des_width = des_width
        self.v = v0
        self._error = 0
        self._last_error = 0
        self._error_sum = 0
        self.width = 0
        self.tool_damping_estimate = 0.1
        self.width_constant_estimate = 1

    def compute_error(self, deflection: float, width: float, importance_weight: float):
        """
        Compute the error based on the deflection and width
        :param deflection: Tool deflection [mm]
        :param width: Isotherm width [px]
        :param importance_weight: Weight of the width component (0-1)
        """
        assert 0 <= importance_weight <= 1
        self._last_error = self._error
        width_error = width - self.des_width if width > self.des_width else 0
        error = (1-importance_weight) * width_error + importance_weight * (-self.width_scale * deflection)
        self._error = width_error

    def compute_dv(self, derror: float) -> float:
        """
        Compute the change in velocity using PID control
        :param derror: Current error rate
        :return: Change in velocity
        """
        deflection_error = -self.tool_damping_estimate * self.v
        dv = self.Kp * self._error + self.Ki * self._error_sum - self.Kd * derror + self.Kf * deflection_error
        if dv > self._max_accel:
            dv = self._max_accel
        return dv

    def estimate_tool_damping(self, v, deflection: float) -> None:
        """
        Estimate the tool damping based on the deflection
        :param deflection: Tool deflection [mm]
        """
        deflection_estimate =  self.tool_damping_estimate * v
        error = deflection_estimate - deflection
        self.tool_damping_estimate += -self.K_adaptive_deflection * v * error

    def estimate_width_constant(self, v, width: float) -> None:
        """
        Estimate the width constant based on the isotherm width
        :param width: Isotherm width
        """
        width_estimate = self.width_constant_estimate / v
        error = width_estimate - width
        self.width_constant_estimate += -(self.K_adaptive_width / v) * error

    def enforce_pid_limits(self, v: float, error_sum: float) -> tuple[float, float]:
        """
        Enforce velocity saturation limits and integration overflow limits.
        :param v: Current tool speed
        :param error_sum: Current error sum
        :return: New tool speed, new error sum
        """
        if v < self._v_min:
            v = self._v_min
            if self._error > 0:
                error_sum += self._error
        elif v > self._v_max:
            v = self._v_max
            if self._error < 0:
                error_sum += self._error
        else:
            error_sum += self._error
        if np.sign(self._error) != np.sign(self._last_error):
            error_sum = 0
        return v, error_sum

    def find_optimal_speed(self, w1: float, w2: float) -> float:
        """
        Find the optimal tool speed based on the estimated parameters using minimization
        :param w1: Weight of the width component
        :param w2: Weight of the deflection component
        """
        v = ((w1/w2) ** 0.25) * (self.width_constant_estimate / self.tool_damping_estimate) ** 0.5
        if np.isnan(v):
            v = ((w1/w2) ** 0.25) * (-self.width_constant_estimate / self.tool_damping_estimate) ** 0.5
        v = np.clip(v, self._v_min, self._v_max)
        return v


    def update(self, v, deflection: float, width) -> float:
        """
        Update the tool speed based on the current state
        :param v: Current tool speed
        :param deflection: Tool deflection state
        :param width: Isotherm width
        :return: New tool speed
        """
        self.width = width
        self.estimate_tool_damping(v, deflection)
        self.estimate_width_constant(v, width)
        self.v = self.find_optimal_speed(1, 10)

        return self.v

    def get_loggable_data(self) -> dict:
        """
        Get the data that can be logged
        :return: dictionary of data
        """
        return {
            "v": self.v,
            "error": self._error,
            "error_sum": self._error_sum,
            "width": self.width,
            "tool_damping": self.tool_damping_estimate
        }



class OnlineVelocityOptimizer:

    def __init__(self,
                 des_width: float = 1,
                 v_min: float = 0.25,
                 v_max: float = 10,
                 v0: float = 1,
                 t_death: float = 50,
                 neutral_tip_pos: tuple = (183, 355)):
        """
        :param des_width: Desired Isotherm width [px]
        :param v_min: minimum tool speed [mm/s]
        :param v_max: maximum tool speed [mm/s]
        :param v0: Starting tool speed [mm/s]
        :param t_death: Isotherm temperature [C]
        """

        self.thermal_controller: ThermalController = ThermalController(des_width=des_width, v_min=v_min, v_max=v_max, v0=v0)
        self._dt = 1/24
        self.t_death = t_death
        self._width_kf = KalmanFilter(dim_x=2, dim_z=1)
        self._width_kf.x = np.array([0, 0])
        self._width_kf.F = np.array([[1, 1], [0, 1]])
        self._width_kf.H = np.array([[1, 0]])
        self._width_kf.P *= 10
        self._width_kf.R = 1
        self._width_kf.Q = Q_discrete_white_noise(dim=2, dt=0.05, var=50)

        self._pos_kf_init = False
        self._pos_init = None
        self._deflection = 0
        self._pos_kf = None

        self.ellipse = None

        self._tool_stiffness = 0.8705 # N/mm

        self.neutral_tip_pos = neutral_tip_pos

    @property
    def width(self):
        return self.thermal_controller.width

    @property
    def pid_velocity(self):
        return self.thermal_controller.v

    @property
    def deflection(self):
        return self._deflection

    @property
    def tool_tip_pos(self):
        """
        Returns the current position of the tool tip (x, y)
        """
        if self._pos_kf is None:
            return None
        return self._pos_kf.x[0], self._pos_kf.x[2]


    def init_pos_kf(self, pos: tuple):
        """
        Initialize the position Kalman filter
        :param pos: Initial position of the tool tip
        """
        damping_ratio = 0.7
        self._pos_kf = KalmanFilter(dim_x=4, dim_z=2)
        self._pos_kf.x = np.array([pos[0], 0, pos[1], 0])
        self._pos_kf.F = np.array([[1, self._dt, 0, 0],
                                   [0, damping_ratio, 0, 0],
                                   [0, 0, 1, self._dt],
                                   [0, 0, 0, damping_ratio]])
        self._pos_kf.H = np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0]])
        self._pos_kf.P *= 1
        self._pos_kf.R = 10 * np.eye(2)
        self._pos_kf.Q = Q_discrete_white_noise(dim=4, dt=self._dt, var=9)

    def update_tool_deflection(self, frame: np.ndarray[float]) -> tuple[float, float]:
        """
        Update the tool deflection based on the current frame
        :param frame: Temperature field from the camera.
        :return: deflection [px], deflection rate [px/s]
        """
        tool_tip = find_tooltip(frame, self.t_death, self.neutral_tip_pos, self.ellipse)
        if tool_tip is None:
            return 0, 0
        elif not self._pos_kf_init:
            self.init_pos_kf(tool_tip)
            self.neutral_tip_pos= np.array(tool_tip)
            self._pos_kf_init = True
        self._pos_kf.predict()
        self._pos_kf.update(tool_tip)
        deflection = np.array([self._pos_kf.x[0] - self.neutral_tip_pos[0], self._pos_kf.x[2] - self.neutral_tip_pos[1]])
        ddeflection = np.array([self._pos_kf.x[1], self._pos_kf.x[3]])
        return np.linalg.norm(deflection), np.linalg.norm(ddeflection) * np.sign(np.dot(deflection, ddeflection))

    def reset_tool_deflection(self):
        """
        Reset the tool deflection state
        """
        self._pos_kf_init = False


    def update_velocity(self, v: float, frame: np.ndarray[float], deflection) -> any:
        """
        Update the tool speed based on the current frame
        :param v: Current tool speed
        :param frame: Temperature field from the camera. If None, the field will be predicted using the current model
        :param deflection: Tool deflection [px]
        :return: new tool speed, ellipse of the isotherm if using CV
        """

        self._deflection = deflection

        z, self.ellipse = cv_isotherm_width(frame, self.t_death)

        self._width_kf.predict()
        self._width_kf.update(z)
        width = self._width_kf.x[0]
        v = self.thermal_controller.update(v, deflection, width)
        return v, self.ellipse