import numpy as np
# from scipy.optimize import minimize, curve_fit
# from scipy.special import kn
import cv2 as cv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Q = 500  # J / mm^2 ??
# k = 0.49e-3  # W/(mm*K)
# To = 23  # C
# rho = 1090e-9  # kg/m^3
# cp = 3421  # J/(kg*K)
# alpha = k / (rho * cp)  # mm^2/s
# a = Q / (2 * np.pi * k)
# b = 1 / (2 * alpha)
#
# x_len = 50  # mm
# x_res = 384  # pixels
# y_res = 288
# cam_mm_per_px = x_len / x_res  # mm per pixel
# y_len = y_res * cam_mm_per_px
#
# ys = np.linspace(-y_len / 2, y_len / 2, y_res)
# xs = np.linspace(-x_len / 2, x_len / 2, x_res)
# grid = np.meshgrid(xs, ys)

gaus_kernel = cv.getGaussianKernel(3, 0)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
        if cv.contourArea(ctr) > 50:
            list_of_pts += [pt[0] for pt in ctr]

    ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
    hull = cv.convexHull(ctr)
    if hull is None or len(hull) < 5:
        return 0, None
    ellipse = cv.fitEllipse(hull)
    w = ellipse[1][0]
    return w, ellipse


# def temp_field_prediction(xi, y, u, alph, beta, To) -> np.ndarray:
#     """
#     Predict the temperature field using the given parameters
#     :param xi: x location relative to the tool [mm]
#     :param y: y location relative to the tool [mm]
#     :param u: tool speed [mm/s]
#     :param alph: lumped parameter 1
#     :param beta: lumped parameter 2
#     :param To: ambient temperature [C]
#     :return: Predicted temperature field [C]
#     """
#     if u < 0:
#         u = -u
#         xi = -xi
#     r = np.sqrt(xi ** 2 + y ** 2)
#     ans = To + alph * u * np.exp(-beta * xi * u, dtype=np.longfloat) * kn(0, beta * r * u)
#     np.nan_to_num(ans, copy=False, nan=To, posinf=np.min(ans), neginf=np.max(ans))
#     return ans


# def predict_v(a_hat, b_hat, v0, v_min, v_max, des_width, To, t_death, grid) -> float:
#     """
#         Compute the optimal tool speed using numerical optimization
#         :param a_hat: lumped parameter 1 estimate
#         :param b_hat: lumped parameter 2 estimate
#         :param v0: speed guess [mm/s]
#         :param v_min: speed lower bound [mm/s]
#         :param v_max: speed upper bound [mm/s]
#         :param des_width: desired isotherm width [px]
#         :param To: ambient temperature [C]
#         :param t_death: isotherm temperature [C]
#         :param grid: meshgrid of x, y locations [mm]
#         :return: optimal tool speed [mm/s]
#         """
#     res = minimize(lambda x: (isotherm_width(temp_field_prediction(grid[0], grid[1],
#                                                                    u=x, alph=a_hat,
#                                                                    beta=b_hat, To=To),
#                                              t_death) - des_width) ** 2, x0=v0,
#                    bounds=((v_min, v_max),),
#                    method='Powell')
#     if not res.success:
#         raise Exception("Optimization failed")
#     return res.x[0]
#
#
# def estimate_params(v, xdata, ydata, a_hat, b_hat) -> tuple[float, float]:
#     """
#     Estimate the parameters of the model using curve fitting
#     :param v: tool speed [mm/s]
#     :param xdata: (xi, y) data [mm]
#     :param ydata: Temperature data [C]
#     :param a_hat: current estimate of a
#     :param b_hat: current estimate of b
#     :return: a_hat, b_hat, cp_hat
#     """
#     if not np.isinf(ydata).any():
#         try:
#             popt, pvoc = curve_fit(
#                 lambda x, ap, bp, cp: temp_field_prediction(x[0], x[1], u=v, alph=ap, beta=bp,
#                                                             To=To) + np.random.normal(0,
#                                                                                       cp),
#                 xdata,
#                 ydata,
#                 p0=[a_hat, b_hat, 1],
#                 bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
#                 method='trf',
#                 nan_policy='omit')
#         except ValueError:
#             return a_hat, b_hat
#         return popt[0], popt[1]
#     else:
#         return a_hat, b_hat


def find_tooltip(therm_frame: np.ndarray, t_death, neutral_tip_pos) -> tuple | None:
    """
    Find the location of the tooltip
    :param therm_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :param neutral_tip_pos: Neutral position of the tool tip [px]
    :return: x, y location of the tooltip [px]
    """
    # tip_mask = np.ones_like(therm_frame).astype(np.uint8)
    # tip_mask[neutral_tip_pos] = 0

    if (therm_frame > t_death).any():
        gaus = cv.filter2D(therm_frame, cv.CV_64F, gaus_kernel)
        norm16 = cv.normalize(gaus, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)
        cl1 = clahe.apply(norm16)
        norm_frame = cv.normalize(cl1, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        weight_mask = np.ones_like(therm_frame)
        weighted_frame = cv.multiply(norm_frame, weight_mask, dtype=cv.CV_8U)
        tip = np.unravel_index(np.argmax(weighted_frame), norm_frame.shape)
        return tip
    else:
        return None


class ThermalPID:
    Kp = 0.2
    # Ki = 0.0005
    Ki = 0
    Kd = 0.1
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

    def compute_error(self, deflection: float, width: float, importance_weight: float):
        """
        Compute the error based on the deflection and width
        :param deflection: Tool deflection [mm]
        :param width: Isotherm width [px]
        :param importance_weight: Weight of the width component (0-1)
        """
        assert 0 <= importance_weight <= 1
        self._last_error = self._error
        self.width = width
        width_error = width - self.des_width if width > self.des_width else 0
        error = (1-importance_weight) * width_error + importance_weight * (-self.width_scale * deflection)
        self._error = error

    def compute_dv(self, derror: float) -> float:
        """
        Compute the change in velocity using PID control
        :param derror: Current error rate
        :return: Change in velocity
        """
        dv = self.Kp * self._error + self.Ki * self._error_sum - self.Kd * derror
        if abs(dv) > self._max_accel:
            dv = np.sign(dv) * self._max_accel
        return dv

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

    def update(self, v, deflection: float, width, dwidth, ddeflection, importance_weight=None) -> float:
        """
        Update the tool speed based on the current state
        :param v: Current tool speed
        :param deflection: Tool deflection state
        :param width: Isotherm width
        :param dwidth: Width rate
        :param importance_weight: Weight of the width component (0-1) compared to the deflection component
        :return: New tool speed
        """
        if importance_weight is None:
            importance_weight = self.importance_weight
        self.compute_error(deflection, width, importance_weight)
        derror = (1-importance_weight) * dwidth + importance_weight * -self.width_scale * ddeflection
        dv = self.compute_dv(derror=derror)
        self.v, self._error_sum = self.enforce_pid_limits(v + dv, self._error_sum)
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
        }



class OnlineVelocityOptimizer:

    def __init__(self,
                 des_width: float = 1,
                 v_min: float = 1,
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

        self.thermal_pid = ThermalPID(des_width=des_width, v_min=v_min, v_max=v_max, v0=v0)
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

        self._tool_stiffness = 0.8705 # N/mm

        self.neutral_tip_pos = neutral_tip_pos

    @property
    def width(self):
        return self.thermal_pid.width

    @property
    def pid_velocity(self):
        return self.thermal_pid.v

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
        self._pos_kf = KalmanFilter(dim_x=4, dim_z=2)
        self._pos_kf.x = np.array([pos[0], 0, pos[1], 0])
        self._pos_kf.F = np.array([[1, self._dt, 0, 0],
                                   [0, 0.7, 0, 0],
                                   [0, 0, 1, self._dt],
                                   [0, 0, 0, 0.7]])
        self._pos_kf.H = np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0]])
        self._pos_kf.P *= 1
        self._pos_kf.R = np.array([[10, 0],
                                   [0, 10]])
        self._pos_kf.Q = Q_discrete_white_noise(dim=4, dt=self._dt, var=9)

    def update_tool_deflection(self, frame: np.ndarray[float]) -> tuple[float, float]:
        """
        Update the tool deflection based on the current frame
        :param frame: Temperature field from the camera.
        :return: deflection [px], deflection rate [px/s]
        """
        tool_tip = find_tooltip(frame, self.t_death, self.neutral_tip_pos)
        if tool_tip is None:
            return 0, 0
        elif not self._pos_kf_init:
            self.init_pos_kf(tool_tip)
            self._pos_kf_init = True
            self.neutral_tip_pos= np.array(tool_tip)
        self._pos_kf.predict()
        self._pos_kf.update(tool_tip)
        deflection = np.array([self._pos_kf.x[0] - self.neutral_tip_pos[0], self._pos_kf.x[2] - self.neutral_tip_pos[1]])
        ddeflection = np.array([self._pos_kf.x[1], self._pos_kf.x[3]])
        return np.linalg.norm(deflection), np.linalg.norm(ddeflection) * np.sign(np.dot(deflection, ddeflection))


    def update_velocity(self, v: float, frame: np.ndarray[float], deflection, ddeflection) -> any:
        """
        Update the tool speed based on the current frame
        :param v: Current tool speed
        :param frame: Temperature field from the camera. If None, the field will be predicted using the current model
        :param deflection: Tool deflection [px]
        :param ddeflection: Tool deflection rate [px/s]
        :return: new tool speed, ellipse of the isotherm if using CV
        """

        self._deflection = deflection

        z, ellipse = cv_isotherm_width(frame, self.t_death)

        self._width_kf.predict()
        self._width_kf.update(z)
        width = self._width_kf.x[0]
        dwidth = self._width_kf.x[1]
        v = self.thermal_pid.update(v, deflection, width, dwidth, ddeflection)
        return v, ellipse