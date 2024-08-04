import numpy as np
import cv2 as cv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from control import lqr
from casadi import *
import do_mpc

# import matplotlib.pyplot as plt
# plt.ion()
# from matplotlib import rcParams
# rcParams['text.usetex'] = True
# rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',r'\usepackage{siunitx}']
# rcParams['axes.grid'] = True
# rcParams['lines.linewidth'] = 2.0
# rcParams['axes.labelsize'] = 'xx-large'
# rcParams['xtick.labelsize'] = 'xx-large'
# rcParams['ytick.labelsize'] = 'xx-large'

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
    w = ellipse[1][0]
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
    # tip_mask = np.ones_like(therm_frame).astype(np.uint8)
    # tip_mask[neutral_tip_pos] = 0

    if (therm_frame > t_death).any():
        # gaus = cv.filter2D(therm_frame, cv.CV_64F, gaus_kernel)
        norm16 = cv.normalize(therm_frame, None, 0, 65535, cv.NORM_MINMAX, cv.CV_16U)
        # cl1 = clahe.apply(norm16)
        # norm_frame = cv.normalize(cl1, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # if last_tool_tip is not None:
        #     weight_mask = np.zeros_like(norm_frame).astype(np.uint8)
        #     weight_mask[int(last_tool_tip[0]), int(last_tool_tip[1])] = 255
        #     weight_mask = cv.distanceTransform(weight_mask, cv.DIST_L2, 5)
        #     weight_mask = cv.normalize(weight_mask, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        #     weight_mask = np.exp(-weight_mask / 10)
        # else:
        #     weight_mask = np.ones_like(norm_frame).astype(np.uint8)
        # weighted_frame = cv.multiply(norm_frame, weight_mask, dtype=cv.CV_8U)
        # tip = np.unravel_index(np.argmax(weighted_frame), weighted_frame.shape)
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


def solve_scalar_riccati(a, b, q, r):
    """
    Solve the scalar Riccati equation
    """
    p = (-2 * a + np.sqrt(4 * a ** 2 - 4 * (b ** 2) / r * q)) / (2 * (b ** 2) / r)
    p2 = (-2 * a - np.sqrt(4 * a ** 2 - 4 * (b ** 2) / r * q)) / (2 * (b ** 2) / r)
    print(p, p2)
    return p, p2


class ThermalController:
    gamma_d = 0.005
    gamma_b = 0.0005
    gamma_a = 0.0005
    am = 1
    q = 1
    r = 1e-4
    M = 30  # control horizon

    model_type = 'continuous'

    def __init__(self,
                 v_min: float = 1,
                 v_max: float = 10,
                 max_accel: float = 10,
                 v0: float = 1):
        """

        """
        self._v_min = v_min
        self._v_max = v_max
        self._max_accel = max_accel
        self.v = v0
        self._error = 0
        self._last_error = 0
        self._error_sum = 0
        self.width = 0
        self.width_constant_estimate = 10
        self.width_estimate = 0
        self.deflection = 0
        self.a_hat = 1
        self.b_hat = -1
        self.d_hat = 0.05

        self.model = do_mpc.model.Model(model_type=self.model_type)
        self._width = self.model.set_variable(var_type='_x', var_name='width', shape=(1, 1))
        self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
        self._a = self.model.set_variable(var_type='_tvp', var_name='a', shape=(1, 1))
        self._b = self.model.set_variable(var_type='_tvp', var_name='b', shape=(1, 1))
        self._d = self.model.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))
        self._deflection = self.model.set_variable(var_type='_z', var_name='deflection', shape=(1, 1))
        self.model.set_alg('deflection', self._d * self._u - self.deflection)
        self.model.set_rhs('width', self._a * self._width + self._b * self._u)
        _, s, _ = lqr(self.a_hat, self.b_hat, self.q, self.r)
        self.model.set_expression('terminal_cost', s.item() * (self._width ** 2))
        self.model.set_expression('running_cost', 20000 * (self._deflection ** 2) +
                                                            self.q * (self._width ** 2))
        self.model.setup()
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': self.M,
            'n_robust': 0,
            'open_loop': 0,
            't_step': 1,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 1,
            'store_full_solution': True,
            # 'nlpsol_opts': {'ipopt.linear_solver': ''},
        }
        # self.mpc.settings.supress_ipopt_output()
        self.mpc.set_param(**setup_mpc)
        self.mterm = self.model.aux['terminal_cost']
        self.lterm = self.model.aux['running_cost']
        self.mpc.set_objective(mterm=self.mterm, lterm=self.lterm)
        self.mpc.set_rterm(u=self.r)
        self.mpc.bounds['lower', '_u', 'u'] = self._v_min
        self.mpc.bounds['upper', '_u', 'u'] = self._v_max
        self.mpc.bounds['lower', '_x', 'width'] = 0
        self.mpc.bounds['upper', '_x', 'width'] = np.inf
        self.mpc.bounds['lower', '_z', 'deflection'] = 0
        self.mpc.bounds['upper', '_z', 'deflection'] = np.inf
        self.mpc.scaling['_z', 'deflection'] = 0.01
        self.mpc.scaling['_u', 'u'] = 1

        # self.mpc.bounds['lower', '_x', 'pos'] = 0
        # self.mpc.bounds['upper', '_x', 'pos'] = np.inf
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)
        self.mpc.setup()
        self.simulator = do_mpc.simulator.Simulator(self.model)
        params_simulator = {
            'integration_tool': 'cvodes',
            'abstol': 1e-8,
            'reltol': 1e-8,
            't_step': 1,
        }
        self.x0 = np.array([0])
        self.simulator.set_param(**params_simulator)
        self.sim_tvp_template = self.simulator.get_tvp_template()
        self.simulator.set_tvp_fun(self.sim_tvp_fun)
        self.simulator.setup()
        self.simulator.x0 = self.x0
        self.mpc.x0 = self.x0
        self.mpc.set_initial_guess()

    def tvp_fun(self, t_now):
        self.tvp_template['_tvp', :] = np.array([self.a_hat, self.b_hat, self.d_hat])
        return self.tvp_template

    def sim_tvp_fun(self, t_now):
        # self.sim_tvp_template['width'] = self.width
        self.sim_tvp_template['a'] = self.a_hat
        self.sim_tvp_template['b'] = self.b_hat
        self.sim_tvp_template['d'] = self.d_hat
        return self.sim_tvp_template

    def estimate_tool_damping(self, v, deflection: float) -> None:
        """
        Estimate the tool damping based on the deflection
        :param deflection: Tool deflection [mm]
        """
        deflection_estimate = self.d_hat * v
        error = deflection_estimate - deflection
        self.d_hat += -self.gamma_d * v * error

    # def estimate_width_constant(self, v, width: float) -> None:
    #     """
    #     Estimate the width constant based on the isotherm width
    #     :param width: Isotherm width
    #     """
    #     width_estimate = self.width_constant_estimate / v
    #     error = width_estimate - width
    #     self.width_constant_estimate += -(self.gamma_b / v) * error

    def estimate_width_dynamics(self, v, width):
        """
        Estimate the width constant based on the isotherm width
        :param v: Tool speed
        :param width: Isotherm width
        """
        width_error = self.width_estimate - width
        self.a_hat += -self.gamma_a * width_error * width
        self.b_hat += -self.gamma_b * width_error * v
        self.width_estimate += -self.am * width_error + self.a_hat * width + self.b_hat * v
        # print(f"width estimate: {self.width_estimate:.1e}, a_hat: {self.a_hat:.1e}, b_hat: {self.b_hat:.1e}, v: {v:.1f}")

    # def find_optimal_speed(self, w1: float, w2: float) -> float:
    #     """
    #     Find the optimal tool speed based on the estimated parameters using minimization
    #     :param w1: Weight of the width component
    #     :param w2: Weight of the deflection component
    #     """
    #     v = ((w1/w2) ** 0.25) * (self.width_constant_estimate / self.d_hat) ** 0.5
    #     if np.isnan(v):
    #         v = ((w1/w2) ** 0.25) * (-self.width_constant_estimate / self.d_hat) ** 0.5
    #     v = np.clip(v, self._v_min, self._v_max)
    #     return v

    def find_mpc_speed(self, width: float) -> float:
        u0 = self.mpc.make_step(np.array([width, self.a_hat, self.b_hat, self.d_hat, self.width_estimate]))
        return u0.item()

    def find_lqr_speed(self, q: float, r: float) -> float:
        k, s, e = lqr(self.a_hat, self.b_hat, q, r)
        # p = solve_scalar_riccati(self.a_hat, self.b_hat, q, r)
        # k = self.d_hat * p[0] / r
        # print(k)
        v = -k.item() * self.width
        print(f"v_lqr: {v:.2f} mm/s")
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
        self.deflection = deflection
        self.estimate_tool_damping(v, deflection)
        # self.estimate_width_constant(v, width)
        # self.v = self.find_optimal_speed(1, 250)

        self.estimate_width_dynamics(v, width)
        # v = abs(self.find_lqr_speed(self.q, self.r))
        # self.v = np.clip(v, self._v_min, self._v_max)
        v_mpc = self.find_mpc_speed(width)
        # print(f"v_mpc: {v_mpc:.2f} mm/s")
        self.v = v_mpc
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
            "tool_damping": self.d_hat
        }


class OnlineVelocityOptimizer:

    def __init__(self,
                 v_min: float = 0.25,
                 v_max: float = 10,
                 v0: float = 1,
                 t_death: float = 50):
        """
        :param v_min: minimum tool speed [mm/s]
        :param v_max: maximum tool speed [mm/s]
        :param v0: Starting tool speed [mm/s]
        :param t_death: Isotherm temperature [C]
        """

        self.thermal_controller: ThermalController = ThermalController(v_min=v_min, v_max=v_max, v0=v0)
        self._dt = 1 / 24
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

        self.neutral_tip_pos = None
        self._neutral_tip_candidates = []

    @property
    def width(self):
        return self.thermal_controller.width

    @property
    def controller_v(self):
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
        if self._pos_kf_init:
            last_tool_tip = self._pos_kf.x[0], self._pos_kf.x[2]
        else:
            last_tool_tip = None
        tool_tip = find_tooltip(frame, self.t_death, last_tool_tip, self.ellipse)
        if tool_tip is None:
            return 0, 0
        elif not self._pos_kf_init:
            self._neutral_tip_candidates.append(tool_tip)
            if len(self._neutral_tip_candidates) > 10:
                self.neutral_tip_pos = np.mean(self._neutral_tip_candidates, axis=0)
                self.init_pos_kf(self.neutral_tip_pos)
                self._pos_kf_init = True
        else:
            self._pos_kf.predict()
            self._pos_kf.update(tool_tip)
            deflection = np.array(
                [self._pos_kf.x[0] - self.neutral_tip_pos[0], self._pos_kf.x[2] - self.neutral_tip_pos[1]])
            ddeflection = np.array([self._pos_kf.x[1], self._pos_kf.x[3]])
            return np.linalg.norm(deflection), np.linalg.norm(ddeflection) * np.sign(np.dot(deflection, ddeflection))
        return 0, 0

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
        if abs(self.thermal_controller.b_hat) > 1e3 or abs(self.thermal_controller.a_hat) > 1e3 or abs(
                self.thermal_controller.width_estimate) > 1e3:
            raise ValueError("Unstable system")
        return v, self.ellipse
