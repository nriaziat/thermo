import numpy as np
import cv2 as cv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from control import lqr
from casadi import *
import do_mpc
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.ion()

rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'
rcParams['axes.grid'] = True
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'xx-large'
rcParams['xtick.labelsize'] = 'xx-large'
rcParams['ytick.labelsize'] = 'xx-large'


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
    gamma_d = 6e-3
    gamma_b = 7e-4
    gamma_b2 = 7e-3
    gamma_a = 7e-4
    am = 1
    qw = 1
    qd = 1
    r = 1e-4
    M = 10  # horizon

    model_type = 'continuous'

    def __init__(self,
                 v_min: float = 1,
                 v_max: float = 10,
                 v0: float = 1,
                 ):
        """

        """
        self._v_min = v_min
        self._v_max = v_max
        self.v = v0
        self._error = 0
        self._last_error = 0
        self._error_sum = 0
        self.width = 0
        self.width_constant_estimate = 10
        self.width_estimate = 0
        self.deflection_estimate = 0
        self.deflection = 0
        self.a_hat = 0
        self.b_hat = -0.5
        self.b2_hat = 0.5
        # self.a_hat = np.array([[0, 1], [-1, -2]])
        # self.b_hat = np.array([[-1], [0]])
        # self.b2_hat = np.array([[0], [1]])
        self.d_hat = 0.1
        self.k_tool_n_per_mm = 1.3 # N/mm
        self.thermal_pixel_per_mm = 5.1337

    def setup(self):
        self.model = do_mpc.model.Model(model_type=self.model_type)
        self._width = self.model.set_variable(var_type='_x', var_name='width', shape=(1, 1))  # isotherm width [mm]
        # self._width_est = self.model.set_variable(var_type='_x', var_name='width_est', shape=(2, 1))  # isotherm width [mm]
        self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(1, 1))  # tool speed [mm/s]
        self._a = self.model.set_variable(var_type='_tvp', var_name='a', shape=(1, 1))  # width dynamics
        self._b = self.model.set_variable(var_type='_tvp', var_name='b', shape=(1, 1))  # width dynamics
        # self._b = self.model.set_variable(var_type='_tvp', var_name='b', shape=(2, 1))  # width dynamics
        self._b2 = self.model.set_variable(var_type='_tvp', var_name='b2', shape=(1, 1)) # width dynamics
        # self._b2 = self.model.set_variable(var_type='_tvp', var_name='b2', shape=(2, 1)) # width dynamics
        self._d = self.model.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # tool damping [s]
        self._deflection_est = self.model.set_variable(var_type='_z', var_name='deflection_est', shape=(1, 1)) # deflection [mm]
        self._defl_energy = self.model.set_variable(var_type='_z', var_name='deflection_energy', shape=(1, 1)) # deflection energy [mJ]
        self.model.set_alg('deflection_energy', (0.5 * self.k_tool_n_per_mm * self._deflection_est ** 2) - self._defl_energy)
        self.model.set_alg('deflection_est', self._d * self._u - self._deflection_est)
        self.model.set_rhs('width', self._a * self._width + self._b * self._u + self._b2)
        self.model.set_expression('terminal_cost', self.qw * (self._width ** 2))
        # self.model.set_expression('terminal_cost', self._width_est.T @ (self.qw * np.eye(2)) @ self._width_est)
        self.model.set_expression('running_cost', self.qd * ((100 * self._defl_energy) ** 2) +
                                  self.qw * (self._width ** 2))
        # self.model.set_expression('running_cost', self.qd * (self._defl_energy ** 2) +
        #                             self._width_est.T @ (self.qw * np.eye(2)) @ self._width_est)
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
        }
        self.mpc.settings.supress_ipopt_output()
        self.mpc.settings.set_linear_solver('ma27')
        self.mpc.set_param(**setup_mpc)
        self.mterm = self.model.aux['terminal_cost']
        self.lterm = self.model.aux['running_cost']
        self.mpc.set_objective(mterm=self.mterm, lterm=self.lterm)
        self.mpc.set_rterm(u=self.r)
        self.mpc.bounds['lower', '_u', 'u'] = self._v_min
        self.mpc.bounds['upper', '_u', 'u'] = self._v_max
        self.mpc.bounds['lower', '_x', 'width'] = 0
        # self.mpc.bounds['upper', '_x', 'width'] = 10
        self.mpc.set_nl_cons('width', self._width, ub=10, soft_constraint=True)
        self.mpc.bounds['lower', '_z', 'deflection_est'] = 0
        # self.mpc.bounds['upper', '_z', 'deflection_est'] = 5
        self.mpc.set_nl_cons('deflection_est', self._deflection_est, ub=5, soft_constraint=True)
        # self.mpc.bounds['lower', '_x', 'deflection'] = 0
        # self.mpc.bounds['upper', '_x', 'deflection'] = np.inf
        self.mpc.scaling['_z', 'deflection_est'] = 0.1
        # self.mpc.scaling['_x', 'deflection'] = 0.1
        self.mpc.scaling['_x', 'width'] = 10
        self.mpc.scaling['_u', 'u'] = 10
        self.mpc.scaling['_z', 'deflection_energy'] = 0.01
        # self.mpc.set_uncertainty_values(a=np.array([self.a_hat, 1.1*self.a_hat, 0.9*self.a_hat]),
        #                                 b=np.array([self.b_hat, 1.1*self.b_hat, 0.9*self.b_hat]),
        #                                 b2=np.array([self.b2_hat, 1.1*self.b2_hat, 0.9*self.b2_hat]),
        #                                 d=np.array([self.d_hat, 1.1*self.d_hat, 0.9*self.d_hat]))
        # self.mpc.bounds['lower', '_x', 'pos'] = 0
        # self.mpc.bounds['upper', '_x', 'pos'] = np.inf
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)
        self.mpc.setup()
        # self.simulator = do_mpc.simulator.Simulator(self.model)
        # params_simulator = {
        #     'integration_tool': 'cvodes',
        #     'abstol': 1e-8,
        #     'reltol': 1e-8,
        #     't_step': 1,
        # }
        # self.simulator.set_param(**params_simulator)
        # self.sim_tvp_template = self.simulator.get_tvp_template()
        # self.simulator.set_tvp_fun(self.sim_tvp_fun)
        # self.simulator.setup()
        self.mpc.set_initial_guess()

    def tvp_fun(self, t_now):
        # b = np.array([self.b_hat, self.b2_hat])
        # r = np.array([[self.qd * self.d_hat ** 4, 0], [0, 0]])
        # _, s, _ = lqr(self.a_hat, b, self.qw, np.eye(2))
        self.tvp_template['_tvp', :] = np.array([self.a_hat, self.b_hat, self.b2_hat, self.d_hat])
        # self.tvp_template['_tvp', :] = [self.a_hat, self.b_hat, self.b2_hat, self.d_hat]
        return self.tvp_template

    # def sim_tvp_fun(self, t_now):
    #     # self.sim_tvp_template['width'] = self.width
    #     self.sim_tvp_template['a'] = self.a_hat
    #     self.sim_tvp_template['b'] = self.b_hat
    #     self.sim_tvp_template['d'] = self.d_hat
    #     return self.sim_tvp_template

    def estimate_tool_damping(self, v, deflection: float) -> None:
        """
        Estimate the tool damping based on the deflection
        :param deflection: Tool deflection [mm]
        """
        error = self.d_hat * v - deflection
        self.d_hat += -self.gamma_d * v * error
        if self.d_hat < 0:
            self.d_hat = 1e-3
        # delection_error = self.deflection_estimate - deflection
        # self.d_hat += -self.gamma_d * delection_error * v
        # self.deflection_estimate += -self.am * delection_error - self.ad * self.deflection_estimate + self.d_hat * v

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
        self.b2_hat += -self.gamma_b2 * width_error
        self.width_estimate += -self.am * width_error + self.a_hat * width + self.b_hat * v + self.b2_hat

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

    def find_mpc_speed(self, width: float, deflection: float) -> float:
        self.mpc.z0['deflection_est', :] = deflection / self.thermal_pixel_per_mm
        u0 = self.mpc.make_step(np.array([width / self.thermal_pixel_per_mm]))
        return u0.item()

    def find_lqr_speed(self, q: float, r: float) -> float:
        k, s, e = lqr(self.a_hat, self.b_hat, q, r)
        # p = solve_scalar_riccati(self.a_hat, self.b_hat, q, r)
        # k = self.d_hat * p[0] / r
        # print(k)
        v = -k.item() * self.width
        print(f"v_lqr: {v:.2f} mm/s")
        return v

    def update(self, v: float, deflection: float, width: float) -> float:
        """
        Update the tool speed based on the current state
        :param v: Current tool speed
        :param deflection: Tool deflection state
        :param width: Isotherm width
        :return: New tool speed
        """
        assert hasattr(self, 'mpc'), "MPC not set up"
        self.width = width
        self.deflection = deflection
        self.estimate_tool_damping(v, deflection)
        self.estimate_width_dynamics(v, width)
        self.v = self.find_mpc_speed(width, deflection)
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
    k_tool = 1.3 # N/mm

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

        self.thermal_pixel_per_mm = 5.1337
        self.thermal_controller.thermal_pixel_per_mm = self.thermal_pixel_per_mm
        self.thermal_controller.k_tool_n_per_mm = self.k_tool
        self.thermal_controller.setup()

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

        self.fig, axs = plt.subplots(7, sharex=True, figsize=(16, 9))
        self.graphics = do_mpc.graphics.Graphics(self.thermal_controller.mpc.data)
        self.graphics.add_line(var_type='_x', var_name='width', axis=axs[0])
        self.graphics.add_line(var_type='_z', var_name='deflection_energy', axis=axs[1])
        self.graphics.add_line(var_type='_u', var_name='u', axis=axs[2])
        self.graphics.add_line(var_type='_tvp', var_name='a', axis=axs[3])
        self.graphics.add_line(var_type='_tvp', var_name='b', axis=axs[4])
        self.graphics.add_line(var_type='_tvp', var_name='b2', axis=axs[5])
        self.graphics.add_line(var_type='_tvp', var_name='d', axis=axs[6])

        axs[0].set_ylabel(r'$w~[\si[per-mode=fraction]{\milli\meter}]$')
        axs[1].set_ylabel(r"$E_{\delta}~[\si[per-mode=fraction]{\milli\joule}]$")
        axs[2].set_ylabel(r"$u~[\si[per-mode=fraction]{\milli\meter\per\second}]$")
        axs[3].set_ylabel(r'$\hat{a}$')
        axs[4].set_ylabel(r'$\hat{b}$')
        axs[5].set_ylabel(r'$\hat{b}_2$')
        axs[6].set_ylabel(r'$\hat{d}$')
        self.fig.align_ylabels()


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
    def deflection_energy(self):
        return 1/2 * self.k_tool * (self.deflection / self.thermal_pixel_per_mm) ** 2

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
            # print(f"Unstable system: b_hat: {self.thermal_controller.b_hat:.2f}, "
            #         f"a_hat: {self.thermal_controller.a_hat:.2f}, "
            #         f"width_estimate: {self.thermal_controller.width_estimate:.2f}")
            raise ValueError("Unstable system")
        return v, self.ellipse

    def plot(self, t_ind=None):
        if t_ind is None:
            self.graphics.plot_results()
            self.graphics.plot_predictions()
            self.graphics.reset_axes()
        else:
            self.graphics.plot_results(t_ind)
            self.graphics.plot_predictions(t_ind)
            self.graphics.reset_axes()
