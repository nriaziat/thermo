from functools import lru_cache
from scipy import optimize
import cv2 as cv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from control import lqr
import numpy as np
import do_mpc
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation
from casadi import fabs

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

exp_gamma = np.exp(0.5772)

@lru_cache
def F(Tc):
    return np.exp(-Tc) * (1 + (1.477 * Tc) ** -1.407) ** 0.7107

# @lru_cache
def ymax(alpha, u, Tc):
    return fabs(4 * alpha / (u * exp_gamma) * F(Tc))

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
    gamma_d = 1
    gamma_a = 1e-2
    gamma_alpha = 1e-2
    am = 1
    qw = 1
    qd = 4
    r = 1 # regularization term
    M = 12  # horizon

    model_type = 'continuous'

    def __init__(self,
                 t_death: float,
                 v_min: float = 1,
                 v_max: float = 10,
                 v0: float = 1,
                 method: 'mpc | lqr | minimize' = 'mpc',
                 ):
        """

        """
        self.cost_data = None
        self.method = method
        self._v_min = v_min
        self._v_max = v_max
        self.v = v0
        self._error = 0
        self._last_error = 0
        self._error_sum = 0
        self.width_mm = 0
        self.width_estimate = 0
        self.dwidth_estimate = 0
        self.deflection_estimate = 0
        self.deflection_mm = 0
        self.a_hat = -1
        self.b_hat = 1
        self.alpha_hat = 0.25  # mm^2/s
        self.d_hat = 0
        self.k_tool_n_per_mm = 1.3 # N/mm
        self.thermal_px_per_mm = 5.1337
        k = 0.24 # W/M -K
        d = 50e-3 # M thickness
        q = 20 # W
        self._Tc = 2 * np.pi * k * d * (t_death - 25) / q


        self.plotting_data = {
            'width': [],
            'width_estimate': [],
            'deflection': [],
            'deflection_estimate': [],
            'v': [],
            'alpha': [],
            'd': [],
            'a': [],
            'cost_data': np.array([])
        }

    def setup_mpc(self):
        self.model = do_mpc.model.Model(model_type=self.model_type)
        self._width = self.model.set_variable(var_type='_x', var_name='width', shape=(1, 1))  # isotherm width [mm]
        # self._dwidth = self.model.set_variable(var_type='_x', var_name='dwidth', shape=(1, 1))  # isotherm width rate [mm/s]
        self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(1, 1))  # tool speed [mm/s]
        self._a = self.model.set_variable(var_type='_tvp', var_name='a', shape=(1, 1))  # thermal time constant [s]
        self._d = self.model.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # tool damping [s]
        self._alpha = self.model.set_variable(var_type='_tvp', var_name='alpha', shape=(1, 1))  # thermal time constant [s]
        self._deflection = self.model.set_variable(var_type='_tvp', var_name='deflection', shape=(1, 1)) # deflection [mm]
        self._deflection_estimate = self.model.set_variable(var_type='_z', var_name='deflection_estimate', shape=(1, 1)) # deflection [mm]
        self._width_estimate = self.model.set_variable(var_type='_tvp', var_name='width_estimate', shape=(1, 1))

        self.model.set_alg('deflection_estimate', self._d * self._u - self._deflection_estimate)
        _, s, _ = lqr(self.a_hat, self.alpha_hat, self.qw, self.r)
        self.model.set_rhs('width', self._a * self._width + ymax(self._alpha, self._u, self._Tc))
        self.model.set_expression('terminal_cost', s * (self._width ** 2))
        self.model.set_expression('running_cost', self.qd * self._deflection_estimate ** 2 +
                                  self.qw * self._width**2)
        self.model.setup()
        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_horizon': self.M,
            'n_robust': 0,
            'open_loop': 0,
            't_step': 1,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 3,
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
        self.mpc.set_nl_cons('width', self._width, ub=5, soft_constraint=True)
        self.mpc.bounds['lower', '_z', 'deflection_estimate'] = 0
        self.mpc.set_nl_cons('deflection_estimate', self._deflection, ub=2, soft_constraint=True)

        self.mpc.scaling['_z', 'deflection_estimate'] = 0.1
        self.mpc.scaling['_x', 'width'] = 1
        self.mpc.scaling['_u', 'u'] = 1

        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)
        self.mpc.setup()
        self.mpc.set_initial_guess()

    def tvp_fun(self, t_now):
        self.tvp_template['_tvp', :] = np.array([self.a_hat,
                                                 self.d_hat,
                                                 self.alpha_hat,
                                                 self.deflection_mm,
                                                 self.width_estimate])
        return self.tvp_template

    def estimate_tool_damping(self, v, deflection_mm: float) -> None:
        """
        Estimate the tool damping based on the deflection
        :param deflection_mm: Tool deflection [mm]
        """
        self.deflection_estimate = self.d_hat * v
        error = self.deflection_estimate - deflection_mm
        self.d_hat += -self.gamma_d * error * v / (1 + abs(v))
        # if self.d_hat < 0:
        #     self.d_hat = 1e-3

    def estimate_width_dynamics(self, v, width_mm) -> None:
        """
        Estimate the width constant based on the isotherm width
        :param v: Tool speed
        :param width_mm: Isotherm width
        """
        y_max_prime = ymax(1, v, self._Tc)
        width_error = self.width_estimate - width_mm
        self.width_estimate += -self.am * width_error + self.a_hat * width_mm + self.alpha_hat * y_max_prime
        self.a_hat += -self.gamma_a * width_error * width_mm
        self.alpha_hat += -self.gamma_alpha * width_error * y_max_prime / (1 + abs(y_max_prime))
        # if self.alpha_hat < 0:
        #     self.alpha_hat = 0
        # if self.a_hat > 0:
        #     self.a_hat = 0

    def find_mpc_speed(self) -> float:
        """
        Find the optimal tool speed using MPC
        """
        # if self.width_mm < 0.05 or self.deflection_mm < 0.05:
        #     # self.mpc.bounds['lower', '_u', 'u'] = 5
        #     print("Fast Travel Mode")
        # else:
        #     print("Normal Mode")
        #     # self.mpc.bounds['lower', '_u', 'u'] = self._v_min
        self.mpc.z0['deflection_estimate', :] = self.v * self.d_hat
        u = self.mpc.make_step(np.array([self.width_mm]))
        return u.item()

    def find_optimal_speed(self) -> float:
        """
        Find the optimal tool speed using a differentiable quadratic cost function
        """
        def defl(v):
            return self.d_hat * v

        def defl_energy(v):
            return 0.5 * self.k_tool_n_per_mm * defl(v) ** 2

        w = lambda v: ymax(self.alpha_hat, v, self._Tc)

        def cost_fun(v):
            width = (w(v) / 5)**2
            deflection = (defl(v) / 0.5)**2
            return self.qw * width + self.qd * deflection + self.r * (self.v-v) ** 2

        self.plotting_data['cost_data'] = cost_fun(np.linspace(self._v_min, self._v_max, 100))

        # res = optimize.least_squares(cost_fun, np.array([self.v]), method='lm')
        # res = optimize.minimize_scalar(cost_fun, bounds=(self._v_min, self._v_max), method='bounded')
        # v = res.x[0]
        # v = abs(2**(1/6) * self.qw**(1/6)*self.qd**(-1/6) * abs(4 * self.alpha_hat * F(self._Tc) / exp_gamma) ** (1/3) * self.k_tool_n_per_mm**(-1/3) * self.d_hat ** (-2/3))
        v = optimize.minimize_scalar(cost_fun, bounds=(self._v_min, self._v_max), method='bounded').x
        # v = np.clip(v, self._v_min, self._v_max)
        # if v != res.x[0]:
        #     print(f"Clipped speed: {res.x[0]:.2f} -> {v:.2f}")
        return v

    def find_lqr_speed(self, q: float, r: float) -> float:
        k, s, e = lqr(0, self.b_hat, q, r)
        # p = solve_scalar_riccati(self.a_hat, self.b_hat, q, r)
        # k = self.d_hat * p[0] / r
        # print(k)
        v = -k.item() * self.width_mm
        print(f"v_lqr: {v:.2f} mm/s")
        return v

    def update(self, deflection_mm: float, width_mm: float) -> float:
        """
        Update the tool speed based on the current state
        :param deflection_mm: Tool deflection state
        :param width_mm: Isotherm width [mm]
        :return: New tool speed
        """
        self.width_mm = width_mm
        self.deflection_mm = deflection_mm
        self.estimate_tool_damping(self.v, deflection_mm)
        self.estimate_width_dynamics(self.v, width_mm)
        if self.method == 'mpc':
            assert hasattr(self, 'mpc'), "MPC not set up"
            self.v = self.find_mpc_speed()
        elif self.method == 'lqr':
            self.v = self.find_lqr_speed(self.q_hat, self.r)
        else:
            self.v = self.find_optimal_speed()
        self.plotting_data['width'].append(width_mm)
        self.plotting_data['width_estimate'].append(self.width_estimate)
        self.plotting_data['deflection'].append(deflection_mm)
        self.plotting_data['deflection_estimate'].append(self.deflection_estimate)
        self.plotting_data['v'].append(self.v)
        self.plotting_data['alpha'].append(self.alpha_hat)
        self.plotting_data['a'].append(self.a_hat)
        self.plotting_data['d'].append(self.d_hat)

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
            "width": self.width_mm,
            "tool_damping": self.d_hat
        }


class OnlineVelocityOptimizer:
    k_tool = 1.3 # N/mm

    def __init__(self,
                 v_min: float = 0.25,
                 v_max: float = 10,
                 v0: float = 1,
                 t_death: float = 50,
                 **kwargs):
        """
        :param v_min: minimum tool speed [mm/s]
        :param v_max: maximum tool speed [mm/s]
        :param v0: Starting tool speed [mm/s]
        :param t_death: Isotherm temperature [C]
        """

        self.thermal_controller: ThermalController = ThermalController(t_death=t_death, v_min=v_min, v_max=v_max, v0=v0, **kwargs)

        self.thermal_px_per_mm = 5.1337
        self.thermal_controller.thermal_px_per_mm = self.thermal_px_per_mm
        self.thermal_controller.k_tool_n_per_mm = self.k_tool

        self._dt = 1 / 24
        self.t_death = t_death
        self._width_kf = KalmanFilter(dim_x=1, dim_z=1)
        self._width_kf.x = np.array([0])
        self._width_kf.F = np.array([[self.thermal_controller.a_hat]])
        self._width_kf.B = np.array([[self.thermal_controller.alpha_hat]])
        self._width_kf.H = np.array([[1]])
        self._width_kf.P *= 10
        self._width_kf.R = 4
        self._width_kf.Q = 50

        # self._width_kf = KalmanFilter(dim_x=2, dim_z=1)
        # self._width_kf.x = np.array([0, 0])
        # self._width_kf.F = np.array([[1, 1], [0, 1]])
        # self._width_kf.H = np.array([[1, 0]])
        # self._width_kf.P *= 10
        # self._width_kf.R = 1
        # self._width_kf.Q = Q_discrete_white_noise(dim=2, dt=self._dt, var=50)

        self._pos_kf_init = False
        self._pos_init = None
        self._deflection_mm = 0
        self._pos_kf = None

        self.ellipse = None

        self.neutral_tip_pos = None
        self._neutral_tip_candidates = []
        self.fig, self.axs = plt.subplots(6, sharex=False, figsize=(16, 9))
        for i in range(1, len(self.axs)-1):
            self.axs[i].sharex(self.axs[0])
            self.axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.line_plots = []

        if self.thermal_controller.method == 'mpc':
            self.thermal_controller.setup_mpc()
            self.graphics = do_mpc.graphics.Graphics(self.thermal_controller.mpc.data)
            self.graphics.add_line(var_type='_x', var_name='width', axis=self.axs[0])
            self.graphics.add_line(var_type='_tvp', var_name='width_estimate', axis=self.axs[0])
            self.graphics.add_line(var_type='_tvp', var_name='deflection', axis=self.axs[1])
            self.graphics.add_line(var_type='_z', var_name='deflection_estimate', axis=self.axs[1])
            self.graphics.add_line(var_type='_u', var_name='u', axis=self.axs[2])
            self.graphics.add_line(var_type='_tvp', var_name='alpha', axis=self.axs[3])
            self.graphics.add_line(var_type='_tvp', var_name='a', axis=self.axs[4])
            self.graphics.add_line(var_type='_tvp', var_name='d', axis=self.axs[5])

        else:
            line1, = self.axs[0].plot([], label='width', color='b')
            line2, = self.axs[0].plot([], label='width estimate', color='r', linestyle='--')
            self.axs[0].legend(loc=2)
            line3, = self.axs[1].plot([], label='deflection', color='b')
            line4, = self.axs[1].plot([], label='deflection estimate', color='r', linestyle='--')
            self.axs[1].legend(loc=2)
            line5, = self.axs[2].plot([], label='v', color='r')
            line6, = self.axs[3].plot([], label='alpha', color='b')
            line7, = self.axs[4].plot([], label='d', color='b')
            line8, = self.axs[5].plot([], [], label='Cost Function', color='b')
            self.line_plots = [line1, line2, line3, line4, line5, line6, line7, line8]
            self.anim = animation.FuncAnimation(self.fig, self.animate, fargs=(self.thermal_controller.plotting_data,), interval=1, blit=True)

        # for line in self.line_plots:
        #     line.set_data([], [])

        self.axs[0].set_ylabel(r'$w~[\si[per-mode=fraction]{\milli\meter}]$')
        self.axs[1].set_ylabel(r"$\delta~[\si[per-mode=fraction]{\milli\meter}]$")
        self.axs[2].set_ylabel(r"$u~[\si[per-mode=fraction]{\milli\meter\per\second}]$")
        self.axs[3].set_ylabel(r'$\hat{\alpha}$')
        # self.axs[4].set_ylabel(r'$\hat{b}$')
        # self.axs[4].set_ylabel(r'$\hat{Q}$')
        # self.axs[4].set_xlabel('Time Step')
        # self.axs[5].set_ylabel(r'$\text{Cost Function}$')
        # self.axs[5].set_xlabel(r'$\text{Speed}~[\si[per-mode=fraction]{\milli\meter\per\second}]$')
        self.axs[4].set_ylabel(r'$1/\hat{\tau}$')
        self.axs[5].set_ylabel(r'$\hat{d}$')
        self.axs[5].set_xlabel('Time Step')
        self.fig.align_ylabels()



    @property
    def width_mm(self):
        return self.thermal_controller.width_mm

    @property
    def controller_v_mm_s(self):
        return self.thermal_controller.v

    @property
    def deflection_mm(self):
        return self._deflection_mm

    @property
    def deflection_energy(self):
        return 1/2 * self.k_tool * self.deflection_mm ** 2

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

    def update_velocity(self, v: float, frame: np.ndarray[float], deflection_mm) -> any:
        """
        Update the tool speed based on the current frame
        :param v: Current tool speed
        :param frame: Temperature field from the camera. If None, the field will be predicted using the current model
        :param deflection_mm: Tool deflection [mm]
        :return: new tool speed, ellipse of the isotherm if using CV
        """

        self._deflection_mm = deflection_mm

        z, self.ellipse = cv_isotherm_width(frame, self.t_death)
        # z = np.max(np.sum(frame > self.t_death, axis=0)) / 2

        self._width_kf.predict(u=ymax(1, v, self.thermal_controller._Tc),
                               B=np.array([self.thermal_controller.alpha_hat]),
                               F=np.array([self.thermal_controller.a_hat]))
        self._width_kf.update(z / self.thermal_px_per_mm)
        width_mm = self._width_kf.x[0]
        v = self.thermal_controller.update(deflection_mm, width_mm)
        if abs(self.thermal_controller.width_estimate) > 1e3:
            print(f"Unstable system: width_estimate: {self.thermal_controller.width_estimate:.2f}")
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

    def animate(self, _, *fargs):
            data_dict = fargs[0]
            l = len(data_dict.values())
            for i, (line, data) in enumerate(zip(self.line_plots, data_dict.values())):
                if i == (l - 1):
                    x = np.linspace(self.thermal_controller._v_min, self.thermal_controller._v_max, 100)
                    line.set_data(x, data)
                line.set_data(range(len(data)), data)
            for ax in self.axs:
                ax.relim()
                ax.autoscale_view()
            self.axs[-1].set_xticks([0, 100], [self.thermal_controller._v_min, self.thermal_controller._v_max])
            return self.line_plots


