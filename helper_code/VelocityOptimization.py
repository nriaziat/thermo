from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import do_mpc
from utils import cv_isotherm_width, find_tooltip, isotherm_width, ymax
from abc import ABC, abstractmethod


class AdaptiveMPC:

    thermal_px_per_mm = 5.1337

    def __init__(self, model: do_mpc.model.Model, t_death_c: float, v_min, v_max):
        """
        :param t_death_c: Isotherm temperature [C]
        :param v_min: minimum tool speed [mm/s]
        :param v_max: maximum tool speed [mm/s]
        """
        super().__init__(model)
        self._tvp_template = None
        self.v_min = v_min
        self.v_max = v_max
        self.v = v_min

        # self._width_adaptation = ScalarFirstOrderAdaptation(0, 1, 0.2,
        #                                                     1e-3, 1e-2,
        #                                                     regularize_input=True)
        self._width_meas = 0
        # self._deflection_adaptation = FirstOrderAdaptation(0, 0.01, 1,
        #                                                    0, 1)
        self._deflection_meas = 0


        k = 0.24 # W/M -K
        d = 50e-3 # M thickness
        q = 20 # W
        self.Tc = 2 * np.pi * k * d * (t_death_c - 25) / q

    # @property
    # def alpha_hat(self):
    #     """
    #     Thermal Diffusivity Estimate
    #     """
    #     return self._width_adaptation.b
    #
    # @property
    # def a_hat(self):
    #     """
    #     Thermal Time Constant Estimate
    #     """
    #     return self._width_adaptation.a
    #
    # @property
    # def width_hat_mm(self):
    #     """
    #     Isotherm Width Estimate [mm]
    #     """
    #     return self._width_adaptation.state_estimate
    #
    # @property
    # def d_hat(self):
    #     """
    #     Tool Damping Estimate
    #     """
    #     return self._deflection_adaptation.b
    #
    # @property
    # def deflection_hat_mm(self):
    #     """
    #     Tool Deflection Estimate [mm]
    #     """
    #     return self._deflection_adaptation.state_estimate


    def setup_mpc(self):
        self.model = do_mpc.model.Model(model_type=self.model_type)
        self._width = self.model.set_variable(var_type='_x', var_name='width', shape=(1, 1))  # isotherm width [mm]
        self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(1, 1))  # tool speed [mm/s]
        self._actual_u = self.model.set_variable(var_type='_tvp', var_name='actual_u', shape=(1, 1))  # tool speed [mm/s]
        self._a = self.model.set_variable(var_type='_tvp', var_name='a', shape=(1, 1))  # thermal time constant [s]
        self._d = self.model.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # tool damping [s]
        self._alpha = self.model.set_variable(var_type='_tvp', var_name='alpha', shape=(1, 1))  # thermal input constant [s]
        if isinstance(self._deflection_adaptation, ScalarLinearAlgabraicAdaptation):
            self._deflection = self.model.set_variable(var_type='_z', var_name='deflection', shape=(1, 1)) # deflection [mm]
            self.model.set_alg('deflection', expr=self._deflection - self._d * np.exp(-self.c_defl / self._u))
        else:
            self._deflection = self.model.set_variable(var_type='_x', var_name='deflection', shape=(1, 1)) # deflection [mm]
            self.model.set_rhs('deflection', -self._deflection_adaptation.a * self._deflection + self._d * self._u)

        self._deflection_meas = self.model.set_variable(var_type='_tvp', var_name='deflection_measurement', shape=(1, 1)) # deflection [mm]
        self._width_estimate = self.model.set_variable(var_type='_tvp', var_name='width_estimate', shape=(1, 1))

        _, s, _ = lqr(-self.a_hat, self.alpha_hat, self.qw, self.r)
        self.model.set_rhs('width', -self._a * self._width + self.w_max/(np.pi/2) * np.arctan(self._alpha * ymax(1, self._u, self.Tc)))
        self.model.set_expression('terminal_cost', s * self._width**2)
        self.model.set_expression('running_cost', self.qd * (10 * self._deflection)**2 +
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
        mterm = self.model.aux['terminal_cost']
        lterm = self.model.aux['running_cost']
        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(u=self.r)

        self.mpc.bounds['lower', '_u', 'u'] = self.v_min
        self.mpc.bounds['upper', '_u', 'u'] = self.v_max

        self.mpc.bounds['lower', '_x', 'width'] = 0
        self.mpc.set_nl_cons('width', self._width, ub=self.w_max, soft_constraint=True)
        if isinstance(self._deflection_adaptation, ScalarLinearAlgabraicAdaptation):
            self.mpc.bounds['lower', '_z', 'deflection'] = 0
            self.mpc.scaling['_z', 'deflection'] = 0.1

        else:
            self.mpc.bounds['lower', '_x', 'deflection'] = 0
            self.mpc.scaling['_x', 'deflection'] = 0.1
        self.mpc.set_nl_cons('deflection', self._deflection, ub=2, soft_constraint=True)

        self.mpc.scaling['_x', 'width'] = 1
        self.mpc.scaling['_u', 'u'] = 1

        self._tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self._tvp_fun)
        self.mpc.setup()
        self.mpc.set_initial_guess()


    def _tvp_fun(self, t_now):
        self._tvp_template['_tvp', :] = np.array([self.v,
                                                  self.a_hat if self.a_hat > 0 else 0,
                                                  self.d_hat if self.d_hat > 0 else 0,
                                                  self.alpha_hat if self.alpha_hat > 0 else 0,
                                                  self._deflection_meas,
                                                  self.width_hat_mm])
        return self._tvp_template

    def _find_speed(self, width_mm=None, deflection_mm=None) -> float:
        """
        Find the optimal tool speed using MPC
        """
        if width_mm is None:
            width_mm = self._width_meas
        if deflection_mm is None:
            deflection_mm = self._deflection_meas

        if isinstance(self._deflection_adaptation, ScalarFirstOrderAdaptation):
            u = self.mpc.make_step(np.array([width_mm, deflection_mm]))
        else:
            u = self.mpc.make_step(np.array([width_mm]))
        return u.item()

    def update(self, deflection_mm: float, width_mm: float) -> float:
        """
        Update the tool speed based on the current state
        :param deflection_mm: Tool deflection state
        :param width_mm: Isotherm width [mm]
        :return: New tool speed
        """
        self._width_meas = width_mm
        self._deflection_meas = deflection_mm
        if self.v > 0:
            self._deflection_adaptation.update(deflection_mm, np.exp(-self.c_defl / self.v))
            self._width_adaptation.update(width_mm, ymax(1, self.v, self.Tc))

        self.v = self._find_speed()

        return self.v


class WidthKF(KalmanFilter):
    def __init__(self, a, alpha):
        super().__init__(dim_x=1, dim_z=1, dim_u=1)
        self.x = np.array([0])
        self.F = np.array([[a]])
        self.B = np.array([[alpha]])
        self.H = np.array([[1]])
        self.P *= 10
        self.R = 4
        self.Q = 50

    @property
    def width_mm(self):
        return self.x[0]


class ToolTipKF(KalmanFilter):
    def __init__(self, damping_ratio, pos=None):
        super().__init__(dim_x=4, dim_z=2, dim_u=1)
        if pos is not None:
            self.x = np.array([pos[0], 0, pos[1], 0])
        else:
            self.x = np.array([0, 0, 0, 0])
        self.F = np.array([[1, 1, 0, 0],
                           [0, damping_ratio, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, damping_ratio]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        self.P *= 4
        self.R = 9 * np.eye(2)
        self.Q = Q_discrete_white_noise(dim=4, dt=1/24, var=36)

    @property
    def pos_mm(self):
        return self.x[0], self.x[2]


class OnlineVelocityOptimizer:

    def __init__(self,
                 v_min: float = 0.25,
                 v_max: float = 10,
                 t_death_c: float = 50):
        """
        :param v_min: minimum tool speed [mm/s]
        :param v_max: maximum tool speed [mm/s]
        :param t_death_c: Isotherm temperature [C]
        """

        self.adaptive_mpc: AdaptiveMPC = AdaptiveMPC(t_death_c=t_death_c, v_min=v_min, v_max=v_max)

        self.thermal_px_per_mm = 5.1337
        self.adaptive_mpc.thermal_px_per_mm = self.thermal_px_per_mm
        self.adaptive_mpc.setup_mpc()

        self._dt = 1 / 24
        self.t_death_c = t_death_c

        self._width_kf = WidthKF(self.adaptive_mpc.a_hat, self.adaptive_mpc.alpha_hat)

        self._pos_kf_init = False
        self._pos_init = None
        self._deflection_mm = 0
        self._pos_kf = None

        self.ellipse = None

        self.neutral_tip_pos = None
        self._neutral_tip_candidates = []

    @property
    def width_mm(self):
        return self._width_kf.width_mm

    @property
    def controller_v_mm_s(self):
        return self.adaptive_mpc.v

    @property
    def deflection_mm(self):
        return self._deflection_mm

    @property
    def tool_tip_pos(self):
        """
        Returns the current position of the tool tip (x, y)
        """
        if self._pos_kf is None:
            return None
        return self._pos_kf.x[0], self._pos_kf.x[2]

    @property
    def v_min(self):
        return self.adaptive_mpc.v_min

    @property
    def v_max(self):
        return self.adaptive_mpc.v_max

    def init_pos_kf(self, pos: tuple):
        """
        Initialize the position Kalman filter
        :param pos: Initial position of the tool tip
        """
        damping_ratio = 0.7
        self._pos_kf = ToolTipKF(damping_ratio, pos)

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
        tool_tip = find_tooltip(frame, self.t_death_c, last_tool_tip, self.ellipse)
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
            if self._pos_kf.x[0] < self.neutral_tip_pos[0]:
                self.neutral_tip_pos = self._pos_kf.x[0], self.neutral_tip_pos[1]
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

    def update_velocity(self,v: float, frame: np.ndarray[float], deflection_mm) -> any:
        """
        Update the tool speed based on the current frame
        :param v: Current tool speed
        :param frame: Temperature field from the camera. If None, the field will be predicted using the current model
        :param deflection_mm: Tool deflection [mm]
        :return: new tool speed, ellipse of the isotherm if using CV
        """

        self._deflection_mm = deflection_mm
        z, self.ellipse = cv_isotherm_width(frame, self.t_death_c)

        unit_y_max = ymax(1, v, self.adaptive_mpc.Tc)
        if np.isinf(unit_y_max):
            raise RuntimeError("0 Velocity led to unstable width KF.")
        self._width_kf.predict(u=unit_y_max,
                               B=np.array([self.adaptive_mpc.alpha_hat if self.adaptive_mpc.alpha_hat > 0 else 0]),
                               F=np.array([self.adaptive_mpc.a_hat if self.adaptive_mpc.a_hat > 0 else 0]))
        self._width_kf.update(z / self.thermal_px_per_mm)
        v = self.adaptive_mpc.update(deflection_mm, self.width_mm)
        if abs(self.adaptive_mpc.width_hat_mm) > 1e3:
            print(f"Unstable system: width_estimate: {self.adaptive_mpc.width_hat_mm:.2f}")
            raise ValueError("Unstable system")
        return v, self.ellipse

