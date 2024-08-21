import cv2
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from control import lqr
import numpy as np
import do_mpc
from AdaptiveID import ScalarFirstOrderAdaptation, ScalarLinearAlgabraicAdaptation
from utils import cv_isotherm_width, find_tooltip, isotherm_width, ymax

class AdaptiveMPC:

    qw = 1  # width cost
    qd = 4  # deflection cost
    r = 1e-1 # regularization term
    M = 50  # horizon

    model_type = 'continuous'

    def __init__(self,
                 t_death_c: float,
                 v_min,
                 v_max,
                 ):
        """
        :param t_death_c: Isotherm temperature [C]
        :param v_min: minimum tool speed [mm/s]
        :param v_max: maximum tool speed [mm/s]
        """
        self._tvp_template = None
        self.v_min = v_min
        self.v_max = v_max
        self.v = v_min

        self._width_adaptation: ScalarFirstOrderAdaptation = ScalarFirstOrderAdaptation(0, 0, 0.2,
                                                            1e-2, 1e-2)
        self._width_meas = 0
        # self._deflection_adaptation = FirstOrderAdaptation(0, 0.01, 1,
        #                                                    0, 1)
        self._deflection_adaptation: ScalarFirstOrderAdaptation | ScalarLinearAlgabraicAdaptation  = ScalarLinearAlgabraicAdaptation(0, 1e-3)
        self._deflection_meas = 0

        self.k_tool_n_per_mm = 1.3 # N/mm
        self.thermal_px_per_mm = 5.1337

        k = 0.24 # W/M -K
        d = 50e-3 # M thickness
        q = 20 # W
        self.Tc = 2 * np.pi * k * d * (t_death_c - 25) / q

    @property
    def alpha_hat(self):
        """
        Thermal Diffusivity Estimate
        """
        return self._width_adaptation.b

    @property
    def a_hat(self):
        """
        Thermal Time Constant Estimate
        """
        return self._width_adaptation.a

    @property
    def width_hat_mm(self):
        """
        Isotherm Width Estimate [mm]
        """
        return self._width_adaptation.state_estimate

    @property
    def d_hat(self):
        """
        Tool Damping Estimate
        """
        return self._deflection_adaptation.b

    @property
    def deflection_hat_mm(self):
        """
        Tool Deflection Estimate [mm]
        """
        return self._deflection_adaptation.state_estimate


    def setup_mpc(self):
        self.model = do_mpc.model.Model(model_type=self.model_type)
        self._width = self.model.set_variable(var_type='_x', var_name='width', shape=(1, 1))  # isotherm width [mm]
        self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(1, 1))  # tool speed [mm/s]
        self._a = self.model.set_variable(var_type='_tvp', var_name='a', shape=(1, 1))  # thermal time constant [s]
        self._d = self.model.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # tool damping [s]
        self._alpha = self.model.set_variable(var_type='_tvp', var_name='alpha', shape=(1, 1))  # thermal time constant [s]
        if isinstance(self._deflection_adaptation, ScalarLinearAlgabraicAdaptation):
            self._deflection = self.model.set_variable(var_type='_z', var_name='deflection', shape=(1, 1)) # deflection [mm]
            self.model.set_alg('deflection', expr=self._deflection - self._d * self._u)
        else:
            self._deflection = self.model.set_variable(var_type='_x', var_name='deflection', shape=(1, 1)) # deflection [mm]
            self.model.set_rhs('deflection', -self._deflection_adaptation.a * self._deflection + self._d * self._u)

        self._deflection_meas = self.model.set_variable(var_type='_tvp', var_name='deflection_measurement', shape=(1, 1)) # deflection [mm]
        self._width_estimate = self.model.set_variable(var_type='_tvp', var_name='width_estimate', shape=(1, 1))

        _, s, _ = lqr(-self.a_hat, self.alpha_hat, self.qw, self.r)
        self.model.set_rhs('width', -self._a * self._width + ymax(self._alpha, self._u, self.Tc))
        self.model.set_expression('terminal_cost', s * (self._width ** 2))
        self.model.set_expression('running_cost', self.qd * self._deflection ** 2 +
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
        self.mpc.set_nl_cons('width', self._width, ub=5, soft_constraint=True)
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
        self.mpc.set_tvp_fun(self.tvp_fun)
        self.mpc.setup()
        self.mpc.set_initial_guess()


    def tvp_fun(self, t_now):
        d = self._deflection_meas if isinstance(self._deflection_adaptation, ScalarLinearAlgabraicAdaptation) else self.deflection_hat_mm
        self._tvp_template['_tvp', :] = np.array([self.a_hat,
                                                  self.d_hat,
                                                  self.alpha_hat,
                                                  d,
                                                  self.width_hat_mm])
        return self._tvp_template

    def find_speed(self, width_mm, deflection_mm) -> float:
        """
        Find the optimal tool speed using MPC
        """
        # self.mpc.z0['deflection'] = np.array([deflection_mm])
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
        self._deflection_adaptation.update(deflection_mm, self.v)
        self._width_adaptation.update(width_mm, ymax(1, self.v, self.Tc))

        self.v = self.find_speed(width_mm, deflection_mm)

        return self.v

class OnlineVelocityOptimizer:
    k_tool = 1.3 # N/mm

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
        self.adaptive_mpc.k_tool_n_per_mm = self.k_tool
        self.adaptive_mpc.setup_mpc()

        self._dt = 1 / 24
        self.t_death_c = t_death_c
        self._width_kf = KalmanFilter(dim_x=1, dim_z=1, dim_u=1)
        self._width_kf.x = np.array([0])
        self._width_kf.F = np.array([[self.adaptive_mpc.a_hat]])
        self._width_kf.B = np.array([[self.adaptive_mpc.alpha_hat]])
        self._width_kf.H = np.array([[1]])
        self._width_kf.P *= 10
        self._width_kf.R = 4
        self._width_kf.Q = 50

        self._pos_kf_init = False
        self._pos_init = None
        self._deflection_mm = 0
        self._pos_kf = None

        self.ellipse = None

        self.neutral_tip_pos = None
        self._neutral_tip_candidates = []

    @property
    def width_mm(self):
        return self._width_kf.x[0]

    @property
    def controller_v_mm_s(self):
        return self.adaptive_mpc.v

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
        self._pos_kf = KalmanFilter(dim_x=4, dim_z=2, dim_u=1)
        self._pos_kf.x = np.array([pos[0], 0, pos[1], 0])
        self._pos_kf.F = np.array([[1, self._dt, 0, 0],
                                   [0, damping_ratio, 0, 0],
                                   [0, 0, 1, self._dt],
                                   [0, 0, 0, damping_ratio]])
        self._pos_kf.H = np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0]])
        self._pos_kf.P *= 4
        self._pos_kf.R = 9 * np.eye(2)
        self._pos_kf.Q = Q_discrete_white_noise(dim=4, dt=self._dt, var=36)

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

        z, self.ellipse = cv_isotherm_width(frame, self.t_death_c)
        # z = np.max(np.sum(frame > self.t_death, axis=0)) / 2


        self._width_kf.predict(u=ymax(1, v, self.adaptive_mpc.Tc),
                               B=np.array([self.adaptive_mpc.alpha_hat if self.adaptive_mpc.alpha_hat > 0 else 0]),
                               F=np.array([self.adaptive_mpc.a_hat if self.adaptive_mpc.a_hat > 0 else 0]))
        self._width_kf.update(z / self.thermal_px_per_mm)
        v = self.adaptive_mpc.update(deflection_mm, self.width_mm)
        if abs(self.adaptive_mpc.width_hat_mm) > 1e3:
            print(f"Unstable system: width_estimate: {self.adaptive_mpc.width_hat_mm:.2f}")
            raise ValueError("Unstable system")
        return v, self.ellipse

