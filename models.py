import do_mpc
from abc import ABC, abstractmethod
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from utils import find_tooltip
from dataclasses import dataclass
from scipy.special import k0

@dataclass
class MaterialProperties:
    """
    Material properties for thermal modeling
    rho: density [kg/mm^3]
    Cp: specific heat capacity [J/kgK]
    k: conductivity [W/mmK]
    alpha: thermal diffusivity [mm^2/s]
    """
    rho: float
    Cp: float
    k: float
    @property
    def alpha(self):
        return self.k / (self.rho * self.Cp)

humanTissue = MaterialProperties(rho=1090e-9, Cp=3421, k=0.46e-3)
hydrogelPhantom = MaterialProperties(rho=1310e-9, Cp=3140, k=0.6e-3)

class ToolTipKF(KalmanFilter):
    def __init__(self, damping_ratio, pos=None):
        super().__init__(dim_x=4, dim_z=2, dim_u=1)
        self._init = False
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
        self._neutral_tip_candidates = []
        self.neutral_tip_pos = None

    @property
    def pos_mm(self):
        return self.x[0], self.x[2]

    def update_with_measurement(self, frame: np.ndarray[float]) -> tuple[float, float]:
        """
        Update the tool deflection based on the current frame
        :param frame: Temperature field from the camera.
        :return: deflection [px], deflection rate [px/s]
        """
        # if self._init:
        #     last_tool_tip = self.x[0], self.x[2]
        # else:
        #     last_tool_tip = None
        tool_tip = find_tooltip(frame, 50)
        if tool_tip is None:
            return 0, 0
        elif not self._init:
            self._neutral_tip_candidates.append(tool_tip)
            if len(self._neutral_tip_candidates) > 10:
                self.neutral_tip_pos = np.mean(self._neutral_tip_candidates, axis=0)
                self.x = np.array([self.neutral_tip_pos[0], 0, self.neutral_tip_pos[1], 0])
                self._init = True
        else:
            self.predict()
            self.update(tool_tip)
            if self.x[0] < self.neutral_tip_pos[0]:
                self.neutral_tip_pos = self.x[0], self.neutral_tip_pos[1]
            deflection = np.array(
                [self.x[0] - self.neutral_tip_pos[0], self.x[2] - self.neutral_tip_pos[1]])
            ddeflection = np.array([self.x[1], self.x[3]])
            return np.linalg.norm(deflection), np.linalg.norm(ddeflection) * np.sign(np.dot(deflection, ddeflection))
        return 0, 0


class ElectrosurgeryModel(ABC, do_mpc.model.Model):

    def __init__(self, model_type: str = "continuous"):
        super().__init__(model_type)
        self.n_isotherms: int = 1

    @property
    @abstractmethod
    def deflection_mm(self) -> float:
        pass

    @property
    @abstractmethod
    def isotherm_widths_mm(self) -> float | np.ndarray[float]:
        pass

    @property
    @abstractmethod
    def isotherm_temps(self) -> np.ndarray[float]:
        pass

    @abstractmethod
    def set_cost_function(self, qw: float, qd: float):
        pass

class MultiIsothermModel(ElectrosurgeryModel):
    """
    This model uses the dynamics of multiple isotherms to predict the width of each isotherm.
    """

    Ta = 20  # Ambient Temperature [C]
    # P = 45  # Tool Power [W]
    c_defl = 0.5  # deflection damping constant [s]
    t_death = 45  # death temperature [C]

    def __init__(self, n_isotherms: int, material: MaterialProperties):
        """
        :param n_isotherms: number of isotherms. Isotherms are evenly spaced such that the middle isotherm is at the death temperature.
        """
        super().__init__()
        self._isotherm_widths = []
        self.n_isotherms = n_isotherms
        self.dT = (self.t_death - self.Ta) / (np.ceil(self.n_isotherms / 2))
        self._isotherm_temps = np.linspace(self.Ta + self.n_isotherms * self.dT,
                                           self.Ta + self.dT,
                                           self.n_isotherms)
        self._isotherm_width_measurement = 0
        self._deflection_mm = 0
        self._material: MaterialProperties = material
        print(f"Isotherm Levels: {self.isotherm_temps}")
        for i in range(self.n_isotherms):
            self._isotherm_widths.append(self.set_variable(var_type='_x', var_name=f'width_{i}', shape=(1, 1))) # isotherm width [mm]
        self._deflection = self.set_variable(var_type='_z', var_name='deflection', shape=(1, 1)) # deflection [mm]
        self._defl_meas = self.set_variable(var_type='_tvp', var_name='defl_meas', shape=(1, 1))  # deflection measurement [mm/s]
        self._d = self.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # adaptive tool damping
        self._P = self.set_variable(var_type='_tvp', var_name='P', shape=(1, 1))  # effective power [W]
        self._velocity = self.set_variable(var_type='_u', var_name='u', shape=(1, 1))  # tool speed [mm/s]
        DIVIDE_BY_ZERO = 1e-9
        L = 2 * self._material.alpha / self._velocity
        S = 4 * self._material.alpha / self._velocity ** 2
        self.set_alg('deflection', expr=self._deflection - self._d * np.exp(-self.c_defl / self._velocity))

        self.set_rhs('width_0', (-2 * self._material.alpha / (self._isotherm_widths[0] + DIVIDE_BY_ZERO)) * (self._isotherm_widths[1] / (DIVIDE_BY_ZERO + self._isotherm_widths[1] - self._isotherm_widths[0])) +
                     self._P * np.exp(-self._isotherm_widths[0] / L) / (np.pi * self._material.Cp * self._material.rho * self.dT * self._isotherm_widths[0] ** 2 + DIVIDE_BY_ZERO) *
                     (1 + (self._isotherm_widths[0] / L)) -
                     (self.isotherm_temps[0] - self.Ta) / (S * self.dT) * (self._isotherm_widths[1] - self._isotherm_widths[0]))
        for i in range(1, self.n_isotherms - 1):
            self.set_rhs(f'width_{i}', (-self._material.alpha / (self._isotherm_widths[i] + DIVIDE_BY_ZERO)) *
                         ((self._isotherm_widths[i + 1] / (DIVIDE_BY_ZERO + self._isotherm_widths[i + 1] - self._isotherm_widths[i])) -
                          (self._isotherm_widths[i - 1] / (DIVIDE_BY_ZERO + self._isotherm_widths[i] - self._isotherm_widths[i - 1]))) -
                         (self.isotherm_temps[i] - self.Ta) / (2 * S * self.dT) *
                         (self._isotherm_widths[i + 1] - self._isotherm_widths[i - 1]))
        self.set_rhs(f'width_{self.n_isotherms - 1}', (-self._material.alpha / (DIVIDE_BY_ZERO + self._isotherm_widths[-1])) *
                     (self._isotherm_widths[-1] - 2 * self._isotherm_widths[-2]) / (DIVIDE_BY_ZERO + self._isotherm_widths[-1] - self._isotherm_widths[-2]) -
                     (self._isotherm_widths[-1] - self._isotherm_widths[-2]) / S)

    @property
    def material_properties(self) -> MaterialProperties:
        return self._material

    def set_cost_function(self, qw: float, qd: float):
        print('Setting cost function')
        # self.set_expression(expr_name='lterm', expr=qw * (sum(self._isotherm_widths[i]**2 for i in range(self.n_isotherms // 2 + 1)) - sum(self._isotherm_widths[i] for i in range(self.n_isotherms // 2 + 1, self.n_isotherms))) + qd * self._deflection)
        self.set_expression(expr_name='lterm', expr=qw * (self._isotherm_widths[self.n_isotherms // 2]) + qd *self._deflection + 10/self._velocity**2)
        # self.set_expression(expr_name='mterm', expr=qw * (sum(self._isotherm_widths[i]**2 for i in range(self.n_isotherms // 2 + 1)) - sum(self._isotherm_widths[i] for i in range(self.n_isotherms // 2 + 1, self.n_isotherms))))
        self.set_expression(expr_name='mterm', expr=qw * (self._isotherm_widths[self.n_isotherms // 2]))

    @property
    def deflection_mm(self) -> float:
        return self._deflection

    @deflection_mm.setter
    def deflection_mm(self, value: float):
        assert value >= 0, "Deflection must be non-negative"
        self._deflection = value

    @property
    def isotherm_widths_mm(self) -> float | np.ndarray[float]:
        return self._isotherm_width_measurement

    @property
    def isotherm_temps(self) -> np.ndarray[float]:
        return self._isotherm_temps


class PseudoStaticModel(ElectrosurgeryModel):
    """
    This model uses a pseudo-static approximation to predict the width of the isotherm.
    """
    Ta = 20  # Ambient Temperature [C]
    # P = 45  # Tool Power [W]
    c_defl = 0.5  # deflection damping constant [s]
    t_death = 45  # death temperature [C]

    def __init__(self, material: MaterialProperties):
        super().__init__()
        self._material = material
        self.n_isotherms = 1
        self._isotherm_width_measurement = 0
        self._width = self.set_variable(var_type='_x', var_name='width_0', shape=(1, 1))
        self._deflection = self.set_variable(var_type='_z', var_name='deflection', shape=(1, 1)) # deflection [mm]
        self._defl_meas = self.set_variable(var_type='_tvp', var_name='defl_meas', shape=(1, 1))  # deflection measurement [mm/s]
        self._d = self.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # adaptive tool damping\
        self._velocity = self.set_variable(var_type='_u', var_name='u', shape=(1, 1))
        self._P = self.set_variable(var_type='_tvp', var_name='P', shape=(1, 1))  # effective power [W]
        self.set_rhs('width_0', 10 * ((self._P * 2 * self._material.alpha / self._velocity) * np.sqrt(self._velocity / (4 * np.pi * self._material.k * self._material.alpha * (self.t_death - self.Ta))) - self._width))
        self.set_alg('deflection', expr=self._deflection - self._d * np.exp(-self.c_defl / self._velocity))

    def set_cost_function(self, qw: float, qd: float):
        # self.set_expression(expr_name='lterm', expr=qw * (sum(self._isotherm_widths[i]**2 for i in range(self.n_isotherms // 2 + 1)) - sum(self._isotherm_widths[i] for i in range(self.n_isotherms // 2 + 1, self.n_isotherms))) + qd * self._deflection)
        self.set_expression(expr_name='lterm', expr=qw * (self._width) + qd *self._deflection + 10/self._velocity**2)
        # self.set_expression(expr_name='mterm', expr=qw * (sum(self._isotherm_widths[i]**2 for i in range(self.n_isotherms // 2 + 1)) - sum(self._isotherm_widths[i] for i in range(self.n_isotherms // 2 + 1, self.n_isotherms))))
        self.set_expression(expr_name='mterm', expr=qw * (self._width))

    @property
    def deflection_mm(self) -> float:
        return self._deflection

    @deflection_mm.setter
    def deflection_mm(self, value: float):
        assert value >= 0, "Deflection must be non-negative"
        self._deflection = value

    @property
    def isotherm_widths_mm(self) -> float | np.ndarray[float]:
        return self._isotherm_width_measurement

    @property
    def isotherm_temps(self) -> np.ndarray[float]:
        return np.array([self.t_death])

