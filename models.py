import do_mpc
from abc import ABC, abstractmethod
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from utils import find_tooltip

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
    P = 100  # Tool Power [W]
    rho = 1090e-9  # tissue density [kg/mm^3]
    Cp = 3421  # tissue specific heat capacity [J/kgK]
    k = 0.49e-3  # tissue conductivity [W/mmK]
    alpha = k / (rho * Cp)  # thermal diffusivity [mm^2/s]
    c_defl = 0.1  # deflection damping constant [s]

    def __init__(self, n_isotherms: int):
        """
        :param n_isotherms: number of isotherms
        :param dT: isotherm temperature step [C]
        """
        super().__init__()
        self.isotherm_widths = []
        self.n_isotherms = n_isotherms
        self.dT = (50 - self.Ta) / self.n_isotherms / 2
        self.isotherm_temps = np.linspace(self.Ta + self.n_isotherms * self.dT,
                                              self.Ta + self.dT,
                                              self.n_isotherms)
        for i in range(self.n_isotherms):
            self.isotherm_widths.append(self.set_variable(var_type='_x', var_name=f'width_{i}', shape=(1, 1))) # isotherm width [mm]
        self.deflection = self.set_variable(var_type='_z', var_name='deflection', shape=(1, 1)) # deflection [mm]
        self._d = self.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # adaptive tool damping
        self.velocity = self.set_variable(var_type='_u', var_name='u', shape=(1, 1))  # tool speed [mm/s]
        L = 2 * self.alpha / self.velocity
        S = 4 * self.alpha / self.velocity ** 2
        self.set_alg('deflection', expr=self.deflection - self._d * np.exp(-self.c_defl / self.velocity))

        self.set_rhs('width_0', (-2 * self.alpha / self.isotherm_widths[0]) * (self.isotherm_widths[1] / (self.isotherm_widths[1] - self.isotherm_widths[0])) +
                     self.P * np.exp(-self.isotherm_widths[0] / L) / (np.pi * self.rho * self.Cp * self.dT * self.isotherm_widths[0] ** 2) *
                     (1 + self.isotherm_widths[0] /L) -
                     (self.isotherm_temps[0] - self.Ta) / (S * self.dT) * (self.isotherm_widths[1] - self.isotherm_widths[0]))
        for i in range(1, self.n_isotherms - 1):
            self.set_rhs(f'width_{i}', (-self.alpha / self.isotherm_widths[i]) *
                         ((self.isotherm_widths[i + 1] / (self.isotherm_widths[i + 1] - self.isotherm_widths[i])) -
                          (self.isotherm_widths[i - 1] / (self.isotherm_widths[i] - self.isotherm_widths[i - 1]))) -
                         (self.isotherm_temps[i] - self.Ta) / (2 * S * self.dT) *
                         (self.isotherm_widths[i + 1] - self.isotherm_widths[i - 1]))
        self.set_rhs(f'width_{self.n_isotherms - 1}', (-self.alpha / self.isotherm_widths[-1]) *
                        (self.isotherm_widths[-1] - 2 * self.isotherm_widths[-2]) / (self.isotherm_widths[-1] - self.isotherm_widths[-2]) -
                        (self.isotherm_widths[-1] - self.isotherm_widths[-2]) / S)


    def set_cost_function(self, qw: float, qd: float):
        self.set_expression(expr_name='lterm', expr=qw * (self.isotherm_widths[self.n_isotherms // 2]) ** 2 + qd * self.deflection ** 2)
        self.set_expression(expr_name='mterm', expr=qw * (self.isotherm_widths[self.n_isotherms // 2]) ** 2)

    def deflection_mm(self) -> float:
        return self.deflection

    def isotherm_widths_mm(self) -> float | np.ndarray[float]:
        return self.isotherm_widths

    def isotherm_temps(self) -> np.ndarray[float]:
        return self.isotherm_temps


class PseudoStaticModel(ElectrosurgeryModel):
    """
    This model uses a pseudo-static approximation to predict the width of the isotherm.
    """

    def __init__(self):
        super().__init__()
        self.n_isotherms = 1

    @property
    def deflection_mm(self) -> float:
        pass

    @property
    def isotherm_widths_mm(self) -> float | np.ndarray[float]:
        pass

    def set_cost_function(self, qw: float, qd: float):
        pass

    def isotherm_temps(self) -> np.ndarray[float]:
        pass

