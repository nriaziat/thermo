import do_mpc
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from functools import lru_cache

DIVIDE_BY_ZERO = 1e-6

@dataclass
class MaterialProperties:
    """
    Material properties for thermal modeling
    rho: density [kg/mm^3]
    Cp: specific heat capacity [J/kgK]
    k: conductivity [W/mmK]
    alpha: thermal diffusivity [mm^2/s]
    """
    _rho: float
    _Cp: float
    _k: float
    _alpha: float = None
    @property
    def alpha(self):
        if self._alpha is None:
            return self.k / (self.rho * self.Cp)
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """
        Set the thermal diffusivity
        """
        if value < 0:
            self._alpha = 1e-6
        else:
            self._alpha = value

    @property
    def rho(self):
        """
        Density [kg/mm^3]
        """
        return self._rho

    @property
    def Cp(self):
        """
        Specific heat capacity [J/kgK]
        """
        return self._Cp

    @property
    def k(self):
        """
        Conductivity [W/mmK]
        """
        return self._k

    @rho.setter
    def rho(self, value):
        if value < 0:
            self._rho = 1e-9
        else:
            self._rho = value

    @Cp.setter
    def Cp(self, value):
        if value < 0:
            self._Cp = 1e-9
        else:
            self._Cp = value

    @k.setter
    def k(self, value):
        if value < 0:
            self._k = 0
        else:
            self._k = value


exp_gamma = np.exp(0.5772)
@lru_cache
def F(Tc):
    if Tc < 1/0.3856:
        return np.exp(-Tc) * (1 + (1.477 * Tc) ** 1.407) ** 0.7107
    return (1 + (1.477 * Tc) ** -1.407) ** 0.7107

# @lru_cache
def ymax(u, material: MaterialProperties, q, dT):
    d = 25 # tissue thickness [mm]
    T_star = Tc(dT, material, q, d)
    if T_star < 1/0.3856:
        return 1/np.sqrt(2*np.pi*np.exp(1)) * q / (u * d * material.rho * material.Cp * dT) * F(T_star)
    return 4 * material.alpha / (u * exp_gamma) * F(T_star)

def Tc(dT: float, material: MaterialProperties, q: float, d: float):
    """
    Calculate the dimensionless temperature.
    :param dT: Temperature difference [K]
    :param material: Material properties
    :param q: Power [W]
    :param d: tissue thickness [mm]
    :return: dimensionless temperature
    """
    return 2 * np.pi * material.k * d * dT / q

humanTissue = MaterialProperties(_rho=1090e-9, _Cp=3421, _k=0.46e-3)
hydrogelPhantom = MaterialProperties(_rho=1310e-9, _Cp=3140, _k=0.6e-3)

class ElectrosurgeryCostMinimizationModel(ABC):

    def __init__(self):
        self.vmin = 0
        self.vmax = 10

    @abstractmethod
    def find_optimal_velocity(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def cost_function(self, *args, **kwargs) -> float:
        pass


class SteadyStateMinimizationModel(ElectrosurgeryCostMinimizationModel):
    t_death: float = 60  # death temperature [C]
    Ta: float = 20 # ambient temperature [C]

    def __init__(self,
                 qw = 1, qd = 1, r = 0.1):
        super().__init__()
        self._isotherm_measurement_mm = 0
        self._deflection_measurement_mm = 0
        self._P = 40
        self._u0 = (self.vmin + self.vmax) / 2
        self.r = r  # input penalty
        self.qw = qw
        self.qd = qd
        self.v_star = 7

    def cost_function(self, v: float, material: MaterialProperties, c_defl: float, q: float) -> float:
        return (self.qw * self.isotherm_width_model(material, v, q) + self.qd * self.deflection_model(v, c_defl) +
                self.r * 0 * (v-self._u0)**2 + self. r * np.max([0, self.v_star - v])**2)

    def isotherm_width_model(self, material: MaterialProperties,
                             v: float, q: float) -> float:
        return ymax(v, material, q, self.t_death - self.Ta)

    @staticmethod
    def deflection_model(v: float, c_defl: float) -> float:
        return 10 * np.exp(-c_defl / v)

    def find_optimal_velocity(self, material: MaterialProperties, c_defl: float, q: float):
        res = minimize_scalar(self.cost_function, bounds=[self.vmin, self.vmax], method='bounded', args=(material, c_defl, q))
        self._u0 = res.x
        return self._u0


class ElectrosurgeryMPCModel(ABC, do_mpc.model.Model):

    def __init__(self, material: MaterialProperties, model_type: str = "continuous"):
        super().__init__(model_type)
        self.n_isotherms: int = 1
        self.material = material


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

class PseudoStaticModel(ElectrosurgeryMPCModel):
    """
    This model uses a pseudo-static approximation to predict the width of the isotherm.
    """
    Ta = 20  # Ambient Temperature [C]
    # P = 45  # Tool Power [W]
    c_defl = 0.5  # deflection damping constant [s]
    t_death = 45  # death temperature [C]

    def __init__(self, material: MaterialProperties):
        super().__init__(material)
        self.n_isotherms = 1
        self._isotherm_width_measurement = 0
        self._width = self.set_variable(var_type='_x', var_name='width_0', shape=(1, 1))
        self._deflection = self.set_variable(var_type='_z', var_name='deflection', shape=(1, 1)) # deflection [mm]
        self._defl_meas = self.set_variable(var_type='_tvp', var_name='defl_meas', shape=(1, 1))  # deflection measurement [mm/s]
        self._d = self.set_variable(var_type='_tvp', var_name='d', shape=(1, 1))  # adaptive tool damping
        self._velocity = self.set_variable(var_type='_u', var_name='u', shape=(1, 1))
        self._P = self.set_variable(var_type='_tvp', var_name='P', shape=(1, 1))  # effective power [W]
        self.set_rhs('width_0', 10 * ((self._P * 2 * self.material.alpha / (self._velocity + DIVIDE_BY_ZERO) * np.sqrt(self._velocity / (4 * np.pi * self.material.k * self.material.alpha * (self.t_death - self.Ta))) - self._width)))
        self.set_alg('deflection', expr=self._deflection - self._d * np.exp(-self.c_defl / (self._velocity + DIVIDE_BY_ZERO)))
        self._tip_lead_dist = self.set_variable('_x', "tip_lead_dist", shape=(1, 1))
        self.set_rhs('tip_lead_dist', 10 * (-self.material.alpha/(self._velocity + DIVIDE_BY_ZERO) * np.log(self._P/(3 * np.pi * self.material.k * (100 - self.Ta))) - self._tip_lead_dist))
        # self.set_rhs('tip_lead_dist', 10 * (self._material.alpha/(2*self._velocity) * lambertw(8*self._material.k**2*np.pi**3*(self.t_death-self.Ta)**2/self._P**2, 0)  - self._tip_lead_dist))

    def set_cost_function(self, qw: float, qd: float):
        # self.set_expression(expr_name='lterm', expr=qw * (sum(self._isotherm_widths[i]**2 for i in range(self.n_isotherms // 2 + 1)) - sum(self._isotherm_widths[i] for i in range(self.n_isotherms // 2 + 1, self.n_isotherms))) + qd * self._deflection)
        self.set_expression(expr_name='lterm', expr=qw * (self._width) + qd * (self._deflection + (self._tip_lead_dist + 2.5)**2))
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
    def isotherm_widths_mm(self) -> np.ndarray[float]:
        return np.array([self._isotherm_width_measurement])

    @property
    def isotherm_temps(self) -> np.ndarray[float]:
        return np.array([self.t_death])

