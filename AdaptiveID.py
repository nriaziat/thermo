from filterpy.kalman import UnscentedKalmanFilter, unscented_transform, MerweScaledSigmaPoints
import numpy as np
from models import ymax, Tc, MaterialProperties
from abc import ABC, abstractmethod

def proj(theta: float, y: float)->float:
    if theta > 0:
        return y
    elif y > 0:
        return y
    else:
        return 0

class ScalarFirstOrderAdaptation:
    """
    Adaptive Parameters
    Identify \hat{a}, \hat{b} for first order system in the form
    \dot{x} = -a x + b u
    """
    am = 1
    def __init__(self, x0: float, a0: float, b0:float, gamma_a:float=1e-2, gamma_b:float=1e-2, regularize_input:bool=True):
        """
        Initialize the adaptive parameters
        :param x0: Initial state estimate
        :param a0: Initial a
        :param b0: Initial b
        :param gamma_a: Learning rate for a
        :param gamma_b: Learning rate for b
        """
        assert a0 >= 0, "a cannot be negative (for stability)"
        assert gamma_a >= 0, "gamma_a cannot be negative"
        assert gamma_b >= 0, "gamma_b cannot be negative"
        self.state_estimate = x0
        self.a = a0
        self.b = b0
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b
        self._regularize_input = regularize_input

    def update(self, measurement, u) -> tuple:
        """
        Update the adaptive parameters
        :param measurement: Measurement
        :param u: Input
        :return: Tuple of updated state estimate, a, b
        """
        if measurement < 0:
            measurement = 0
        error = self.state_estimate - measurement
        self.state_estimate += -self.am * error - self.a * measurement + self.b * u
        if self.state_estimate < 0:
            self.state_estimate = 0
        if self._regularize_input:
            measurement = measurement / (1+abs(measurement))
            u = u / (1+abs(u))
        self.a += -self.gamma_a * proj(self.a, -error * measurement)
        if self.a < 0:
            self.a = 0.01
        self.b += -self.gamma_b * proj(self.b, error * u)
        return self.state_estimate, self.a, self.b

class ScalarLinearAlgabraicAdaptation:
    def __init__(self, b: float, gamma: float=1e-2):
        """
        Initialize the adaptive parameters y= b u
        :param b: Initial b
        :param gamma: Learning rate for b
        """
        assert gamma >= 0, "gamma cannot be negative"
        self.b = b
        self.gamma = gamma
        self.state_estimate = 0

    def update(self, measurement, u) -> float:
        """
        Update the adaptive parameters
        :param measurement: Measurement
        :param u: Input
        :return: Updated c
        """
        if measurement < 0:
            measurement = 0
        self.state_estimate = self.b * u
        error = self.state_estimate - measurement
        # self.b += self.gamma * proj(self.b, -error * u)
        self.b += -self.gamma * error * u / (1+abs(u))
        return self.b



class UKFIdentification(ABC):
    def __init__(self, w0: np.array, labels: list[str]):
        """
        :param w0: Initial parameter estimate
        """
        n = len(w0)
        assert len(labels) == len(w0), "Labels must be the same length as the parameter estimate"
        self._labels = labels
        self.points = MerweScaledSigmaPoints(n=n,
                                             alpha=1e-3,
                                             beta=2,
                                             kappa=0)
        self.ukf = UnscentedKalmanFilter(dim_x=n,
                                         dim_z=1,
                                         dt=1/24,
                                         hx=self.hx,
                                         fx=self.fx,
                                         points=self.points)
        self.ukf.x = w0
        self.ukf.P = np.eye(n) * 0.1
        self.ukf.R = 0.1
        self.ukf.Q = np.eye(n) * 0.1
        self._alpha_sq =1**2 # Fading memory factor
        self._data = {label: [] for label in labels}

    @property
    def labels(self) -> list[str]:
        """
        List of parameter labels
        """
        return self._labels

    @abstractmethod
    def hx(self, x, **kwargs) -> np.array:
        """
        Measurement function
        """
        pass

    @abstractmethod
    def fx(self, x, dt) -> np.array:
        """
        State transition function
        """
        pass

    def update(self, measurement, **kwargs) -> tuple:
        """
        Update the adaptive parameters
        :param measurement: Measurement
        :param kwargs: Additional keyword arguments to pass to the measurement function
        :return: Tuple of updated state estimate, a, b
        """
        self.ukf.predict(**kwargs)
        # self.ukf.P *= self._alpha_sq
        self.ukf.update(measurement, **kwargs)
        self._data = {label: self._data[label] + [value] for label, value in zip(self.labels, self.ukf.x)}
        return self.ukf.x

    @property
    def data(self) -> dict[str, list]:
        """
        Dictionary of parameter labels and values
        """
        return self._data


class DeflectionAdaptation(UKFIdentification):
    def __init__(self, w0: np.array, labels):
        """
        :param w0: Initial parameter estimate [defl, defl_rate, k, b, c]
        """
        super().__init__(w0, labels)
        assert len(labels) == len(w0), "Labels must be the same length as the parameter estimate"
        self.ukf.P = np.diag([4, 4, 10, 10, 10])
        self.ukf.R = 4
        self.ukf.Q = np.diag([4, 10, 10, 10, 10])

    @property
    def b(self):
        return self.ukf.x[3]

    @property
    def c(self):
        return self.ukf.x[4]

    @property
    def k(self):
        return self.ukf.x[2]

    def hx(self, x, **kwargs):
        """
        Measurement function
        """
        # v = kwargs.get("v")
        # return np.array([x[0] * np.exp(-x[1] / v)])
        return np.array([x[0]])

    def fx(self, x, dt, **kwargs):
        """
        State transition function
        """
        v = kwargs.get("v")
        x[0] += x[1] * dt
        F = x[3] * np.exp(-x[4] / v)
        x[1] += (-x[2] * x[0] + F) * dt
        return x

class ThermalAdaptation(UKFIdentification):
    def __init__(self, w0: np.array, labels):
        """
        :param w0: Initial parameter estimate  [q]
        """
        super().__init__(w0, labels)
        self.ukf.P = np.diag([0.5*w0[0]])
        self.ukf.R = 4
        self.ukf.Q = 1
        self.Cp = 3421
        self.rho = 1090e-9
        self.k = 0.46e-3

    # @property
    # def k(self):
    #     return self.ukf.x[0]

    @property
    def q(self):
        return self.ukf.x

    @property
    def alpha(self):
        return self.k / (self.rho * self.Cp)

    def hx(self, x, **kwargs):
        """
        Measurement function
        """
        x = np.abs(x)
        v = kwargs.get("v")
        dT = kwargs.get("dT")
        alpha = self.k / (self.rho * self.Cp)
        material = MaterialProperties(k=self.k, rho=self.rho, Cp=self.Cp)
        y = ymax(alpha, v, Tc(dT, material, x[0]))
        if np.isinf(y) or np.isnan(y):
            return self.ukf.y
        return np.array([y])

    def fx(self, x, dt, **kwargs):
        """
        State transition function
        """
        return x