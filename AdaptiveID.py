from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np
from models import ymax, Tc, MaterialProperties
import scipy
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

class BoundedMerweScaledSigmaPoints(MerweScaledSigmaPoints):
    def __init__(self, n, alpha, beta, kappa, low=0, high=np.inf):
        super().__init__(n, alpha, beta, kappa, sqrt_method=self.sqrt)
        if np.isscalar(low):
            low = np.array([low]*n)
        self.low = low
        if np.isscalar(high):
            high = np.array([high]*n)
        self.high = high

    def sqrt(self, x):
        try:
            result = scipy.linalg.cholesky(x)
        except np.linalg.LinAlgError:
            x = (x + x.T) / 2 + np.eye(x.shape[0]) * 1e-6
            result = scipy.linalg.cholesky(x)
        return result


    def sigma_points(self, x, P):
        """
        Computes the sigma points for an unscented Kalman filter
        using the formulation of Wan and van der Merwe. This method
        is less efficient than the simple method, but is more stable
        for near-linear systems.
        """
        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)
        sigmas = np.zeros((2*n+1, n))
        # sigmas[0] = x
        sigmas[0] = np.minimum(self.high - 1e-6, np.maximum(self.low + 1e-6, x))
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = np.minimum(self.high-1e-6, np.maximum(self.subtract(x, -U[k]), self.low + 1e-6))
            sigmas[n+k+1] = np.minimum(self.high-1e-6, np.maximum(self.subtract(x, U[k]), self.low + 1e-6))


        return sigmas



class UKFIdentification(ABC):
    def __init__(self, w0: np.array, dim_z: int, labels: list[str], lower_bounds: np.ndarray = -np.inf,
                 upper_bounds: np.ndarray = np.inf):
        """
        :param w0: Initial parameter estimate
        """
        n = len(w0)
        assert len(labels) == len(w0), "Labels must be the same length as the parameter estimate"
        self._labels = labels

        self.points = BoundedMerweScaledSigmaPoints(n=n,
                                             alpha=1e-3,
                                             beta=2,
                                             kappa=0,
                                             low=lower_bounds,
                                             high=upper_bounds)


        self.ukf = UnscentedKalmanFilter(dim_x=n,
                                         dim_z=dim_z,
                                         dt=1/24,
                                         hx=self.hx,
                                         fx=self.fx,
                                         points=self.points)
        self.ukf.x = w0
        self._data = {label: [(w0[i], w0[i], w0[i])] for i, label in enumerate(self.labels)}

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
        self.ukf.update(measurement, **kwargs)
        ci_low, ci_high = self.ukf.x - np.sqrt(np.diag(self.ukf.P)), self.ukf.x + 3 * np.sqrt(np.diag(self.ukf.P))
        self._data = {label: self._data[label] + [(ci_low[i], self.ukf.x[i], ci_high[i])] for i, label in enumerate(self.labels)}
        return self.ukf.x

    @property
    def data(self) -> dict[str, list[tuple]]:
        """
        Dictionary of parameter labels and values (with confidence intervals) [label: [(value, low_ci, high_ci)]]
        """
        return self._data


class DeflectionAdaptation(UKFIdentification):
    def __init__(self, w0: np.array, labels):
        """
        :param w0: Initial parameter estimate [x, y, xd, yd, k, b, c_defl]
        """
        super().__init__(w0, dim_z=2, labels=labels, lower_bounds=np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0]))
        assert len(labels) == len(w0), "Labels must be the same length as the parameter estimate"
        self.ukf.P = np.diag([0, 0, 0, 0, 1, 1, 1])
        self.ukf.R = 1 * np.eye(2)
        self.ukf.Q = np.block([[Q_discrete_white_noise(dim=2, dt=1/24, var=0.1, block_size=2), np.zeros((4, 3))],
                                 [np.zeros((3, 4)), np.diag([1, 1, 1])]])

    @property
    def b(self):
        return self.ukf.x[3]

    @property
    def c_defl(self):
        return self.ukf.x[4]

    @property
    def k(self):
        return self.ukf.x[2]

    @property
    def defl_mm(self):
        return np.linalg.norm(self.ukf.x[0:2])

    def hx(self, x, **kwargs):
        """
        Measurement function
        """
        # v = kwargs.get("v")
        # return np.array([x[0] * np.exp(-x[1] / v)])
        return np.array([x[0], x[1]])

    def fx(self, x, dt, **kwargs):
        """
        State transition function
        """
        v = kwargs.get("v")
        x[0] += x[2] * dt
        x[1] += x[3] * dt
        v_tool = np.sqrt(x[2] ** 2 + x[3] ** 2) * -np.sign(x[2])
        if 0 < v_tool < v:
            F = x[5] * np.exp(-x[6] / (v - v_tool + 1e-6)) + x[4] * np.linalg.norm(x[0:2])
        else:
            F = x[5] * np.exp(-x[6] / (v + 1e-6))  + x[4] * np.linalg.norm(x[0:2])
        x[0:2] -= F * np.array([x[0], x[1]]) * dt
        return x

class ThermalAdaptation(UKFIdentification):
    def __init__(self, w0: np.array, labels):
        """
        :param w0: Initial parameter estimate  [w, q, k]
        """
        super().__init__(w0, dim_z=1, labels=labels, lower_bounds=np.array([0, 0, 0]))
        self.ukf.P = np.diag([0, 5, 1e-8])
        self.ukf.R = 1
        self.ukf.Q = np.diag([1, 1, 1e-8])
        self.Cp = 3421
        self.rho = 1090e-9
        # self.k = 0.46e-3

    @property
    def k(self):
        return self.ukf.x[2]

    @property
    def q(self):
        return self.ukf.x[1]

    @property
    def alpha(self):
        return self.k / (self.rho * self.Cp)

    @property
    def w_mm(self):
        return self.ukf.x[0]

    def hx(self, x, **kwargs):
        """
        Measurement function
        """
        return np.array([x[0]])

    def fx(self, x, dt, **kwargs):
        """
        State transition function
        """
        v = kwargs.get("v")
        dT = kwargs.get("dT")
        material = MaterialProperties(k=x[2], rho=self.rho, Cp=self.Cp)
        y = ymax(v, material, x[1], dT)
        if (Ro:= 1/Tc(v, material, x[1])) < 0.3:
            print(f"Regime IV, Ro = {Ro:.2f}")
        if np.isinf(y) or np.isnan(y):
            return self.ukf.x
        x[0] = y
        return x