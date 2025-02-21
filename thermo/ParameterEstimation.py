from .Kalman import UKFIdentification
from filterpy.common import Q_discrete_white_noise
import numpy as np
from thermo.models import MaterialProperties, isotherm_width_model, cut_force_model

class DeflectionAdaptation(UKFIdentification):
    def __init__(self, w0: np.array,
                 labels: list[str],
                 px_per_mm:float= 1,
                 frame_size_px=(384, 288)):
        """
        :param w0: Initial parameter estimate [x, y, xd, yd, x_rest, y_rest, c_defl, k_tool, d_defl] in mm, mm/s, mm, mm, mm, mm, N/mm, N/mm, mm/s
        :param labels: List of parameter labels
        :param px_per_mm: Pixels per mm in the thermal frame
        :frame_size_px: Size of the thermal frame in pixels [width, height]
        """
        super().__init__(w0, dim_z=2, labels=labels, lower_bounds=np.array([0, 0, -np.inf, -np.inf, 0, 0, 0]),
                         upper_bounds=np.array([frame_size_px[0] / px_per_mm, frame_size_px[1] / px_per_mm, np.inf, np.inf, frame_size_px[0] / px_per_mm, frame_size_px[1] / px_per_mm, np.inf]))
        assert len(labels) == len(w0), "Labels must be the same length as the parameter estimate"
        self.kf.P = np.diag([2**2, 2**2, 5, 5, 8**2, 8**2, 1])
        self.kf.R = np.diag([1.2**2, 1.2**2])
        self.kf.Q = np.block([[Q_discrete_white_noise(dim=2, dt=1/24, var=6**2, block_size=2, order_by_dim=False), np.zeros((4, 3))],
                               [np.zeros((3, 4)), np.diag([0.001**2, 0.001**2, 1**2])]])

    @property
    def x(self):
        """
        Estimated state vector [x, y, xd, yd, x_rest, y_rest, c_defl, k_tool] in mm, mm/s, mm, mm, mm, mm, N/mm, N/mm
        """
        x = self.kf.x
        return x

    @property
    def c_defl(self):
        """
        Estimated deflection damping constant [s]
        """
        return self.kf.x[6]

    @property
    def defl_mm(self):
        """
        Distance between the estimated tip position and the estimated neutral tip position [mm]
        """
        d = self.kf.x[0:2] - self.kf.x[4:6]
        return np.linalg.norm(d)
        # return np.linalg.norm(d) * -np.sign(d[1])

    @property
    def defl_std(self):
        """
        Standard deviation of the deflection, computed with covariance on the tip position and neutral tip position [mm].
        """
        mu = self.kf.x[0:2] - self.kf.x[4:6]
        P = self.kf.P[0:2, 0:2] + self.kf.P[4:6, 4:6]
        EX2  = np.linalg.norm(mu) ** 2 + np.trace(P)
        VarX2 = 2 * np.trace(P @ P) + 4 * mu.T @ P @ mu
        varX = VarX2 / (4 * EX2)
        return np.sqrt(varX)

    @property
    def defl_hist_mm(self) -> list[float]:
        """
        Deflection history in mm, with most recent neutral tip position
        """
        return [np.linalg.norm(x[0:2] - self.kf.x[4:6]) for x in self._x_hist]

    @property
    def neutral_tip_mm(self) -> np.array:
        """
        Neutral tip position [x, y] in mm
        """
        return self.kf.x[4:6]

    def hx(self, x, **kwargs) -> np.array:
        """
        Measurement function, returns x, y, and yn
        :param x: State vector
        """
        # v = kwargs.get("v")
        # return np.array([x[0] * np.exp(-x[1] / v)])
        return np.array([x[0], x[1]])
        # return np.array([x[0], x[1], x[5]])

    def fx(self, x, dt, **kwargs):
        """
        State transition function
        :param x: State vector
        :param dt: Time step
        """
        v = kwargs.get("v")
        d = x[6]
        assert d > 0, "d must be positive, but is {}".format(d)
        defl = x[0:2] - x[4:6]
        v_net = v * np.array([1, 0]) + x[2:4]
        kx = 10 if defl[0] > 0 else 3
        ky = 5 if defl[1] > 0 else 1.5
        defl_force = -kx * defl[0] - ky * defl[1]
        cut_force = cut_force_model(v, d)
        F = defl_force - (cut_force * np.array([1, 0])) - (1 * x[2:4])
        x[0] += x[2] * dt
        x[1] += x[3] * dt
        x[2:4] += F * dt
        return x

    def update_bounds(self) -> None:
        return
        if self.kf.low_bound[4] < self.kf.x[0]:
            self.kf.low_bound[4] = self.kf.x[0]


class ThermalAdaptation(UKFIdentification):
    def __init__(self, w0: np.array, labels, material: MaterialProperties):
        """
        :param w0: Initial parameter estimate  [w, q, Cp, rho, k] in mm, mm/s, W, J/kgK
        """
        super().__init__(w0, dim_z=1, labels=labels, lower_bounds=np.array([0, 0, 0, 0, 0]))
        self.kf.P = np.diag([1, 5, 25 ** 2, 25 ** 2, 1])
        self.kf.R = 0.5 ** 2
        self.kf.Q = np.diag([1, 9, 25 ** 2, 25 ** 2, 1])

    @property
    def x(self) -> np.array:
        """
        State estimate [w, q, Cp] in mm, mm/s W, J/kgK
        """
        # return np.array([self.w_mm, self.q, self.Cp])
        return self.kf.x

    @property
    def q(self):
        """Heat generation rate [W]"""
        return self.kf.x[1]

    @property
    def Cp(self):
        """Specific heat capacity [J/kgK]"""
        return self.kf.x[2]

    @property
    def rho(self):
        """Density [kg/mm^3]"""
        return self.kf.x[3] * 1e-9

    @property
    def lambda_therm(self):
        return self.kf.x[4] * 1e-3

    @property
    def w_mm(self):
        """
        Thermal width [mm]
        """
        return self.kf.x[0]

    @property
    def w_std(self):
        """
        Standard deviation of the thermal width [mm].
        """
        return np.sqrt(self.kf.P[0,0])

    def hx(self, x, **kwargs):
        """
        Measurement function returning the thermal width
        :param x: State vector
        """
        return np.array([x[0]])

    def fx(self, x, dt, **kwargs):
        """
        State transition function
        :param x: State vector
        :param dt: Time step
        """
        v = kwargs.get("v")
        dT = kwargs.get("dT")
        Cp = x[2]
        q = x[1]
        assert q >= 0, "q must be positive"
        assert Cp >= 0, "Cp must be positive"
        material = MaterialProperties(_rho=x[3] * 1e-9, _Cp=Cp, _k=x[4] * 1e-3)
        y = isotherm_width_model(material, v, q, dT)
        if np.isinf(y) or np.isnan(y):
            return x
        x[0] = y
        return x

    def update_bounds(self):
        pass