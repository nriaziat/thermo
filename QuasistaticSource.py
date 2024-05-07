import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import kn


class QuasiStaticSource:
    def __init__(self, frame_shape: tuple[int, int] = (384, 288),
                 v_min: float = 0.01,
                 v_max: float = 25,
                 Kp: float = 0.02,
                 Ki: float = 0.005,
                 Kd: float = 0.01,
                 v0: float = 0.1,
                 des_width: float = 50,
                 t_death: float = 50):
        """

        :param frame_shape: Size of thermal frame
        :param v_min: minimum tool speed
        :param v_max: maximum tool speed
        :param Kp: Proportional gain
        :param Ki: Integral gain
        :param Kd: Derivative gain
        :param v0: Starting tool speed
        :param des_width: Desired Isotherm width in pixels
        :param t_death: Isotherm temperature
        """
        self._Q = 500  # J / mm^2 ??
        self._k = 0.49e-3  # W/(mm*K)
        self.To = 23  # C
        self._rho = 1090e-9  # kg/m^3
        self._cp = 3421  # J/(kg*K)
        self._alpha = self._k / (self._rho * self._cp)  # mm^2/s
        self._a = self._Q / (2 * np.pi * self._k)
        self._b = 1 / (2 * self._alpha)

        self._frame_shape = frame_shape

        self._v_min = v_min
        self._v_max = v_max

        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd

        self.v0 = v0
        self.des_width = des_width
        self.t_death = t_death

        self.v = self.v0
        self._error = 0
        self._last_error = 0
        self._error_sum = 0

    def isotherm_width(self, t) -> int:
        """
        Calculate the width of the isotherm
        :param t: Temperature field
        :return: Width of the isotherm
        """
        return np.max(np.sum(t > self.t_death, axis=0))

    def temp_field_prediction(self, xi, y, u, alph, beta) -> np.ndarray:
        """
        Predict the temperature field using the given parameters
        :param xi: x location relative to the tool
        :param y: y location relative to the tool
        :param u: tool speed
        :param alph: lumped parameter 1
        :param beta: lumped parameter 2
        :return: Predicted temperature field
        """
        r = np.sqrt(xi ** 2 + y ** 2)
        ans = self.To + alph * u * np.exp(-beta * xi * u, dtype=np.longfloat) * kn(0, beta * r * u)
        np.nan_to_num(ans, copy=False, nan=self.To, posinf=np.min(ans), neginf=np.max(ans))
        return ans

    def predict_v(self, a_hat, b_hat, v0, v_min, v_max, des_width) -> float:
        """
        Compute the optimal tool speed using numerical optimization
        :param a_hat: lumped parameter 1 estimate
        :param b_hat: lumped parameter 2 estimate
        :param v0: speed guess
        :param v_min: speed lower bound
        :param v_max: speed upper bound
        :param des_width: desired isotherm width in pixels
        :return: optimal tool speed
        """
        grid = np.meshgrid(np.linspace(-50, 0, self._frame_shape[0]), np.linspace(-50, 50, self._frame_shape[1]))
        res = minimize(lambda x: (self.isotherm_width(self.temp_field_prediction(grid[0], grid[1],
                                                                                 u=x, alph=a_hat,
                                                                                 beta=b_hat)) - des_width) ** 2, x0=v0,
                       bounds=((v_min, v_max),),
                       method='Powell')
        if not res.success:
            raise Exception("Optimization failed")
        return res.x[0]

    def estimate_params(self, v, xdata, ydata, a_hat, b_hat) -> tuple[float, float, float]:
        """
        Estimate the parameters of the model using curve fitting
        :param v: tool speed
        :param xdata: (xi, y) data
        :param ydata: Temperature data
        :param a_hat: current estimate of a
        :param b_hat: current estimate of b
        :return: a_hat, b_hat, cp_hat
        """
        if not np.isinf(ydata).any():
            try:
                popt, pvoc = curve_fit(
                    lambda x, ap, bp, cp: self.temp_field_prediction(x[0], x[1], u=v, a=ap, b=bp) + np.random.normal(0,
                                                                                                                     cp),
                    xdata,
                    ydata,
                    p0=[a_hat, b_hat, 1],
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                    method='trf',
                    nan_policy='omit')
            except ValueError:
                return a_hat, b_hat, 1
            return popt[0], popt[1], popt[2]
        else:
            return a_hat, b_hat, 1

    def update_velocity(self, frame: np.ndarray) -> float:
        """
        Update the tool speed based on the current frame
        :param frame: Temperature field from the camera
        :return: new tool speed
        """
        self._last_error = self._error
        self._error = self.isotherm_width(frame) - self.des_width
        self._error_sum += self._error
        self.v += self._Kp * self._error + self._Ki * self._error + self._Kd * (self._error - self._last_error)
        if self.v < self._v_min:
            self.v = self._v_min
            if self._error > 0:
                self._error_sum += self._error
        elif self.v > self._v_max:
            self.v = self._v_max
            if self._error < 0:
                self._error_sum += self._error
        else:
            self._error_sum += self._error
        if np.sign(self._error) != np.sign(self._last_error):
            self._error_sum = 0

        return self.v
