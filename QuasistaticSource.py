import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import kn

Q = 500  # J / mm^2 ??
k = 0.49e-3  # W/(mm*K)
To = 23  # C
rho = 1090e-9  # kg/m^3
cp = 3421  # J/(kg*K)
alpha = k / (rho * cp)  # mm^2/s
a = Q / (2 * np.pi * k)
b = 1 / (2 * alpha)

grid = np.meshgrid(np.linspace(-50, 0, 384), np.linspace(-50, 50, 288))


def isotherm_width(t_frame: np.array, t_death: float) -> int:
    """
    Calculate the width of the isotherm
    :param t_frame: Temperature field
    :param t_death: Isotherm temperature
    :return: Width of the isotherm
    """
    return np.max(np.sum(t_frame > t_death, axis=0))


def temp_field_prediction(xi, y, u, alph, beta, To) -> np.ndarray:
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
    ans = To + alph * u * np.exp(-beta * xi * u, dtype=np.longfloat) * kn(0, beta * r * u)
    np.nan_to_num(ans, copy=False, nan=To, posinf=np.min(ans), neginf=np.max(ans))
    return ans


def predict_v(a_hat, b_hat, v0, v_min, v_max, des_width, To, t_death, grid) -> float:
    """
        Compute the optimal tool speed using numerical optimization
        :param a_hat: lumped parameter 1 estimate
        :param b_hat: lumped parameter 2 estimate
        :param v0: speed guess
        :param v_min: speed lower bound
        :param v_max: speed upper bound
        :param des_width: desired isotherm width in pixels
        :param To: ambient temperature
        :param t_death: isotherm temperature
        :param grid: meshgrid of x, y locations
        :return: optimal tool speed
        """
    res = minimize(lambda x: (isotherm_width(temp_field_prediction(grid[0], grid[1],
                                                                   u=x, alph=a_hat,
                                                                   beta=b_hat, To=To),
                                             t_death) - des_width) ** 2, x0=v0,
                   bounds=((v_min, v_max),),
                   method='Powell')
    if not res.success:
        raise Exception("Optimization failed")
    return res.x[0]


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

    def update_velocity(self, frame: np.ndarray = None) -> float:
        """
        Update the tool speed based on the current frame
        :param frame: Temperature field from the camera. If None, the field will be predicted using the current model
        :return: new tool speed
        """
        if frame is None:
            frame = temp_field_prediction(grid[0], grid[1], self.v, a, b, To)

        self._last_error = self._error
        self._error = isotherm_width(frame, self.t_death) - self.des_width
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
