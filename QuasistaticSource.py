import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.special import kn
import cv2 as cv
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

Q = 500  # J / mm^2 ??
k = 0.49e-3  # W/(mm*K)
To = 23  # C
rho = 1090e-9  # kg/m^3
cp = 3421  # J/(kg*K)
alpha = k / (rho * cp)  # mm^2/s
a = Q / (2 * np.pi * k)
b = 1 / (2 * alpha)

x_len = 50  # mm
x_res = 384  # pixels
y_res = 288
cam_mm_per_px = x_len / x_res  # mm per pixel
y_len = y_res * cam_mm_per_px

ys = np.linspace(-y_len / 2, y_len / 2, y_res)
xs = np.linspace(-x_len / 2, x_len / 2, x_res)
grid = np.meshgrid(xs, ys)


def isotherm_width(t_frame: np.array, t_death: float) -> int:
    """
    Calculate the width of the isotherm
    :param t_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :return: Width of the isotherm [px]
    """
    return np.max(np.sum(t_frame > t_death, axis=0))


def cv_isotherm_width(t_frame: np.ndarray, t_death: float) -> (float, tuple | None):
    """
    Calculate the width of the isotherm using ellipse fitting
    :param t_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :return: Width of the isotherm [px] and the ellipse
    """
    binary_frame = (t_frame > t_death).astype(np.uint8)
    binary_frame = cv.medianBlur(binary_frame, 5)
    contours = cv.findContours(binary_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = contours[0]

    # cv.imshow("Contours", cnt_frame)
    # cv.waitKey(1)

    # contours = sorted(contours, key=cv.contourArea, reverse=True)

    # ellipse = cv.fitEllipse(contours[0])

    list_of_pts = []
    for ctr in contours:
        if cv.contourArea(ctr) > 50:
            list_of_pts += [pt[0] for pt in ctr]
    ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
    hull = cv.convexHull(ctr)
    if hull is None or len(hull) < 5:
        return 0, None
    ellipse = cv.fitEllipse(hull)
    w = ellipse[1][0]
    return w, ellipse


def temp_field_prediction(xi, y, u, alph, beta, To) -> np.ndarray:
    """
    Predict the temperature field using the given parameters
    :param xi: x location relative to the tool [mm]
    :param y: y location relative to the tool [mm]
    :param u: tool speed [mm/s]
    :param alph: lumped parameter 1
    :param beta: lumped parameter 2
    :param To: ambient temperature [C]
    :return: Predicted temperature field [C]
    """
    if u < 0:
        u = -u
        xi = -xi
    r = np.sqrt(xi ** 2 + y ** 2)
    ans = To + alph * u * np.exp(-beta * xi * u, dtype=np.longfloat) * kn(0, beta * r * u)
    np.nan_to_num(ans, copy=False, nan=To, posinf=np.min(ans), neginf=np.max(ans))
    return ans


def predict_v(a_hat, b_hat, v0, v_min, v_max, des_width, To, t_death, grid) -> float:
    """
        Compute the optimal tool speed using numerical optimization
        :param a_hat: lumped parameter 1 estimate
        :param b_hat: lumped parameter 2 estimate
        :param v0: speed guess [mm/s]
        :param v_min: speed lower bound [mm/s]
        :param v_max: speed upper bound [mm/s]
        :param des_width: desired isotherm width [px]
        :param To: ambient temperature [C]
        :param t_death: isotherm temperature [C]
        :param grid: meshgrid of x, y locations [mm]
        :return: optimal tool speed [mm/s]
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


def estimate_params(v, xdata, ydata, a_hat, b_hat) -> tuple[float, float]:
    """
    Estimate the parameters of the model using curve fitting
    :param v: tool speed [mm/s]
    :param xdata: (xi, y) data [mm]
    :param ydata: Temperature data [C]
    :param a_hat: current estimate of a
    :param b_hat: current estimate of b
    :return: a_hat, b_hat, cp_hat
    """
    if not np.isinf(ydata).any():
        try:
            popt, pvoc = curve_fit(
                lambda x, ap, bp, cp: temp_field_prediction(x[0], x[1], u=v, alph=ap, beta=bp,
                                                            To=To) + np.random.normal(0,
                                                                                      cp),
                xdata,
                ydata,
                p0=[a_hat, b_hat, 1],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                method='trf',
                nan_policy='omit')
        except ValueError:
            return a_hat, b_hat
        return popt[0], popt[1]
    else:
        return a_hat, b_hat


def find_tooltip(therm_frame: np.ndarray, t_death) -> tuple | None:
    """
    Find the location of the tooltip
    :param therm_frame: Temperature field [C]
    :param t_death: Isotherm temperature [C]
    :return: x, y location of the tooltip [px]
    """
    if (therm_frame > t_death).any():
        return np.unravel_index(np.argmax(therm_frame), therm_frame.shape)
    else:
        return None


class OnlineVelocityOptimizer:
    Kp = 0.1
    Ki = 0.001
    Kd = 0.03
    Kp2 = 0.5

    def __init__(self,
                 des_width: float = 0.5,
                 v_min: float = 1,
                 v_max: float = 25,
                 v0: float = 1,
                 t_death: float = 50,
                 use_cv: bool = True):
        """
        :param des_width: Desired Isotherm width [px]
        :param v_min: minimum tool speed [mm/s]
        :param v_max: maximum tool speed [mm/s]
        :param v0: Starting tool speed [mm/s]
        :param t_death: Isotherm temperature [C]
        :param use_cv: Use OpenCV for isotherm width calculation, otherwise use heuristic measure.
        """

        self._v_min = v_min
        self._v_max = v_max

        self.des_width = des_width
        self.t_death = t_death

        self.v = v0
        self._error = 0
        self._last_error = 0
        self._error_sum = 0
        self.width = 0

        self._width_kf = KalmanFilter(dim_x=2, dim_z=1)
        self._width_kf.x = np.array([self.width, 0])
        self._width_kf.F = np.array([[1, 1], [0, 1]])
        self._width_kf.H = np.array([[1, 0]])
        self._width_kf.P *= 10
        self._width_kf.R = 1
        self._width_kf.Q = Q_discrete_white_noise(dim=2, dt=0.05, var=50)

        self._pos_kf_init = False
        self._pos_init = None
        self._deflection = 0
        self._pos_kf = None

        self._tool_stiffness = 0.8705 # N/mm

        self._cv = use_cv

    def init_pos_kf(self, pos: tuple):
        self._pos_kf = KalmanFilter(dim_x=4, dim_z=2)
        self._pos_kf.x = np.array([pos[0], 0, pos[1], 0])
        self._pos_kf.F = np.array([[1, 0.25, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0.25],
                                   [0, 0, 0, 1]])
        self._pos_kf.H = np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0]])
        self._pos_kf.P *= 1
        self._pos_kf.R = np.array([[15, 0],
                                   [0, 15]])
        self._pos_kf.Q = Q_discrete_white_noise(dim=4, dt=0.05, var=5)

    def update_tool_deflection(self, frame: np.ndarray[float]) -> tuple[float, float]:
        """
        Update the tool deflection based on the current frame
        :param frame: Temperature field from the camera.
        :return: deflection [px], deflection rate [px/s]
        """
        tool_tip = find_tooltip(frame, self.t_death)
        if tool_tip is None:
            self._pos_kf_init = False
            return 0, 0
        if not self._pos_kf_init:
            self.init_pos_kf(tool_tip)
            self._pos_kf_init = True
            self._pos_init = tool_tip
        self._pos_kf.predict()
        self._pos_kf.update(tool_tip)
        tool_tip = self._pos_kf.x[0], self._pos_kf.x[2]
        deflection = np.array(tool_tip) - np.array(self._pos_init)
        ddeflection = np.array([self._pos_kf.x[1], self._pos_kf.x[3]])
        return np.linalg.norm(deflection), np.linalg.norm(ddeflection) * np.sign(np.dot(deflection, ddeflection))

    def update_velocity(self, v: float, frame: np.ndarray[float]) -> any:
        """
        Update the tool speed based on the current frame
        :param v: Current tool speed
        :param frame: Temperature field from the camera. If None, the field will be predicted using the current model
        :return: new tool speed, ellipse of the isotherm if using CV
        """

        self._last_error = self._error

        self._deflection, ddeflection = self.update_tool_deflection(frame)
        f = self._tool_stiffness * self._deflection
        df = self._tool_stiffness * ddeflection

        ellipse = None
        if not self._cv:
            z = isotherm_width(frame, self.t_death)
        else:
            z, ellipse = cv_isotherm_width(frame, self.t_death)

        self._width_kf.predict()
        self._width_kf.update(z)
        self.width = self._width_kf.x[0]
        self._error = (self.width - self.des_width) - 1.75 * f

        dwidth = self._width_kf.x[1]

        # self._error = self.width - self.des_width
        self._error_sum += self._error
        self.v = v + self.Kp * self._error + self.Ki * self._error - self.Kd * (dwidth - 1.75 * df)

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

        return self.v, ellipse

    def get_loggable_data(self) -> dict:
        """
        Get the data that can be logged
        :return: dictionary of data
        """
        return {
            "v": self.v,
            "error": self._error,
            "error_sum": self._error_sum,
            "width": self.width,
            "force": self._tool_stiffness * self._deflection,
        }


if __name__ == "__main__":
    print("Running QuasiStaticSource.py in Simulation Mode.")

    import logging
    import matplotlib.pyplot as plt

    logging.basicConfig(filename=f"log.log",  # "log.log
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    logger = logging.getLogger("qs_test")

    tool_wd = 5 / cam_mm_per_px  # px
    # Example usage
    optimizer = OnlineVelocityOptimizer(des_width=20 / cam_mm_per_px)
    for i in range(100):
        frame = temp_field_prediction(grid[0], grid[1], u=-optimizer.v, alph=a, beta=b, To=To)
        # add occlusion
        y, x = frame.shape
        y_min = int(y // 2 - tool_wd / 2)
        y_max = int(y // 2 + tool_wd / 2)
        frame[y_min:y_max, x // 2:] = To
        # add noise
        frame += np.random.normal(0, 5, frame.shape)
        plt.imshow(frame, cmap='hot', interpolation='nearest')
        plt.pause(0.001)
        v = optimizer.update_velocity(frame)
        logger.debug(optimizer.get_loggable_data())
    plt.imshow(frame, cmap='hot', interpolation='nearest')
    plt.show()
    print(
        f"V = {v:.2f} mm/s, Error = {abs(optimizer._error):.2f} mm, Measured width = {cv_isotherm_width(frame, optimizer.t_death) * cam_mm_per_px:.2f} mm")
