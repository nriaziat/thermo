from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, ExtendedKalmanFilter, unscented_transform
from SquareRootUnscentedKalmanFilter import SquareRootUnscentedKalmanFilterParameterEstimation, SquareRootUnscentedKalmanFilter, SquareRootMerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
import numpy as np
from models import ymax, MaterialProperties
import scipy
from abc import ABC, abstractmethod
from copy import deepcopy
from numpy import eye, zeros, dot, isscalar, outer


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

    # def _compute_weights(self, theta:np.ndarray=None):
    #     """ Computes the weights for the scaled unscented Kalman filter.
    #
    #     """
    #     n = self.n
    #     lambda_ = self.alpha**2 * (n +self.kappa) - n
    #
    #     if theta is not None:
    #         a = (2 * lambda_ - 1) / (2 * (n + lambda_) * (np.sum(theta) - (2*n+1) * np.sqrt(n+lambda_)))
    #         b = (1 / (2 * (n + lambda_))) - (2 * lambda_ - 1) / (2 * np.sqrt(n + lambda_) * (np.sum(theta) - (2*n+1) * np.sqrt(n+lambda_)))
    #         self.Wc[0] = b
    #         self.Wm[0] = b
    #         self.Wc[1:] = a * theta + b
    #         self.Wm[1:] = a * theta + b
    #     else:
    #         c = .5 / (n + lambda_)
    #         self.Wc = np.full(2*n + 1, c)
    #         self.Wm = np.full(2*n + 1, c)
    #         # self.Wc[0] = lambda_ / (n + lambda_)
    #         self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
    #         self.Wm[0] = lambda_ / (n + lambda_)

    def sqrt(self, x):
        try:
            result = scipy.linalg.cholesky(x)
        except np.linalg.LinAlgError as e:
            raise ValueError("Cholesky decomposition failed. x={}".format(x)) from e
        return result


    # def sigma_points(self, x, P):
    #     """
    #     Computes the sigma points for an unscented Kalman filter
    #     using the formulation of Wan and van der Merwe. This method
    #     is less efficient than the simple method, but is more stable
    #     for near-linear systems.
    #     """
    #     if self.n != np.size(x):
    #         raise ValueError("expected size(x) {}, but size is {}".format(
    #             self.n, np.size(x)))
    #
    #     n = self.n
    #
    #     if np.isscalar(x):
    #         x = np.asarray([x])
    #
    #     if  np.isscalar(P):
    #         P = np.eye(n)*P
    #     else:
    #         P = np.atleast_2d(P)
    #
    #     lambda_ = self.alpha**2 * (n + self.kappa) - n
    #     # U = self.sqrt((lambda_ + n)*P)
    #     sigmas = np.zeros((2*n+1, n))
    #     sigmas[0] = x
    #     # sigmas[0] = np.minimum(self.high - 1e-6, np.maximum(self.low + 1e-6, x))
    #     sqrtP = scipy.linalg.cholesky(P, lower=True)
    #     S = np.hstack([sqrtP, -sqrtP])
    #     Theta = np.zeros_like(S)
    #     rows, cols = np.where(S > 0)
    #     Theta[S > 0] = np.minimum(np.sqrt(n+lambda_), (self.high[rows] - x[rows]) / S[S > 0])
    #     rows, cols = np.where(S < 0)
    #     Theta[S < 0] = np.minimum(np.sqrt(n+lambda_), (self.low[rows] - x[rows]) / S[S < 0])
    #     Theta[np.isclose(S, 0)] = np.sqrt(n + lambda_)
    #     Theta = Theta.min(axis=0)
    #     for k in range(n):
    #         # pylint: disable=bad-whitespace
    #         sigmas[k+1] = np.subtract(x, -Theta[k] * sqrtP[:, k])
    #         sigmas[n+k+1] = np.subtract(x, Theta[n+k] * sqrtP[:, k])
    #     self._compute_weights(theta=Theta)
    #     assert np.isclose(np.sum(self.Wm), 1)
    #     assert np.isclose(np.sum(self.Wc), 1)


      # for k in range(n):
      #       # pylint: disable=bad-whitespace
      #       sigmas[k+1]   = np.minimum(self.high-1e-6, np.maximum(self.subtract(x, -U[k]), self.low + 1e-6))
      #       sigmas[n+k+1] = np.minimum(self.high-1e-6, np.maximum(self.subtract(x, U[k]), self.low + 1e-6))

        # for k in range(n):
        #     sigmas[k+1] = self.subtract(x, -U[k])
        #     sigmas[n+k+1] = self.subtract(x, U[k])


        # return sigmas

class BoundedSquareRootMerweScaledSigmaPoints(SquareRootMerweScaledSigmaPoints):
    def __init__(self, n, alpha, beta, kappa, low:float | np.ndarray=0, high:float | np.ndarray=np.inf):
        super().__init__(n, alpha, beta, kappa)
        if np.isscalar(low):
            low = np.array([low]*n)
        if np.isscalar(high):
            high = np.array([high]*n)
        assert n == len(low) == len(high), "n must be the same length as low and high"
        self.low = low
        self.high = high
        self._epsilon = 1e-3

    def sigma_points(self, x, sqrt_P):
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

        if  np.isscalar(sqrt_P):
            sqrt_P = np.eye(n)*sqrt_P
        else:
            sqrt_P = np.atleast_2d(sqrt_P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = np.sqrt(lambda_ + n) * sqrt_P
        sigmas = np.zeros((2*n+1, n))
        # sigmas[0] = x
        sigmas[0] = np.minimum(self.high - self._epsilon, np.maximum(self.low + self._epsilon, x))
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = np.minimum(self.high-self._epsilon, np.maximum(self.subtract(x, -U[k]), self.low+self._epsilon))
            sigmas[n+k+1] = np.minimum(self.high-self._epsilon, np.maximum(self.subtract(x, U[k]), self.low+self._epsilon))

        return sigmas

class TruncatedIntervalUnscentedKalmanFilter(UnscentedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, hx, fx, points, Q=None, R=None):
        super().__init__(dim_x, dim_z, dt, hx, fx, points, Q, R)
        self.x, self.P = self.PDF_truncation(self.x, self.P)

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.

        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)
        # and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.points_fn.Wm, self.points_fn.Wc, self.Q,
                            self.x_mean, self.residual_x)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """

        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        # sigmas = self.points_fn.sigma_points(self.x, self.P)
        # sigmas_h = []
        # for i, s in enumerate(sigmas):
        #     sigmas_h.append(hx(s, **hx_args))
        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.points_fn.Wm, self.points_fn.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        self.x, self.P = self.PDF_truncation(self.x, self.P)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def PDF_truncation(self, x, P):
        xt = x
        Pt = P
        n = self.points_fn.n
        for i in range(n):
            Pt[np.isclose(Pt, 0)] = 0
            # assert np.linalg.det(Pt) > 0, f"Pt is not positive definite. Pt={Pt}"
            # assert np.allclose(Pt, Pt.T), f"Pt is not symmetric. Pt={Pt}"
            D, S = scipy.linalg.schur(Pt)
            sqrtD = np.diag(np.sqrt(np.diag(D)))
            invSqrtD = np.diag(np.reciprocal(np.diag(sqrtD)))
            assert not (np.isnan(sqrtD)).any(), f"Negative value in D, D ={D}"
            assert np.allclose(D, np.diag(np.diag(D))), f"Schur decomposition failed. D is not diagonal. D={D}"
            assert np.allclose(S @ S.T, np.eye(len(S))), f"Schur decomposition failed. S is not orthogonal. S={S}"
            theta = np.zeros((n, n))
            for l in range(n):
                if l == 0:
                    theta[l, :] = 1 / (np.sqrt(Pt[i, i])) * (S[i:i + 1, :] @ sqrtD)
                else:
                    el = np.eye(n)[:, l:l + 1]
                    theta[l, :] = (
                            el - np.sum([(el.T @ theta.T[:, q:q + 1]) * theta.T[:, q:q + 1] for q in range(l)],
                                        axis=0)).T
                    if (theta[l, :] == 0).all():
                        el = np.eye(n)[:, 0:1]
                        theta[l, :] = (
                                el - np.sum([(el.T @ theta.T[:, q:q + 1]) * theta.T[:, q:q + 1] for q in range(l)],
                                            axis=0)).T
                    theta[l, :] /= np.linalg.norm(theta[l, :])
                if np.isnan(theta[l, :]).any():
                    raise (ValueError(f"Theta is NaN: {theta}\nPtii={Pt[i, i]}\nSi={S[i, :]}\nsqrtD={sqrtD}"))
            aki = 1 / np.sqrt(Pt[i, i]) * (self.points_fn.low[i] - xt[i])
            bki = 1 / np.sqrt(Pt[i, i]) * (self.points_fn.high[i] - xt[i])
            # zki = theta @ invSqrtD @ S.T @ (self.x - xt)
            # assert aki <= zki[i] <= bki, f"zki={zki[i]}, aki={aki}, bki={bki}"
            alpha_i = (2 ** 0.5 )/ (
                    (np.pi ** 0.5) * scipy.special.erf(bki / (2 ** 0.5)) - scipy.special.erf(aki / (2 ** 0.5)))
            mu_i = alpha_i * (np.exp(-aki ** 2 / 2) - np.exp(-bki ** 2 / 2))
            A = np.exp(-(aki ** 2) / 2) * (aki - 2 * mu_i) if not np.isinf(-aki) else 0
            B = np.exp(-(bki ** 2) / 2) * (bki - 2 * mu_i) if not np.isinf(bki) else 0
            sigma2_i = alpha_i * (A - B) + mu_i ** 2 + 1
            Pzz = np.eye(n)
            Pzz[0, 0] = sigma2_i
            zki_bar = np.zeros((n, 1))
            zki_bar[0] = mu_i
            # print(xt.shape, S.shape, sqrtD.shape, theta.T.shape, zki_bar.shape, self.x.shape)
            xt = (S @ sqrtD @ theta.T @ zki_bar).reshape(n) + xt
            Pt = S @ sqrtD @ theta.T @ Pzz @ theta @ sqrtD @ S.T
        return xt, Pt


class UKFIdentification(ABC):
    def __init__(self, w0: np.array, dim_z: int, labels: list[str], lower_bounds: np.ndarray | float = -np.inf,
                 upper_bounds: np.ndarray | float = np.inf):
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


        self.kf = TruncatedIntervalUnscentedKalmanFilter(dim_x=n,
                                        dim_z=dim_z,
                                        dt=1/24,
                                        hx=self.hx,
                                        fx=self.fx,
                                        points=self.points)


        self.kf.x = w0
        self._data = {label: [(w0[i], w0[i], w0[i])] for i, label in enumerate(self.labels)}
        self._x_hist = []

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
        self.kf.predict(**kwargs)
        # self.ukf.P[6:8, 6:8] /= 0.99999
        # self.ukf.P[9:11, 9:11] /= 0.99999 # forgettting factor
        self.kf.update(z=measurement, **kwargs)
        ci_low, ci_high = self.x - np.sqrt(np.diag(self.kf.P)), self.x + np.sqrt(np.diag(self.kf.P))
        self._data = {label: self._data[label] + [(ci_low[i], self.x[i], ci_high[i])] for i, label in enumerate(self.labels)}
        self._x_hist.append(self.x)
        self.update_bounds()
        return self.kf.x

    @property
    def data(self) -> dict[str, list[tuple]]:
        """
        Dictionary of parameter labels and values (with confidence intervals) [label: [(value, low_ci, high_ci)]]
        """
        return self._data

    @property
    @abstractmethod
    def x(self) -> np.array:
        """
        State estimate
        """
        pass

    @abstractmethod
    def update_bounds(self):
        """
        Update the bounds
        """
        pass

class EKFIdentification(ABC):
    def __init__(self, w0: np.array, dim_z: int, labels: list[str], lower_bounds: np.ndarray = -np.inf,
                 upper_bounds: np.ndarray = np.inf):
        """
        :param w0: Initial parameter estimate
        """
        n = len(w0)
        assert len(labels) == len(w0), "Labels must be the same length as the parameter estimate"
        self._labels = labels

        self.kf = ExtendedKalmanFilter(dim_x=n,
                                       dim_z=dim_z)
        self.kf.x = w0
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
    def HJacobian(self, x, **kwargs) -> np.array:
        """
        Jacobian of the measurement function
        """
        pass

    @abstractmethod
    def fx(self, x, dt) -> np.array:
        """
        State transition function
        """
        pass

    @abstractmethod
    def update_bounds(self):
        """
        Update the bounds
        """
        pass

    def update(self, measurement, **kwargs) -> tuple:
        """
        Update the adaptive parameters
        :param measurement: Measurement
        :param kwargs: Additional keyword arguments to pass to the measurement function
        :return: Tuple of updated state estimate, a, b
        """
        u = kwargs.get("v")
        self.kf.predict_x(u=u)
        self.kf.update(measurement, HJacobian=self.HJacobian, Hx=self.hx)
        ci_low, ci_high = self.kf.x - np.sqrt(np.diag(self.kf.P)), self.kf.x + np.sqrt(np.diag(self.kf.P))
        self._data = {label: self._data[label] + [(ci_low[i], self.kf.x[i], ci_high[i])] for i, label in enumerate(self.labels)}
        self.update_bounds()
        return self.kf.x

    @property
    def data(self) -> dict[str, list[tuple]]:
        """
        Dictionary of parameter labels and values (with confidence intervals) [label: [(value, low_ci, high_ci)]]
        """
        return self._data


class DeflectionAdaptation(UKFIdentification):
    def __init__(self, w0: np.array, labels, px_per_mm=1, frame_size_px=(384, 288)):
        """
        :param w0: Initial parameter estimate [x, y, xd, yd, x_rest, y_rest, c_defl, k_tool] in mm, mm/s, mm, mm, mm, mm, N/mm, N/mm
        """
        super().__init__(w0, dim_z=3, labels=labels, lower_bounds=np.array([0, 0, -np.inf, -np.inf, 0, 0, 0, 0]),
                         upper_bounds=np.array([frame_size_px[0] / px_per_mm, frame_size_px[1] / px_per_mm, np.inf, np.inf, frame_size_px[0] / px_per_mm, frame_size_px[1] / px_per_mm, np.inf, np.inf]))
        assert len(labels) == len(w0), "Labels must be the same length as the parameter estimate"
        self.kf.P = np.diag([0.5**2, 0.5**2, 1, 1, 0.5**2, 0.5**2, 1, 0.25])
        self.kf.R = np.diag([0.5**2, 0.5**2, 0.5**2])
        if isinstance(self.kf, SquareRootUnscentedKalmanFilterParameterEstimation):
            self.kf.sqrt_Q = np.block([[Q_discrete_white_noise(dim=2, dt=1/24, var=1, block_size=2), np.zeros((4, 3))],
                                       [np.zeros((3, 4)), np.diag([1, 1, 1])]])
        else:
            self.kf.Q = np.block([[Q_discrete_white_noise(dim=2, dt=1/24, var=4, block_size=2), np.zeros((4, 4))],
                                   [np.zeros((4, 4)), np.diag([0.01**2, 0.01**2, 1**2, 0.1**2])]])
        self.kf.predict_x = lambda u: self.fx(self.kf.x, 1/24, v=u)

    @property
    def x(self):
        # x = np.array([*self.kf.x[0:6], self.c_defl, self.k_tool])
        x = self.kf.x
        return x

    @property
    def c_defl(self):
        return abs(self.kf.x[6])

    @property
    def k_tool(self):
        return abs(self.kf.x[7])

    @property
    def defl_mm(self):
        return np.linalg.norm(self.kf.x[0:2] - self.kf.x[4:6])

    @property
    def defl_std(self):
        # N = 1000
        # x1 = np.random.multivariate_normal(self.kf.x[0:2], self.kf.P[0:2, 0:2], N)
        # x2 = np.random.multivariate_normal(self.kf.x[4:6], self.kf.P[4:6, 4:6], N)
        # defl = np.linalg.norm(x1 - x2, axis=1)
        # return np.std(defl)
        mu = self.kf.x[0:2] - self.kf.x[4:6]
        P = self.kf.P[0:2, 0:2] + self.kf.P[4:6, 4:6]
        EX2  = np.linalg.norm(mu) ** 2 + np.trace(P)
        VarX2 = 2 * np.trace(P @ P) + 4 * mu.T @ P @ mu
        varX = VarX2 / (4 * EX2)
        return np.sqrt(varX)

    @property
    def defl_hist_mm(self) -> list[float]:
        return [np.linalg.norm(x[0:2] - self.kf.x[4:6]) for x in self._x_hist]

    @property
    def neutral_tip_mm(self):
        return self.kf.x[4:6]

    def hx(self, x, **kwargs):
        """
        Measurement function
        x, y, yn
        """
        # v = kwargs.get("v")
        # return np.array([x[0] * np.exp(-x[1] / v)])
        return np.array([x[0], x[1], x[5]])

    def HJacobian(self, x, **kwargs):
        """
        Jacobian of the measurement function
        """
        return np.array([[1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0]])

    def fx(self, x, dt, **kwargs):
        """
        State transition function
        """
        v = kwargs.get("v")
        v_net = v * np.array([1, 0]) - x[2:4]
        v_hat = v_net / np.linalg.norm(v_net)
        k = x[7]
        c = x[6]
        assert k >= 0, "k must be positive, but is {}".format(k)
        assert c >= 0, "c must be positive, but is {}".format(c)
        defl = x[0:2] - x[4:6]
        F = -k * defl - c * np.exp(-np.linalg.norm(defl) / np.linalg.norm(v_net)) * np.array([1, 0])
        x[0] += x[2] * dt
        x[1] += x[3] * dt
        x[2:4] += F * dt
        return x

    def update_bounds(self):
        return
        if self.points.low[4] < self.kf.x[0] - 0.25:
            self.points.low[4] = self.kf.x[0] - 0.25


class ThermalAdaptation(UKFIdentification):
    def __init__(self, w0: np.array, labels):
        """
        :param w0: Initial parameter estimate  [w, q, Cp] in mm, W, J/kgK
        """
        super().__init__(w0, dim_z=1, labels=labels, lower_bounds=np.array([0, 0, 0]))
        self.kf.P = np.diag([1, 5, 100 ** 2])
        self.kf.R = 0.5 ** 2
        if isinstance(self.kf, SquareRootUnscentedKalmanFilterParameterEstimation):
            self.kf.sqrt_Q = np.diag([4, 5, 100])
        else:
            self.kf.Q = np.diag([1, 9, 50 ** 2])
        # self.Cp = 3421
        self.rho = 1090  # kg/m^3
        # self.k = 0.46e-3

    @property
    def x(self) -> np.array:
        # return np.array([self.w_mm, self.q, self.Cp])
        return self.kf.x

    @property
    def rho_kg_mm3(self):
        return self.rho * 1e-9

    @property
    def k_w_mmK(self):
        return 0.46e-3
        # return self.ukf.x[2] * 1e-3 if self.ukf.x[2] > 0 else 0

    @property
    def q(self):
        return abs(self.kf.x[1])

    @property
    def Cp(self):
        return abs(self.kf.x[2])

    @property
    def alpha(self):
        return self.k_w_mmK / (self.rho_kg_mm3 * self.Cp)

    @property
    def w_mm(self):
        return self.kf.x[0]

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
        k = self.k_w_mmK
        Cp = x[2]
        q = x[1]
        assert q >= 0, "q must be positive"
        assert Cp >= 0, "Cp must be positive"
        # if q <= 0:
        #     y = 0
        # else:
        material = MaterialProperties(_k=k, _rho=self.rho*1e-9, _Cp=Cp)
        y = ymax(v, material, q, dT)
        if np.isinf(y) or np.isnan(y):
            return self.kf.x
        x[0] = y
        return x

    def update_bounds(self):
        pass