from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, ExtendedKalmanFilter, unscented_transform
import numpy as np
import scipy
from abc import ABC, abstractmethod
from copy import deepcopy
from numpy import eye, dot, isscalar

class TruncatedUnscentedKalmanFilter(UnscentedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 low_bound: np.ndarray=None, high_bound: np.ndarray=None):
        super().__init__(dim_x, dim_z, dt, hx, fx, points)
        if low_bound is None:
            self.low_bound = np.array([-np.inf]*dim_x)
        else:
            self.low_bound = low_bound
        if high_bound is None:
            self.high_bound = np.array([np.inf]*dim_x)
        else:
            self.high_bound = high_bound

        assert len(self.low_bound) == dim_x, "Lower bound must be the same length as the state vector"
        assert len(self.high_bound) == dim_x, "Upper bound must be the same length as the state vector"
        assert (self.low_bound < self.high_bound).all(), "Lower bound must be less than upper bound"
        self.eta = 0  # sensitivity parameter
        self.alpha = 1  # forgetting factor
        self.x, self.P = self.PDF_truncation(self.x, self.P)

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

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual


        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)


        self.eta = self.y.T @ self.SI @ self.y

        # # TODO: Determine if this is necessary
        # if self.eta > 3:
        #     self.alpha = 1 if (a:=np.dot(self.y, self.y)) <= (b:=np.trace(self.S)) else b/a
        # else:
        #     self.alpha = 1

        self.alpha = 1


        self.P = (self.P - dot(self.K, dot(self.S, self.K.T))) / self.alpha

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
        epsilon = 1e-9
        Pt = 1/2 * (P + P.T) + epsilon * np.eye(P.shape[0])  # Ensure P is symmetric and positive definite
        n = self.points_fn.n
        for i in range(n):
            Pt[np.isclose(Pt, 0)] = 0
            D, S = scipy.linalg.schur(Pt)
            sqrtD = np.diag(np.sqrt(np.diag(D)))
            # invSqrtD = np.diag(np.reciprocal(np.diag(sqrtD)))
            # assert not (np.isnan(sqrtD)).any(), f"Negative value in D, D ={D}"
            # assert np.allclose(D, np.diag(np.diag(D))), f"Schur decomposition failed. D is not diagonal. D={D}"
            # assert np.allclose(S @ S.T, np.eye(len(S))), f"Schur decomposition failed. S is not orthogonal. S={S}"
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
            aki = 1 / np.sqrt(Pt[i, i]) * (self.low_bound[i] - xt[i])
            bki = 1 / np.sqrt(Pt[i, i]) * (self.high_bound[i] - xt[i])
            # zki = theta @ invSqrtD @ S.T @ (self.x - xt)
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
            xt = (S @ sqrtD @ theta.T @ zki_bar).reshape(n) + xt
            Pt = S @ sqrtD @ theta.T @ Pzz @ theta @ sqrtD @ S.T
            Pt = 1 / 2 * (Pt + Pt.T) + epsilon * np.eye(Pt.shape[0])  # Ensure P is symmetric and positive definite
        return xt, Pt


class UKFIdentification(ABC):
    def __init__(self, w0: np.array, dim_z: int, labels: list[str], lower_bounds: np.ndarray = None,
                 upper_bounds: np.ndarray = None):
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

        self.kf = TruncatedUnscentedKalmanFilter(dim_x=n,
                                                 dim_z=dim_z,
                                                 dt=1/24,
                                                 hx=self.hx,
                                                 fx=self.fx,
                                                 points=self.points,
                                                 low_bound=lower_bounds,
                                                 high_bound=upper_bounds)

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