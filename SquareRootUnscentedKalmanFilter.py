# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import eye, zeros, dot, isscalar, outer
from scipy.linalg import cholesky, solve_triangular
from filterpy.stats import logpdf
from filterpy.common import pretty_str


def forwardsubs(A, B):
    # x_ik = (b_ik - Sum_aij_xjk) / a_ii

    N = np.shape(A)[0]
    X = np.zeros((B.shape[0], B.shape[1]))

    for k in range(B.shape[1]):
        for i in range(N):
            sum_aij_xj = B[i, k]

            for j in range(i):
                sum_aij_xj -= A[i, j] * X[j, k]

            X[i, k] = sum_aij_xj / A[i, i]

    return X


def backsubs(A, B):
    # x_ik = (b_ik - Sum_aij_xjk) / a_ii

    N = np.shape(A)[0]
    X = np.zeros((B.shape[0], B.shape[1]))

    for k in range(B.shape[1]):
        for i in range(N - 1, -1, -1):
            sum_aij_xj = B[i, k]

            for j in range(N - 1, i, -1):
                sum_aij_xj -= A[i, j] * X[j, k]

            X[i, k] = sum_aij_xj / A[i, i]

    return X

def unscented_transform(sigmas, Wm, Wc, noise_cov=None,
                        mean_fn=None, residual_fn=None):
    r"""
    Computes unscented transform of a set of sigma points and weights.
    returns the mean and covariance in a tuple.

    This works in conjunction with the UnscentedKalmanFilter class.


    Parameters
    ----------

    sigmas: ndarray, of size (n, 2n+1)
        2D array of sigma points.

    Wm : ndarray [# sigmas per dimension]
        Weights for the mean.


    Wc : ndarray [# sigmas per dimension]
        Weights for the covariance.

    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.

    mean_fn : callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.

        .. code-block:: Python

            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.

                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x

    residual_fn : callable (x, y), optional

        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

        .. code-block:: Python

            def residual(a, b):
                y = a[0] - b[0]
                y = y % (2 * np.pi)
                if y > np.pi:
                    y -= 2*np.pi
                return y

    Returns
    -------

    x : ndarray [dimension]
        Mean of the sigma points after passing through the transform.

    P : ndarray
        covariance of the sigma points after passing through the transform.


    """

    kmax, n = sigmas.shape

    try:
        if mean_fn is None:
            # new mean is just the sum of the sigmas * weight
            x = np.dot(Wm, sigmas)   # dot = \Sigma^n_1 (W[k]*Xi[k])
        else:
            x = mean_fn(sigmas, Wm)
    except:
        print(sigmas)
        raise

    if residual_fn is None:
        residual_fn = np.subtract

    resid = np.zeros(sigmas.shape)
    for i, s in enumerate(sigmas):
        resid[i] = residual_fn(s, x.reshape(-1))
    x_dev = resid[0]

    C = np.concatenate((resid[1:, :] * np.sqrt(Wc[1]), noise_cov), axis=0)

    _, S = np.linalg.qr(C)
    S = cholupdate(S.T, x_dev.reshape(n, 1), Wc[0])
    return x, S


def cholupdate(L, W, beta):
    r = np.shape(W)[1]
    m = np.shape(L)[0]

    for i in range(r):
        L_out = np.copy(L)
        b = 1.0

        for j in range(m):
            Ljj_pow2 = L[j, j] ** 2
            wji_pow2 = W[j, i] ** 2
            if Ljj_pow2 + (beta / b) * wji_pow2 < 0:
                # raise ValueError(f"Matrix is not positive definite. L[{j}, {j}] = {L[j, j]}, W[{j}, {i}] = {beta / b* W[j, i]}")
                L_out[j, j] = 0
            else:
                L_out[j, j] = np.sqrt(Ljj_pow2 + (beta / b) * wji_pow2)
            upsilon = (Ljj_pow2 * b) + (beta * wji_pow2)

            for k in range(j + 1, m):
                W[k, i] -= (W[j, i] / L[j, j]) * L[k, j]
                L_out[k, j] = ((L_out[j, j] / L[j, j]) * L[k, j]) + (L_out[j, j] * beta * W[j, i] * W[k, i] / upsilon)

            b += beta * (wji_pow2 / Ljj_pow2)

        L = np.copy(L_out)

    return L_out

class SquareRootUnscentedKalmanFilter(object):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    r"""
    Implements the Scaled Unscented Kalman filter (UKF) as defined by
    Simon Julier in [1], using the formulation provided by Wan and Merle
    in [2]. This filter scales the sigma points to avoid strong nonlinearities.


    Parameters
    ----------

    dim_x : int
        Number of state variables for the filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.


    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

        This is for convience, so everything is sized correctly on
        creation. If you are using multiple sensors the size of `z` can
        change based on the sensor. Just provide the appropriate hx function


    dt : float
        Time between steps in seconds.



    hx : function(x)
        Measurement function. Converts state vector x into a measurement
        vector of shape (dim_z).

    fx : function(x,dt)
        function that returns the state x transformed by the
        state transistion function. dt is the time step in seconds.

    points : class
        Class which computes the sigma points and weights for a UKF
        algorithm. You can vary the UKF implementation by changing this
        class. For example, MerweScaledSigmaPoints implements the alpha,
        beta, kappa parameterization of Van der Merwe, and
        JulierSigmaPoints implements Julier's original kappa
        parameterization. See either of those for the required
        signature of this class if you want to implement your own.

    sqrt_fn : callable(ndarray), default=None (implies scipy.linalg.cholesky)
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing as far as this class is concerned.

    x_mean_fn : callable  (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this if your state variable contains nonlinear
        values such as angles which cannot be summed.

        .. code-block:: Python

            def state_mean(sigmas, Wm):
                x = np.zeros(3)
                sum_sin, sum_cos = 0., 0.

                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += sin(s[2])*Wm[i]
                    sum_cos += cos(s[2])*Wm[i]
                x[2] = atan2(sum_sin, sum_cos)
                return x

    z_mean_fn : callable  (sigma_points, weights), optional
        Same as x_mean_fn, except it is called for sigma points which
        form the measurements after being passed through hx().

    residual_x : callable (x, y), optional
    residual_z : callable (x, y), optional
        Function that computes the residual (difference) between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars. One is for the state variable,
        the other is for the measurement state.

        .. code-block:: Python

            def residual(a, b):
                y = a[0] - b[0]
                if y > np.pi:
                    y -= 2*np.pi
                if y < -np.pi:
                    y = 2*np.pi
                return y

    Attributes
    ----------

    x : numpy.array(dim_x)
        state estimate vector

    P : numpy.array(dim_x, dim_x)
        covariance estimate matrix

    x_prior : numpy.array(dim_x)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    z : ndarray
        Last measurement used in update(). Read only.

    R : numpy.array(dim_z, dim_z)
        measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        process noise matrix

    K : numpy.array
        Kalman gain

    y : numpy.array
        innovation residual

    log_likelihood : scalar
        Log likelihood of last measurement update.

    likelihood : float
        likelihood of last measurment. Read only.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

    mahalanobis : float
        mahalanobis distance of the measurement. Read only.

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead:

        .. code-block:: Python

            kf.inv = np.linalg.pinv


    Examples
    --------

    Simple example of a linear order 1 kinematic filter in 2D. There is no
    need to use a UKF for this example, but it is easy to read.

    >>> def fx(x, dt):
    >>>     # state transition function - predict next state based
    >>>     # on constant velocity model x = vt + x_0
    >>>     F = np.array([[1, dt, 0, 0],
    >>>                   [0, 1, 0, 0],
    >>>                   [0, 0, 1, dt],
    >>>                   [0, 0, 0, 1]], dtype=float)
    >>>     return np.dot(F, x)
    >>>
    >>> def hx(x):
    >>>    # measurement function - convert state into a measurement
    >>>    # where measurements are [x_pos, y_pos]
    >>>    return np.array([x[0], x[2]])
    >>>
    >>> dt = 0.1
    >>> # create sigma points to use in the filter. This is standard for Gaussian processes
    >>> points = MerweScaledSigmaPoints(4, alpha=.1, beta=2., kappa=-1)
    >>>
    >>> kf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=fx, hx=hx, points=points)
    >>> kf.x = np.array([-1., 1., -1., 1]) # initial state
    >>> kf.P *= 0.2 # initial uncertainty
    >>> z_std = 0.1
    >>> kf.R = np.diag([z_std**2, z_std**2]) # 1 standard
    >>> kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.01**2, block_size=2)
    >>>
    >>> zs = [[i+randn()*z_std, i+randn()*z_std] for i in range(50)] # measurements
    >>> for z in zs:
    >>>     kf.predict()
    >>>     kf.update(z)
    >>>     print(kf.x, 'log-likelihood', kf.log_likelihood)

    """

    def __init__(self, dim_x, dim_z, dt, hx, fx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None):
        """
        Create a Kalman filter. You are responsible for setting the
        various state variables to reasonable values; the defaults below will
        not give you a functional filter.

        """

        #pylint: disable=too-many-arguments

        self.x = zeros(dim_x)
        self.P = eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = eye(dim_x)
        self.R = eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        if sqrt_fn is None:
            self.msqrt = cholesky
        else:
            self.msqrt = sqrt_fn

        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x

        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))    # Kalman gain
        self.y = np.zeros((dim_z))           # residual
        self.z = np.array([[None]*dim_z]).T  # measurement
        self.S = np.zeros((dim_z, dim_z))    # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))   # inverse system uncertainty
        self.sqrt_P = None                   # square-root system uncertainty
        self.sqrt_Q = None                   # square-root measurement uncertainty
        self.sqrt_R = None                   # square-root process noise

        self.inv = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

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

        if self.sqrt_P is None:
            self.sqrt_P = self.msqrt(self.P)
        if self.sqrt_Q is None:
            self.sqrt_Q = self.msqrt(self.Q)
        if self.sqrt_R is None:
            self.sqrt_R = self.msqrt(self.R)

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)

        #and pass sigmas through the unscented transform to compute prior
        self.x, self.sqrt_P = UT(self.sigmas_f, self.Wm, self.Wc, self.sqrt_Q,
                            self.x_mean, self.residual_x)
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = self.sqrt_P.T @ self.sqrt_P

    def update(self, z, UT=None, hx=None, **hx_args):
        """
        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

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

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, S_y = UT(self.sigmas_h, self.Wm, self.Wc, self.sqrt_R, self.z_mean, self.residual_z)
        self.S = S_y.T @ S_y
        # self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h).T

        K = forwardsubs(S_y, Pxz)
        self.K = backsubs(S_y.T, K).T
        self.y = self.residual_z(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + self.K @ self.y
        U = self.K @ S_y
        self.sqrt_P = cholupdate(self.sqrt_P, U, -1.0)
        self.P = self.sqrt_P.T @ self.sqrt_P

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        xdim = np.shape(x)[0]
        ydim = np.shape(z)[0]

        n_sigmas = np.shape(sigmas_f)[0]
        Pxy = np.zeros((xdim, ydim))
        for i in range(n_sigmas):
            dx = self.residual_x(sigmas_f[i, :], x.reshape(-1))
            dy = self.residual_z(sigmas_h[i, :], z.reshape(-1))
            Pxy += self.Wc[i] * np.outer(dx, dy)
        return Pxy

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """
        computes the values of sigmas_f. Normally a user would not call
        this, but it is useful if you need to call update more than once
        between calls to predict (to update for multiple simultaneous
        measurements), so the sigmas correctly reflect the updated state
        x, P.
        """

        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.sqrt_P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)

    def batch_filter(self, zs, Rs=None, dts=None, UT=None, saver=None):
        """
        Performs the UKF filter over the list of measurement in `zs`.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self._dt` Missing
            measurements must be represented by 'None'.

        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.

            If Rs is None then self.R is used for all epochs.

            If it is a list of matrices or a 3D array where
            len(Rs) == len(zs), then it is treated as a list of R values, one
            per epoch. This allows you to have varying R per epoch.

        dts : None, scalar or list-like, default=None
            optional value or list of delta time to be passed into predict.

            If dtss is None then self.dt is used for all epochs.

            If it is a list where len(dts) == len(zs), then it is treated as a
            list of dt values, one per epoch. This allows you to have varying
            epoch durations.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

        Returns
        -------

        means: ndarray((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance: ndarray((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        Examples
        --------

        .. code-block:: Python

            # this example demonstrates tracking a measurement where the time
            # between measurement varies, as stored in dts The output is then smoothed
            # with an RTS smoother.

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = ukf.batch_filter(zs, dts=dts)
            (xs, Ps, Ks) = ukf.rts_smoother(mu, cov)

        """
        #pylint: disable=too-many-arguments

        try:
            z = zs[0]
        except TypeError:
            raise TypeError('zs must be list-like')

        if self._dim_z == 1:
            if not(isscalar(z) or (z.ndim == 1 and len(z) == 1)):
                raise TypeError('zs must be a list of scalars or 1D, 1 element arrays')
        else:
            if len(z) != self._dim_z:
                raise TypeError(
                    'each element in zs must be a 1D array of length {}'.format(self._dim_z))

        z_n = np.size(zs, 0)
        if Rs is None:
            Rs = [self.R] * z_n

        if dts is None:
            dts = [self._dt] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((z_n, self._dim_x))
        else:
            means = zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r, dt) in enumerate(zip(zs, Rs, dts)):
            self.predict(dt=dt, UT=UT)
            self.update(z, r, UT=UT)
            means[i, :] = self.x
            covariances[i, :, :] = self.P

            if saver is not None:
                saver.save()

        return (means, covariances)

    def rts_smoother(self, Xs, Ps, Qs=None, dts=None, UT=None):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Qs: list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        dt : optional, float or array-like of float
            If provided, specifies the time step of each step of the filter.
            If float, then the same time step is used for all steps. If
            an array, then each element k contains the time  at step k.
            Units are seconds.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K) = rts_smoother(mu, cov, fk.F, fk.Q)
        """
        #pylint: disable=too-many-locals, too-many-arguments

        if len(Xs) != len(Ps):
            raise ValueError('Xs and Ps must have the same length')

        n, dim_x = Xs.shape

        if dts is None:
            dts = [self._dt] * n
        elif isscalar(dts):
            dts = [dts] * n

        if Qs is None:
            Qs = [self.Q] * n

        if UT is None:
            UT = unscented_transform

        # smoother gain
        Ks = zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = zeros((num_sigmas, dim_x))

        for k in reversed(range(n-1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dts[k])

            xb, Pb = UT(
                sigmas_f, self.Wm, self.Wc, self.Q,
                self.x_mean, self.residual_x)

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], xb)
                z = self.residual_x(sigmas[i], Xs[k])
                Pxb += self.Wc[i] * outer(z, y)

            # compute gain
            K = dot(Pxb, self.inv(Pb))

            # update the smoothed estimates
            xs[k] += dot(K, self.residual_x(xs[k+1], xb))
            ps[k] += dot(K, ps[k+1] - Pb).dot(K.T)
            Ks[k] = K

        return (xs, ps, Ks)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = sqrt(float(dot(dot(self.y.T, self.SI), self.y)))
        return self._mahalanobis

    def __repr__(self):
        return '\n'.join([
            'UnscentedKalmanFilter object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('S', self.S),
            pretty_str('K', self.K),
            pretty_str('y', self.y),
            pretty_str('log-likelihood', self.log_likelihood),
            pretty_str('likelihood', self.likelihood),
            pretty_str('mahalanobis', self.mahalanobis),
            pretty_str('sigmas_f', self.sigmas_f),
            pretty_str('h', self.sigmas_h),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('residual_x', self.residual_x),
            pretty_str('residual_z', self.residual_z),
            pretty_str('msqrt', self.msqrt),
            pretty_str('hx', self.hx),
            pretty_str('fx', self.fx),
            pretty_str('x_mean', self.x_mean),
            pretty_str('z_mean', self.z_mean)
            ])

class SquareRootUnscentedKalmanFilterParameterEstimation(SquareRootUnscentedKalmanFilter):
    def __init__(self, dim_x, dim_z, dt, hx, points,
                 sqrt_fn=None, x_mean_fn=None, z_mean_fn=None,
                 residual_x=None,
                 residual_z=None,
                 gamma=0.9995):
        assert 1 - gamma < 5e-3, "gamma must be close to 1"
        fx = lambda x: x
        super().__init__(dim_x, dim_z, dt, hx, fx, points, sqrt_fn, x_mean_fn, z_mean_fn, residual_x, residual_z)
        self.gamma = gamma

    def predict(self, **fx_args):
        def UT(sigmas, Wm, Wc, noise_cov, mean_fn, residual_fn):
            kmax, n = sigmas.shape
            try:
                if mean_fn is None:
                    # new mean is just the sum of the sigmas * weight
                    x = np.dot(sigmas.T, Wm.reshape(kmax, 1))  # dot = \Sigma^n_1 (W[k]*Xi[k])
                else:
                    x = mean_fn(sigmas, Wm)
            except:
                print(sigmas)
                raise

            if residual_fn is None:
                residual_fn = np.subtract

            resid = np.zeros(sigmas.shape)
            for i, s in enumerate(sigmas):
                resid[i] = residual_fn(s, x)
            x_dev = residual_fn(sigmas[0, :], x)

            C = np.concatenate((resid * np.sqrt(Wc[1]), noise_cov.T), axis=0)

            _, S = np.linalg.qr(C)
            S = cholupdate(S.T, x_dev.reshape(n, 1), Wc[0])
            return x, S

        if self.sqrt_P is None:
            self.sqrt_P = self.msqrt(self.P)
        if self.sqrt_Q is None:
            self.sqrt_Q = self.msqrt(self.Q)
        if self.sqrt_R is None:
            self.sqrt_R = self.msqrt(self.R)
        else:
            self.sqrt_P = self.gamma ** -0.5 * self.sqrt_P

        self.sigmas_f = self.points_fn.sigma_points(self.x, self.sqrt_P)
        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = self.sqrt_P @ self.sqrt_P.T

    def update(self, x, z, UT=None, hx=None, **hx_args):
        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(x, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, S_y = UT(self.sigmas_h, self.Wm, self.Wc, self.sqrt_R, self.z_mean, self.residual_z)
        # self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h).T

        K = forwardsubs(S_y, Pxz)
        self.K = backsubs(S_y.T, K).T
        self.y = self.residual_z(z, zp)


class SquareRootMerweScaledSigmaPoints(object):

    """
    Generates sigma points and weights according to Van der Merwe's
    2004 dissertation[1] for the UnscentedKalmanFilter class.. It
    parametizes the sigma points using alpha, beta, kappa terms, and
    is the version seen in most publications.

    Unless you know better, this should be your default choice.

    Parameters
    ----------

    n : int
        Dimensionality of the state. 2n+1 weights will be generated.

    alpha : float
        Determins the spread of the sigma points around the mean.
        Usually a small positive value (1e-3) according to [3].

    beta : float
        Incorporates prior knowledge of the distribution of the mean. For
        Gaussian x beta=2 is optimal, according to [3].

    kappa : float, default=0.0
        Secondary scaling parameter usually set to 0 according to [4],
        or to 3-n according to [5].

    sqrt_method : function(ndarray), default=scipy.linalg.cholesky
        Defines how we compute the square root of a matrix, which has
        no unique answer. Cholesky is the default choice due to its
        speed. Typically your alternative choice will be
        scipy.linalg.sqrtm. Different choices affect how the sigma points
        are arranged relative to the eigenvectors of the covariance matrix.
        Usually this will not matter to you; if so the default cholesky()
        yields maximal performance. As of van der Merwe's dissertation of
        2004 [6] this was not a well reseached area so I have no advice
        to give you.

        If your method returns a triangular matrix it must be upper
        triangular. Do not use numpy.linalg.cholesky - for historical
        reasons it returns a lower triangular matrix. The SciPy version
        does the right thing.

    subtract : callable (x, y), optional
        Function that computes the difference between x and y.
        You will have to supply this if your state variable cannot support
        subtraction, such as angles (359-1 degreees is 2, not 358). x and y
        are state vectors, not scalars.

    Attributes
    ----------

    Wm : np.array
        weight for each sigma point for the mean

    Wc : np.array
        weight for each sigma point for the covariance

    Examples
    --------

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


    References
    ----------

    .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
           Inference in Dynamic State-Space Models" (Doctoral dissertation)

    """


    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        #pylint: disable=too-many-arguments

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method

        if subtract is None:
            self.subtract = np.subtract
        else:
            self.subtract = subtract

        self._compute_weights()


    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1


    def sigma_points(self, x, sqrt_P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        sqrt_P : scalar, or np.array
           Square Root Covariance of the filter. If scalar, is treated as eye(n)*P.

        Returns
        -------

        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.

            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if np.isscalar(sqrt_P):
            sqrt_P = sqrt_P * np.eye(n)
        else:
            sqrt_P = np.atleast_2d(sqrt_P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = np.sqrt((lambda_ + n))*sqrt_P

        sigmas = np.zeros((2*n+1, n))
        x = x.reshape(-1)
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[:, k])
            sigmas[n+k+1] = self.subtract(x, U[:, k])

        return sigmas


    def _compute_weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.

        """

        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n

        c = .5 / (n + lambda_)
        self.Wc = np.full(2*n + 1, c)
        self.Wm = np.full(2*n + 1, c)
        self.Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        self.Wm[0] = lambda_ / (n + lambda_)



    def __repr__(self):

        return '\n'.join([
            'MerweScaledSigmaPoints object',
            pretty_str('n', self.n),
            pretty_str('alpha', self.alpha),
            pretty_str('beta', self.beta),
            pretty_str('kappa', self.kappa),
            pretty_str('Wm', self.Wm),
            pretty_str('Wc', self.Wc),
            pretty_str('subtract', self.subtract),
            pretty_str('sqrt', self.sqrt)
            ])
