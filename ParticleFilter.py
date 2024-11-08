import numpy as np
import scipy
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
from filterpy.common import Q_discrete_white_noise

class ParticleFilter:
    MAX_PARTICLES = 500
    dt = 1/24

    def __init__(self, xlim, ylim):
        self.min_state = np.array([xlim[0], ylim[0], -np.inf, -np.inf, xlim[0], ylim[0], 0, 0])
        self.max_state = np.array([xlim[1], ylim[1], np.inf, np.inf, xlim[1], ylim[1], np.inf, np.inf])
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 8))
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.points = MerweScaledSigmaPoints(n=len(self.min_state), alpha=1e-3, beta=2, kappa=0)
        self.Q = np.block([[Q_discrete_white_noise(dim=2, dt=1/24, var=2, block_size=2), np.zeros((4, 4))],
                                   [np.zeros((4, 4)), np.diag([0.01**2, 0.01**2, 0.1**2, 0.1**2])]])

        self.R = np.diag([0.5 ** 2, 0.5 ** 2])

    def initialize_particles(self, x, y):
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.particles[:, 0] = x + np.random.normal(loc=0.0, scale=0.5, size=self.MAX_PARTICLES)
        self.particles[:, 1] = y + np.random.normal(loc=0.0, scale=0.5, size=self.MAX_PARTICLES)
        self.particles[:, 2] = np.random.normal(loc=0, scale=0.1, size=self.MAX_PARTICLES)
        self.particles[:, 3] = np.random.normal(loc=0, scale=0.1, size=self.MAX_PARTICLES)
        self.particles[:, 4] = x + np.random.normal(loc=0.0, scale=1, size=self.MAX_PARTICLES)
        self.particles[:, 5] = y + np.random.normal(loc=0.0, scale=1, size=self.MAX_PARTICLES)
        self.particles[:, 6] = np.random.normal(loc=1, scale=5, size=self.MAX_PARTICLES)
        self.particles[:, 7] = np.random.normal(loc=1, scale=5, size=self.MAX_PARTICLES)

    def motion_model(self, proposal_dist, action):
        proposal_dist[:, 0:2] += proposal_dist[:, 2:4] * self.dt
        proposal_dist[:, 2] += self.dt * (-proposal_dist[:, 6] * (proposal_dist[:, 0] - proposal_dist[:, 4]) - proposal_dist[:, 7] * action[0])
        proposal_dist[:, 2] += self.dt * (-proposal_dist[:, 6] * (proposal_dist[:, 1] - proposal_dist[:, 5]) - proposal_dist[:, 7] * action[1])
        proposal_dist[:, 2:4] += np.random.normal(loc=0.0, scale=0.1*self.dt, size=(len(proposal_dist), 2))

        proposal_dist[:, 4:6] += np.random.normal(loc=0.0, scale=0.1*self.dt, size=(len(proposal_dist), 2))
        proposal_dist[:, 6:] += np.random.normal(loc=0.0, scale=1*self.dt, size=(len(proposal_dist), 2))

    def sensor_model(self, proposal_dist: np.array, obs, weights):
        for i in range(self.MAX_PARTICLES):
            if (self.min_state > proposal_dist[i, :]).any() or (proposal_dist[i, :] > self.max_state).any():
                weights[i] = 0
            elif proposal_dist[i, 0] > proposal_dist[i, 4] or obs[0] > proposal_dist[i, 4]:
                weights[i] = 0
            else:
                dist = np.linalg.norm(proposal_dist[i, 0:2] - obs)
                defl = np.linalg.norm(proposal_dist[i, 0:2] - proposal_dist[i, 4:6])
                y_defl = abs(proposal_dist[i, 1] - proposal_dist[i, 4])
                if defl > 5:
                    weights[i] = 0
                else:
                    weights[i] = scipy.stats.norm.pdf(dist, loc=0, scale=1) * scipy.stats.norm.pdf(y_defl, loc=0, scale=1)


    def MCL(self, a, o):
        proposal_indices = np.random.choice(self.particle_indices, self.MAX_PARTICLES, p=self.weights)
        proposal_distribution = self.particles[proposal_indices, :]
        self.motion_model(proposal_distribution, a)
        self.sensor_model(proposal_distribution, o, self.weights)
        self.weights /= np.sum(self.weights)
        self.particles = proposal_distribution

    def expected_pose(self):
        return np.dot(self.particles.T, self.weights)

    def get_n_best_particle_indices(self, n):
        indices = sorted(self.particle_indices, key=lambda i: self.weights[i])
        return indices[:n]

def cross_variance(x, z, sigmas_f, sigmas_h, Wc):
    """
    Compute cross variance of the state `x` and measurement `z`.
    """

    Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
    N = sigmas_f.shape[0]
    for i in range(N):
        dx = np.subtract(sigmas_f[i], x)
        dz = np.subtract(sigmas_h[i], z)
        Pxz += Wc[i] * np.outer(dx, dz)
    return Pxz