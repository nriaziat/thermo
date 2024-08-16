
def proj(theta: float, y: float)->float:
    if theta >= 0:
        return y
    elif y >= 0:
        return y
    else:
        return 0

class AdaptiveParameters:
    """
    Adaptive Parameters
    Identify \hat{a}, \hat{b} for first order system in the form
    \dot{x} = a x + b u
    """
    am = 1
    def __init__(self, x0, a, b, gamma_a=1e-2, gamma_b=1e-2):
        """
        Initialize the adaptive parameters
        :param x0: Initial state estimate
        :param a: Initial a
        :param b: Initial b
        :param gamma_a: Learning rate for a
        :param gamma_b: Learning rate for b
        """
        self.state_estimate = x0
        self.a = a
        self.b = b
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b

    def update(self, measurement, v):
        error = self.state_estimate - measurement
        self.state_estimate += -self.am * error + self.a * measurement + self.b * v
        self.a += -self.gamma_a * proj(self.a, error * measurement)
        self.b += -self.gamma_b * proj(self.b, error * v)
        return self.state_estimate, self.a, self.b