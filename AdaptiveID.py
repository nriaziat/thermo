
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
