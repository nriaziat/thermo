import cv2
import numpy as np
from scipy.special import kn
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
import time
import cv2 as cv

Q = 500  # J / mm^2 ??
k = 0.49e-3  # W/(mm*K)
To = 23  # C
rho = 1090e-9  # kg/m^3
cp = 3421  # J/(kg*K)
alpha = k / (rho * cp)  # mm^2/s
a = Q / (2 * np.pi * k)
b = 1 / (2 * alpha)


def T(xi, y, u, alph, beta):
    r = np.sqrt(xi ** 2 + y ** 2)
    ans = To + alph * u * np.exp(-beta * xi * u, dtype=np.longfloat) * kn(0, beta * r * u)
    np.nan_to_num(ans, copy=False, nan=To, posinf=np.min(ans), neginf=np.max(ans))
    return ans


v0 = 0.1  # mm/s
des_width = 14  # mm
t_death = 50  # C

v_min = 0.01
v_max = 25

x_len = 50
x_res = 384
y_res = 288
scale = x_len / x_res
y_len = y_res * scale

ys = np.linspace(-y_len / 2, y_len / 2, y_res)
xs = np.linspace(-x_len / 2, x_len / 2, x_res)
grid = np.meshgrid(xs, ys)


def cv_field_width(v, t=None, **kwargs):
    if t is None:
        assert v is not None, "Either v or t must be provided"
        t = T(grid[0], grid[1], v, **kwargs)
    else:
        t = cv.GaussianBlur(t, (5, 5), 0)
    contours = cv.findContours((t > t_death).astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = contours[0]
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    try:
        ellipse = cv.fitEllipse(contours[0])
        if cv.contourArea(contours[0]) < 100:
            return 0
        w = ellipse[1][0] / 5
        return w
    except cv2.error:
        return 0


def temp_field_width(v=None, t=None, **kwargs):
    if t is None:
        assert v is not None, "Either v or t must be provided"
        t = T(grid[0], grid[1], v, **kwargs)
    return np.max(np.sum(t > t_death, axis=0)) * scale


a_hat = a * np.random.normal(1, 0.25)
b_hat = b * np.random.normal(1, 0.25)


def predict_v(a_hat, b_hat, v0, v_min, v_max, des_width):
    res = minimize(lambda x: (temp_field_width(x, a=a_hat, b=b_hat) - des_width) ** 2, x0=v0, bounds=((v_min, v_max),),
                   method='Powell')
    if not res.success:
        raise Exception("Optimization failed")
    return res.x[0]


def estimate_params(v, xdata, ydata, a_hat, b_hat):
    if not np.isinf(ydata).any():
        try:
            popt, pvoc = curve_fit(lambda x, ap, bp, cp: T(x[0], x[1], u=v, a=ap, b=bp) + np.random.normal(0, cp),
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


v = v0

error = 0
error_sum = 0
errors = []
vs = []
a_hats = []
b_hats = []
pred_errs = []
dts = []

Kp = 0.02
Ki = 0.005
Kd = 0.01

total_plate = np.ones((y_res, x_res * 1500), dtype=np.float32) * To

xdata = np.array([grid[0].flatten(), grid[1].flatten()]).reshape(2, -1)
x0 = 0
for i in range(1000):

    plate = T(grid[0], grid[1], v, a, b)
    dx = v * 0.1 / x_len * x_res
    x0 += dx
    x0 = int(x0)
    if dx > 0:
        total_plate[:, x0:x0 + x_res] = np.maximum(total_plate[:, x0:x0 + x_res], plate)

    ydata = plate.flatten()

    t0 = time.time()

    width = temp_field_width(t=plate)

    old_error = error
    error = width - des_width
    errors.append(error)
    vs.append(v)

    # a_hat, b_hat, _ = estimate_params(v, xdata, ydata, a_hat, b_hat)
    # a_hats.append(a_hat)
    # b_hats.append(b_hat)
    # v = predict_v(a_hat, b_hat, v, v_min, v_max, des_width)

    v += Kp * error + Ki * error_sum + Kd * (error - old_error)

    if v < v_min:
        v = v_min
        if error > 0:
            error_sum += error
    elif v > v_max:
        v = v_max
        if error < 0:
            error_sum += error
    else:
        error_sum += error

    # print(
    #     f"V = {v:.2f} mm/s, Error = {error:.2f} mm, Measured width = {width:.2f} mm")
    dts.append(time.time() - t0)

    if len(errors) > 1:
        if np.sign(errors[-1]) != np.sign(errors[-2]):
            error_sum = 0

    p1 = plt.imshow(plate, cmap='hot', interpolation='nearest')
    # plt.imshow(total_plate[:, :x0+x_res], cmap='hot', interpolation='nearest')
    # plt.clf()
    # plt.contourf(grid[0], grid[1], plate, levels=[23, 50, 100], cmap='hot')
    plt.pause(0.001)

if __name__ == "__main__":
    total_plate = total_plate[:, :x0 + x_res]
    plt.imshow(total_plate, cmap='hot', interpolation='nearest')
    plt.show()
    plt.clf()
    plt.contourf(total_plate, levels=[23, 50, 100], cmap='hot')
    plt.show()

    print(f"V final = {np.mean(vs[-50:]):.2f} mm/s")
    print(f"Average time per iteration = {np.mean(dts):.2e} s, {1 / np.mean(dts):.2f} Hz")
    width = temp_field_width(t=plate)
    print(f"Final width = {width:.2f} mm, RMSE = {np.sqrt(np.mean(np.array(errors[-50:]) ** 2)):.2f}")

    errors = np.abs(np.array(errors))
    if len(a_hats) > 0:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].set_title("a")
        axs[0, 0].plot(a_hats)
        axs[0, 0].plot(a * np.ones_like(a_hats), '--')
        axs[0, 1].plot(b_hats)
        axs[0, 1].plot(b * np.ones_like(a_hats), '--')
        axs[0, 1].set_title("b")
        axs[1, 0].plot(errors)
        axs[1, 0].plot(np.ones_like(errors) * np.mean(errors[-50]), '--')
        axs[1, 0].text(0, np.mean(errors[-50]), f"Error = {np.mean(errors[-50:]):.2f} mm")
        axs[1, 0].set_title("Errors")
        axs[1, 1].plot(vs)
        axs[1, 1].set_title("Velocities")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Velocity [mm/s]")
        axs[1, 1].plot(np.ones_like(vs) * np.mean(vs[-50]), '--')
        axs[1, 1].text(0, np.mean(vs[-50]), f"V = {np.mean(vs[-50]):.2f} mm/s")

    else:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(errors)
        axs[0].plot(np.ones_like(errors) * np.mean(errors[-50]), '--')
        axs[0].text(0, np.mean(errors[-50]), f"Error = {np.mean(errors[-50:]):.2f} mm")
        axs[0].set_title("Errors")
        axs[0].set_ylabel("Error [mm]")
        axs[0].legend(["Error", "Steady state error"])
        axs[1].plot(vs)
        axs[1].set_title("Velocities")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Velocity [mm/s]")
        axs[1].plot(np.ones_like(vs) * np.mean(vs[-50]), '--')
        axs[1].text(0, np.mean(vs[-50]), f"V = {np.mean(vs[-50]):.2f} mm/s")
        axs[1].legend(["Velocity", "Steady state velocity"])

    plt.show()
