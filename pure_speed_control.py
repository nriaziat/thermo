from testbed import Testbed
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import os
from sklearn.mixture import GaussianMixture
from scipy.stats import linregress
from scipy.signal import chirp

# testbed = Testbed()
# home = input("Press Enter to home the testbed or 's' to skip: ")
# if home != 's':
#     print("Homing testbed...")
#     testbed.home()
#     print("Testbed homed.")
# # should_chirp = input("Press Enter to chirp or 's' to skip: ")
# # if should_chirp != 's':
# #     t = np.linspace(0, 60, 60000)
# #     y = chirp(t, f0=0.5, t1=60, f1=100)
# #     for pos in y:
# #         testbed.set_speed()
# speed = input("Enter speed: ")
# speed = float(speed)
# testbed.set_speed(speed)
# input("Press Enter to stop the testbed: ")
# testbed.stop()

# speed *= 1e-3
# with open("atracsys_velocity.pkl", "rb") as f:
#     p, v, t = pkl.load(f)
#     v = v[10:]
#     plt.plot(v)
#     plt.ylabel("Velocity")
#     plt.hlines(speed, 0, len(v), colors='r', linestyles='dashed')
#     plt.show()
#     mean_error = 1e3 * np.sqrt(np.mean((v - speed)**2))
#     std_error = 1e3 * np.std(v - speed)
#     print(f"Mean error: {mean_error:.2f} mm/s")
#     print(f"Standard deviation of error: {std_error:.2f} mm/s")
#     os.rename("atracsys_velocity.pkl", f"logs/atracsys_25n_{speed*1e3:.1f}mm_velocity.pkl")

errors = []
for file in os.listdir("logs"):
    if "pkl" in file and "atracsys" in file and "25n" not in file:
        with open("logs/" + file, "rb") as f:
            p, v, t = pkl.load(f)
            v = v[10:]
            v_des = 1e-3*float(file.split("_")[1][:-2])
            # plt.plot(v)
            # plt.ylabel("Velocity")
            # plt.hlines(v_des, 0, len(v), colors='r', linestyles='dashed')
            # plt.show()
            mean_error = np.sqrt(np.mean((v - v_des)**2))
            percent_error = mean_error / v_des * 100
            std_error = np.std(v - v_des)
            percent_std_error = std_error / v_des * 100
            errors.append((v_des, np.mean(v), mean_error, percent_error, std_error, percent_std_error))
            # print(f"Mean error: {mean_error:.2f} mm/s")
            # print(f"Standard deviation of error: {std_error:.2f} mm/s")
v_command = np.array([x[0] for x in errors]) * 1000
v_meas = np.array([x[1] for x in errors]) * 1000
data = np.vstack([v_command, v_meas]).T
data = data[data[:, 0].argsort()]
# model = GaussianMixture(n_components=2, covariance_type='full').fit(data)
# prediction = model.predict(data)
# split_idx = np.max(np.argwhere(prediction)) + 1
split_idx = 11
prediction = [1 if i < split_idx else 0 for i in range(len(data))]
plt.scatter(data[:, 0], data[:, 1], c=prediction)
# plt.scatter([x[0] for x in errors], [x[2] for x in errors], label="Mean error")
# plt.scatter([x[0] for x in errors], [x[4] for x in errors], label="Percent Standard deviation of error")
# plt.legend()
lin_fit = linregress(data[:split_idx, 0], data[:split_idx, 1])
plt.plot(data[:split_idx, 0], lin_fit.intercept + lin_fit.slope*data[:split_idx, 0])
plt.text(2.5, 0, f"R2 = {lin_fit.rvalue:.3f}, y = {lin_fit.slope:0.2f}x + {lin_fit.intercept:0.2f}")
# plt.xticks()
plt.xlabel("Command Velocity [mm/s]")
plt.ylabel("Actual Velocity [mm/s]")
plt.show()
error = np.abs(data[:split_idx, 1] - data[:split_idx, 0])
print(f"Mean Error: {np.mean(error):.2f} mm/s")

with open('logs/atracsys_25n_5.0mm_velocity.pkl', 'rb') as f:
    p, v, t = pkl.load(f)
    v_force = v[10:]
with open('logs/atracsys_5.0mm_velocity.pkl', 'rb') as f:
    p, v, t = pkl.load(f)
    v = v[10:]

plt.plot(v_force[:len(v)]*1e3, label="25N force applied")
plt.plot(v*1e3, label="No Force")
plt.hlines(5,0, len(v), 'r')
plt.ylim([0, 5.5])
plt.legend()
plt.show()