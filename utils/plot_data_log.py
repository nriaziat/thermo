import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import argwhere
from ExperimentManager import LoggingData
from os import PathLike, listdir

plt.ioff()

def cost_fun(w, d):
    return w**2 + 4 * d**2


def plot_data_log(file: str | PathLike, ax: plt.Axes):
    exp_type = file.split("/")[-1].split("_")[1].lower()
    color = "r" if "adaptive" in exp_type else "b"
    with open(file, "rb") as f:
        data: LoggingData = pkl.load(f)
    if len(data.velocities) < 100:
        return -1, exp_type
    # mean_width = np.mean(data.widths)
    # max_width = np.max(data.widths)
    # mean_deflection = np.mean(data.deflections)
    # std_width = np.std(data.widths)
    # std_deflection = np.std(data.deflections)
    if hasattr(data, "positions"):
        position = np.array(data.positions)
        position[0] = 0
    else:
        position = np.cumsum(np.array(data.velocities) * 1 / 24)
    min_dist = position[-1]
    if min_dist is None:
        return -1, exp_type
    if min_dist < 200:
        return -1, exp_type
    position = position[:argwhere(position > 200)[0][0]]
    data.velocities = data.velocities[:len(position)]
    data.widths = data.widths[:len(position)]
    data.deflections = data.deflections[:len(position)]
    cost = [cost_fun(w, d) for w, d in zip(data.widths , data.deflections)]
    cost[-1] += cost_fun(data.widths[-1], 0)
    ax[0].plot(position, data.velocities, color)
    ax[0].set_title("Velocities")
    ax[0].set_ylabel("Velocity (mm/s)")
    ax[1].plot(position, data.widths, color)
    ax[1].set_title("Widths")
    ax[1].set_ylabel("Width (mm)")
    ax[2].plot(position, data.deflections, color)
    ax[2].set_title("Deflections")
    ax[2].set_ylabel("Deflection (mm)")
    ax[3].plot(position, np.cumsum(cost), color)
    ax[3].set_title("Cost")
    ax[3].set_ylabel("Cost")
    return min_dist, exp_type

def plot_log_dir(file_dir: str | PathLike):
    """
    Plot the data log from an experiment.
    :param file_dir: str | PathLike - The paths to the data log files.
    """
    fig, axs = plt.subplots(4, 1)
    exp_types = []
    file_paths = [file_dir + "/" + f for f in listdir(file_dir) if f.endswith(".pkl") and f.startswith("data")]
    min_dist = np.inf
    data_logs = []
    for fp in file_paths:
        with open(fp, "rb") as f:
            data_logs.append(pkl.load(f))
    average_data_logs([data_logs])
    for fp in file_paths:
        dist, exp_type = plot_data_log(fp, axs)
        if dist == -1:
            continue
        if dist < min_dist:
            min_dist = dist
        exp_types.append(exp_type)
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(0, min_dist)
    # plt.legend(
    plt.show()

def average_data_logs(data_logs: list[LoggingData]) -> LoggingData:
    """
    Combine multiple data logs into one by averaging the data.
    :param data_logs: data_logs
    :return: LoggingData - The combined data log.
    """
    # data = LoggingData()
    # for slice in
    # return data


# plot_log_dir("../logs")
fig, ax = plt.subplots(4, 1)
min_dist1, _ = plot_data_log("../logs/data_adaptive_2024-08-14-15:59.pkl", ax)
min_dist2, _ = plot_data_log("../logs/data_5.0mm-s_2024-08-13-14:31.pkl", ax)
for a in ax:
    a.set_xlim(0, min(min_dist1, min_dist2))
    a.legend(["Adaptive", "Constant"])
plt.show()


