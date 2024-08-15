import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import argwhere
from ExperimentManager import LoggingData
from os import PathLike, listdir

thermal_px_per_mm = 5.1337 # px/mm

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
    if hasattr(data, "positions_mm"):
        position = np.array(data.positions_mm)
        position[0] = 0
    elif hasattr(data, "positions"):
        convert_old_data_log(file)
        with open(file, "rb") as f:
            data: LoggingData = pkl.load(f)
        position = np.cumsum(np.array(data.velocities) * 1 / 24)
    min_dist = position[-1]
    if min_dist is None:
        return -1, exp_type
    if min_dist < 200:
        return -1, exp_type
    position = position[:argwhere(position > 200)[0][0]]
    data.velocities = data.velocities[:len(position)]
    data.widths_mm = data.widths_mm[:len(position)]
    data.deflections_mm = data.deflections_mm[:len(position)]
    cost = [cost_fun(w, d) for w, d in zip(data.widths_mm, data.deflections_mm)]
    cost[-1] += cost_fun(data.widths_mm[-1], 0)
    ax[0].plot(position, data.velocities, color)
    ax[1].plot(position, np.array(data.widths_mm) * thermal_px_per_mm, color)
    ax[2].plot(position, data.deflections_mm, color)
    if len(ax) == 4:
        ax[3].plot(position, np.cumsum(cost), color)
    return min_dist, exp_type

def plot_log_dir(file_dir: str | PathLike = None, list_of_files: list[str | PathLike] = None):
    """
    Plot the data log from an experiment.
    :param file_dir: str | PathLike - The paths to the data log files.
    :param list_of_files: list[str | PathLike] - A list of paths to the data log files.
    """
    fig, axs = plt.subplots(3, 1)
    exp_types = []
    if file_dir is not None:
        file_paths = [file_dir + "/" + f for f in listdir(file_dir) if f.endswith(".pkl") and f.startswith("data")]
    else:
        file_paths = list_of_files
    min_dist = np.inf
    data_logs = []
    for fp in file_paths:
        with open(fp, "rb") as f:
            data_logs.append(pkl.load(f))
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
    axs[0].set_ylabel("Velocity (mm/s)")
    axs[1].set_ylabel("Width (mm)")
    axs[2].set_ylabel("Deflection (mm)")
    if len(axs) == 4:
        axs[3].set_ylabel("Cost")
    axs[-1].set_xlabel("Distance (mm)")
    for ax in axs:
        ax.legend(exp_types)
    plt.show()

def convert_old_data_log(old_log_path):
    print("Convert old data log")
    with open(old_log_path, "rb") as f:
        log = pkl.load(f)
    new_log = LoggingData()
    new_log.positions_mm = log.positions
    new_log.deflections_mm = log.deflections
    new_log.widths_mm = log.widths
    if hasattr(log, "costs"):
        new_log.a_hats = log.a_hats
    if hasattr(log, "alpha_hats"):
        new_log.alpha_hats = log.alpha_hats
    new_log.velocities = log.velocities
    new_log.damping_estimates = log.damping_estimates
    with open(old_log_path, 'wb') as f:
        pkl.dump(new_log, f)


plot_log_dir(list_of_files=["../logs/data_adaptive_2024-08-15-11:46.pkl",
                            "../logs/data_7.0mm-s_2024-08-15-11:56.pkl"])



