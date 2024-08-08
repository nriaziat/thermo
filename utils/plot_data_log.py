import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from ExperimentManager import LoggingData
from os import PathLike, listdir

def plot_data_log(file: str | PathLike, ax: plt.Axes):
    exp_type = file.split("_")[1].lower()
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
    if hasattr(data, "position"):
        position = data.position
    else:
        position = np.cumsum(np.array(data.velocities) * 1 / 24)
    if hasattr(data, "damping_estimates"):
        damping_estimates = data.damping_estimates
    else:

        damping_estimates = []
        damping_estimate = 0.1
        for vel, defl in zip(data.velocities, data.deflections):
            if vel < 2:
                damping_estimates.append(damping_estimate)
                continue
            error = defl - damping_estimate * vel
            damping_estimate += 0.05 * error
            damping_estimates.append(damping_estimate)

    # print(f"Experiment Speed: {exp_type}")
    # print(f"Mean width: {mean_width:.2f} +- {std_width:.2f} mm")
    # print(f"Mean deflection: {mean_deflection:.2f} +- {std_deflection:.2f} mm")
    # print(f"Max width: {max_width:.2f} mm")

    min_dist = position[-1]
    ax[0].plot(position, data.velocities, color)
    ax[0].set_title("Velocities")
    ax[0].set_ylabel("Velocity (mm/s)")
    ax[1].plot(position, data.widths, color)
    ax[1].set_title("Widths")
    ax[1].set_ylabel("Width (mm)")
    ax[2].plot(position, data.deflections, color)
    ax[2].set_title("Deflections")
    ax[2].set_ylabel("Deflection (mm)")
    ax[3].plot(position, damping_estimates, color)
    ax[3].set_title("Damping Estimates")
    ax[3].set_ylabel("Damping Estimate")
    ax[4].plot(position, data.q_hats, color)
    ax[4].set_title("a_hat")
    ax[4].set_ylabel("a_hat")
    ax[5].plot(position, data.b_hats, color)
    ax[5].set_title("b_hat")
    ax[5].set_ylabel("b_hat")
    if hasattr(data, "width_estimates"):
        ax[6].plot(position, data.width_estimates, color)
        ax[6].set_title("Width Estimates")
        ax[6].set_ylabel("Width Estimate (mm)")
        ax[6].set_xlabel("Position (mm)")
    else:
        ax[5].set_xlabel("Position (mm)")
    return min_dist, exp_type

def plot_log_dir(file_dir: str | PathLike):
    """
    Plot the data log from an experiment.
    :param file_dir: str | PathLike - The paths to the data log files.
    """
    fig, axs = plt.subplots(7, 1)
    exp_types = []
    # adaptive_data_log = LoggingData()
    # constant_data_log = LoggingData()
    file_paths = [file_dir + "/" + f for f in listdir(file_dir) if f.endswith(".pkl") and f.startswith("data")]
    min_dist = np.inf
    for fp in file_paths:
        dist, exp_type = plot_data_log(fp, axs)
        if dist == -1:
            continue
        if dist < min_dist:
            min_dist = dist
        exp_types.append(exp_type)

    # plt.legend(
    plt.show()


# plot_log_dir("../logs")
fig, ax = plt.subplots(7, 1)
min_dist1, _ = plot_data_log("../logs/data_adaptive_2024-07-31-18:11.pkl", ax)
# min_dist2, _ = plot_data_log("../logs/data_adaptive_2024-07-31-15:05.pkl", ax)
# for a in ax:
#     a.set_xlim(0, min(min_dist1, min_dist2))
plt.show()


