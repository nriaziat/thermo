import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from ExperimentManager import LoggingData
from os import PathLike

def plot_data_log(file_paths: list[str | PathLike]):
    """
    Plot the data log from an experiment.
    :param file_paths: list[str | PathLike] - The paths to the data log files.
    """
    fig, ax = plt.subplots(3, 1)
    exp_types = []
    # adaptive_data_log = LoggingData()
    # constant_data_log = LoggingData()
    min_dist = np.inf
    for fp in file_paths:
        exp_type = fp.split("_")[1].lower()
        color = "r" if "adaptive" in exp_type else "b"
        exp_types.append(exp_type)
        with open(fp, "rb") as f:
            data: LoggingData = pkl.load(f)
        mean_width = np.mean(data.widths)
        max_width = np.max(data.widths)
        mean_deflection = np.mean(data.deflections)
        std_width = np.std(data.widths)
        std_deflection = np.std(data.deflections)
        if hasattr(data, "position"):
            position = data.position
        else:
            position = np.cumsum(np.array(data.velocities) * 1/24)
        # print(f"Experiment Speed: {exp_type}")
        # print(f"Mean width: {mean_width:.2f} +- {std_width:.2f} mm")
        # print(f"Mean deflection: {mean_deflection:.2f} +- {std_deflection:.2f} mm")
        # print(f"Max width: {max_width:.2f} mm")

        if position[-1] < min_dist:
            min_dist = position[-1]
        ax[0].set_xlim(0, min_dist)
        ax[1].set_xlim(0, min_dist)
        ax[2].set_xlim(0, min_dist)
        ax[0].plot(position, data.velocities, color)
        ax[0].set_title("Velocities")
        ax[0].set_ylabel("Velocity (mm/s)")
        # ax[0].set_xlabel("Position (mm)")
        ax[1].plot(position, data.widths, color)
        ax[1].set_title("Widths")
        ax[1].set_ylabel("Width (mm)")
        # ax[1].set_xlabel("Position (mm)")
        ax[2].plot(position, data.deflections, color)
        ax[2].set_title("Deflections")
        ax[2].set_ylabel("Deflection (mm)")
        ax[2].set_xlabel("Position (mm)")

    # plt.legend(
    plt.show()


plot_data_log(["../logs/data_Adaptive_2024-07-23-13:29.pkl",
                "../logs/data_adaptive_2024-07-23-13:51.pkl",
                 "../logs/data_3.0mm-s_2024-07-23-13:52.pkl",
               "../logs/data_3.0mm-s_2024-07-23-14:06.pkl",
                "../logs/data_adaptive_2024-07-23-14:05.pkl"])

