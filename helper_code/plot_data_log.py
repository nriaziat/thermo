import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from utils import LoggingData, find_wavefront_distance, cv_isotherm_width, find_tooltip
from os import PathLike, listdir
from matplotlib import rcParams
from scipy.stats import ttest_ind, ranksums
import seaborn as sns
import pandas as pd
import statannot
sns.set_style("white")

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

rcParams['text.usetex'] = True
rcParams["font.family"] = "Times New Roman"
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'
rcParams['axes.grid'] = False
rcParams['lines.linewidth'] = 2.0
rcParams['axes.labelsize'] = 'large'
rcParams['xtick.labelsize'] = 'large'
rcParams['ytick.labelsize'] = 'large'
rcParams['figure.figsize'] = [16, 9]
rcParams["axes.spines.right"] = False
rcParams["axes.spines.top"] = False

thermal_px_per_mm = 5.1337 # px/mm

plt.ioff()


def plot_data_log(file: str | PathLike, main_ax: list[plt.Axes], param_ax: list[plt.Axes], use_position=False):
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
    elif isinstance(data, pd.DataFrame):
        position = np.cumsum(np.array(data.velocities) * 1 / 24)
    elif hasattr(data, "positions"):
        convert_old_data_log(file)
        with open(file, "rb") as f:
            data: LoggingData = pkl.load(f)
        position = np.cumsum(np.array(data.velocities) * 1 / 24)
    min_dist = position[-1]
    if min_dist is None:
        return -1, exp_type
    data.velocities = data.velocities[:len(position)]
    data.velocities = np.clip(data.velocities, -10, 10)
    data.widths_mm = data.widths_mm[:len(position)]
    print(f"Experiment: {exp_type}")
    print(f"Mean WIDTH: {np.mean(data.widths_mm):.2f} mm")
    print(f"Max WIDTH: {np.max(data.widths_mm):.2f} mm")
    data.deflections_mm = data.deflections_mm[:len(position)]
    print(f"Mean DEFLECTION: {np.mean(data.deflections_mm):.2f} mm")
    print(f"Max DEFLECTION: {np.max(data.deflections_mm):.2f} mm")
    if use_position:
        x = position
    else:
        x = np.arange(len(data.velocities))
    main_ax[0].plot(x, data.velocities, color)
    main_ax[1].plot(x, data.widths_mm, color)
    main_ax[2].plot(x, data.deflections_mm, color)

    param_ax[0].plot(x, data.thermal_diffusivity_estimates, color)
    param_ax[1].plot(x, data.damping_estimates)

    return min_dist, "DECAF" if "adaptive" in exp_type else "Constant Velocity"

def plot_log_dir(file_dir: str | PathLike = None, list_of_files: list[str | PathLike] = None, plot_cost=False):
    """
    Plot the data log from an experiment.
    :param file_dir: str | PathLike - The paths to the data log files.
    :param list_of_files: list[str | PathLike] - A list of paths to the data log files.
    :param plot_cost: bool - Whether to plot the cost.
    """
    if plot_cost:
        fig, axs = plt.subplots(4, 1)
    else:
        fig, axs = plt.subplots(3, 1)
    param_fig, param_axs = plt.subplots(2, 1)

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
        dist, exp_type = plot_data_log(fp, axs, param_axs, use_position=use_position)
        if dist == -1:
            continue
        if dist < min_dist:
            min_dist = dist
        exp_types.append(exp_type)
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
        if use_position:
            ax.set_xlim(0, min_dist)
    for ax in param_axs:
        ax.relim()
        ax.autoscale_view()
        if use_position:
            ax.set_xlim(0, min_dist)

    axs[0].set_ylabel("Velocity (mm/s)")
    axs[1].set_ylabel("Width (mm)")
    axs[2].set_ylabel("Deflection (mm)")
    if len(axs) == 4:
        axs[3].set_ylabel("Cost")
    if use_position:
        axs[-1].set_xlabel("Distance (mm)")
        param_axs[-1].set_xlabel("Distance (mm)")
    else:
        axs[-1].set_xlabel("Time Step")
        param_axs[-1].set_xlabel("Time Step")

    param_axs[0].set_ylabel(r"$\hat{\alpha}$")
    param_axs[1].set_ylabel(r"$\hat{c}$")
    axs[0].legend(exp_types)
    param_axs[0].legend(exp_types)
    fig.suptitle(f"Experiment Data")
    param_fig.suptitle(f"Parameter Estimates")
    fig.savefig(f"../plots/plot_{list_of_files[0].split('/')[-1].split('.')[0]}.svg")
    param_fig.savefig(f"../plots/param_plot_{list_of_files[0].split('/')[-1].split('.')[0]}.svg")
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

def summarize_data_logs(list_of_const_vel: list[str | PathLike] =None, list_of_adaptive: list[str | PathLike] =None):
    """
    Plot the average width and deflection for each experiment type with standard deviation area.
    """

    def plot_summary_data(axs, list_of_files, exp_type, color="r"):
        const_vel_widths = []
        const_vel_deflections = []
        for file in list_of_const_vel:
            with open(file, "rb") as f:
                data: LoggingData = pkl.load(f)
            const_vel_widths.append(pd.DataFrame(data.widths_mm, columns=["widths_mm"]))
            const_vel_deflections.append(pd.DataFrame(data.deflections_mm, columns=["deflections_mm"]))
        const_vel_widths = pd.concat(const_vel_widths)
        const_vel_widths = const_vel_widths.groupby(const_vel_widths.index).mean()
        const_vel_widths = const_vel_widths.reset_index()
        const_vel_deflections = pd.concat(const_vel_deflections)
        const_vel_deflections = const_vel_deflections.groupby(const_vel_deflections.index).mean()
        const_vel_deflections = const_vel_deflections.reset_index()

        const_vel_widths["exp_type"] = "Constant Velocity"
        const_vel_deflections["exp_type"] = "Constant Velocity"

        axs[0].plot(const_vel_widths["widths_mm"], label=exp_type, color=color)
        axs[0].fill_between(const_vel_widths.index, const_vel_widths["widths_mm"] - const_vel_widths["widths_mm"].std(),
                            const_vel_widths["widths_mm"] + const_vel_widths["widths_mm"].std(), alpha=0.3, color=color)
        axs[1].plot(const_vel_deflections["deflections_mm"], label=exp_type, color=color)
        axs[1].fill_between(const_vel_deflections.index,
                            const_vel_deflections["deflections_mm"] - const_vel_deflections["deflections_mm"].std(),
                            const_vel_deflections["deflections_mm"] + const_vel_deflections["deflections_mm"].std(),
                            alpha=0.3, color=color)

    fig, axs = plt.subplots(2, 1)
    if list_of_const_vel is not None:
        plot_summary_data(axs, list_of_const_vel, exp_type="Constant Velocity", color="r")
    if list_of_adaptive is not None:
        plot_summary_data(axs, list_of_adaptive, exp_type="DECAF", color="b")



    plt.legend()
    plt.show()

def wavefront_dist_vs_deflection(file: str | PathLike):
    t_death = 100
    try:
        with open(file, "rb") as f:
            data: LoggingData = pkl.load(f)
    except EOFError:
        return [], []
    if isinstance(data, dict):
        data = LoggingData(**data)
    elif hasattr(data, "positions"):
        convert_old_data_log(file)
        with open(file, "rb") as f:
            data: LoggingData = pkl.load(f)
    wavefront_dists = []
    defls = []
    for frame, v, defl in zip(data.thermal_frames, data.velocities, data.deflections_mm):
        if defl==0:
            continue
        tool_tip = find_tooltip(frame, t_death)
        if tool_tip is None:
            continue
        defls.append(defl)
        _, ellipse = cv_isotherm_width(frame, 60)

        wavefront_dists.append(find_wavefront_distance(ellipse, tool_tip) / thermal_px_per_mm)
    return list(wavefront_dists), defls

use_position = True
dirs = "../logs/3mm_step/", "../logs/3mm_const_vel_tuning/", "../logs/3mm_thick_step/", "../logs/2mm_thick_step/"
fnames = []
ax = None
summary = pd.DataFrame(columns=["widths_mm", "deflections_mm", "time", "exp_type", "mean_speed", 'success', "Step Size (mm)"])
if len(fnames) == 0:
    fnames = []
    for dir in dirs:
        fnames += [dir + f for f in listdir(dir) if f.endswith(".pkl") and f.startswith("data")]
for file in fnames:
    with open(file, "rb") as f:
        df: pd.DataFrame = pkl.load(f)
    thermal_params = df["thermal_estimates"][0].keys()
    for param in thermal_params:
        df[param] = df["thermal_estimates"].apply(lambda x: x[param])
    deflection_params = df["deflection_estimates"][0].keys()
    for param in deflection_params:
        df[param] = df["deflection_estimates"].apply(lambda x: x[param])
    # df.plot(subplots=True, kind='line', y=["deflections_mm", "c_defl"])
    # plt.show()
    # df["position_mm"] = np.cumsum(df["velocities"] * 1 / 24)
    # df.drop(columns=["thermal_frames"], inplace=True)
    # print(df["position_mm"].max())
    success = df["position_mm"].max() > 200 or "adaptive" in file and df["deflections_mm"].max() < 10
    exp_type = "adaptive" if "adaptive" in file else "constant"
    step_size = float(file.split("step")[0].split("mm")[0].split("/")[2].split("_")[-1])
    mean_speed = df["velocities"].mean() if exp_type == "adaptive" else float(file.split("/")[-1].split("_")[1].split("mm")[0])
    summary.loc[-1] = [df["widths_mm"].mean(), df["deflections_mm"].mean(), df["time_sec"].max(),
                       exp_type, mean_speed, success, step_size]
    summary.index = summary.index + 1

width_res = ttest_ind(summary[summary["exp_type"] == "adaptive"]["widths_mm"], summary[summary["exp_type"] == "constant"]["widths_mm"])
defl_res = ttest_ind(summary[summary["exp_type"] == "adaptive"]["deflections_mm"], summary[summary["exp_type"] == "constant"]["deflections_mm"])
print(f"Widths: {width_res}")
print(f"Deflections: {defl_res}")
# summary.set_index("exp_type", inplace=True)
fig, ax = plt.subplots(1, 2)
summary.rename(columns={"widths_mm": "Width (mm)", "deflections_mm": "Deflection (mm)",
                        "time": "Time (s)", "success": "Success",
                        "exp_type": "Experiment Type"}, inplace=True)
# summary.plot(subplots=True, ax=ax, kind='box', by="exp_type", column=["Width (mm)", "Deflection (mm)"])
sns.boxplot(data=summary, x="Experiment Type", y="Width (mm)", ax=ax[0], showfliers=False, width=0.2)
sns.boxplot(data=summary, x="Experiment Type", y="Deflection (mm)", ax=ax[1], showfliers=False, width=0.2)
sns.scatterplot(data=summary, x="Experiment Type", y="Width (mm)", ax=ax[0], style="Success",
                hue="Step Size (mm)", palette="vlag", style_order=[True, False], legend=False)
sns.scatterplot(data=summary, x="Experiment Type", y="Deflection (mm)", ax=ax[1], style="Success",
                hue="Step Size (mm)", palette="vlag", style_order=[True, False], legend=True)
for dots in ax[0].collections:
    offsets = dots.get_offsets()
    jittered_offsets = offsets + np.random.normal(0, 0.02, offsets.shape)
    dots.set_offsets(jittered_offsets)
for dots in ax[1].collections:
    offsets = dots.get_offsets()
    jittered_offsets = offsets + np.random.normal(0, 0.02, offsets.shape)
    dots.set_offsets(jittered_offsets)

statannot.add_stat_annotation(ax[0], data=summary, x="Experiment Type", y="Width (mm)", box_pairs=[("adaptive", "constant")],
                                test="t-test_ind", text_format="star", loc="inside", verbose=2)
statannot.add_stat_annotation(ax[1], data=summary, x="Experiment Type", y="Deflection (mm)", box_pairs=[("adaptive", "constant")],
                                test="t-test_ind", text_format="star", loc="inside", verbose=2)
plt.show()


# plot_log_dir(list_of_files=fnames,
#                 plot_cost=False)
# summarize_data_logs(list_of_const_vel=fnames)
# dists = []
# defls = []

