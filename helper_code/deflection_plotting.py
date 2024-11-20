import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from ParameterEstimation import DeflectionAdaptation
from scipy.stats import linregress
import cv2 as cv
from utils import thermal_frame_to_color, find_tooltip, list_of_frames_to_video
from matplotlib import rcParams

rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

thermal_px_per_mm = 7

with open('./deflection_test_202411051319.pkl', 'rb') as f:
    data = pkl.load(f)
    therm_defl = data['therm']
    aruco_defl = data['aruco']
    if 'therm_frames' in data:
        therm_frames = data['therm_frames']
        colors_frames = []
        # for frame in therm_frames:
        #     gray = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        #     gray = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        #     colors_frames.append(gray)
        # list_of_frames_to_video(colors_frames, 'therm_deflection.mp4')
    else:
        therm_frames = [None] * len(therm_defl)

defl_adapt = None
filtered_therm_defl = []
filtered_aruco_defl = []
neutral_pt = []
therm_defl = np.array(therm_defl)
first_tip = None

aruco_defl = np.array(aruco_defl)
min_x_aruco = aruco_defl[0]
params = []

for tip_therm, tip_aruco, therm_frame in zip(therm_defl, aruco_defl, therm_frames):
    if therm_frame is not None:
        tip_therm = find_tooltip(therm_frame, 70)
    if tip_therm is None:
        aruco_defl = aruco_defl[1:]
        therm_defl = therm_defl[1:]
        continue
    tip_therm_mm = (tip_therm[0] / thermal_px_per_mm, tip_therm[1] / thermal_px_per_mm)
    if first_tip is None:
        first_tip = np.array(tip_therm_mm)
        first_aruco_tip = np.array(tip_aruco)
    if defl_adapt is None:
        defl_adapt = DeflectionAdaptation(np.array([tip_therm_mm[0], tip_therm_mm[1], 0, 0, tip_therm_mm[0], tip_therm_mm[1], 15]),
                                          labels=['x', 'y', 'xd', 'yd', 'xn', 'yn', 'd_defl'])
        aruco_adapt = DeflectionAdaptation(np.array([tip_aruco[0], tip_aruco[1], 0, 0, tip_aruco[0], tip_aruco[1], 15]),
                                          labels=['x', 'y', 'xd', 'yd', 'xn', 'yn', 'd_defl'])
    z_therm = np.array([tip_therm_mm[0], tip_therm_mm[1]])
    defl_adapt.update(z_therm, v=7)
    z_aruco = np.array([tip_aruco[0], tip_aruco[1]])
    aruco_adapt.update(z_aruco, v=7)
    params.append(defl_adapt.kf.x[6:9])
    filtered_therm_defl.append(defl_adapt.defl_mm)
    filtered_aruco_defl.append(aruco_adapt.defl_mm)
    neutral_pt.append(defl_adapt.kf.x[4:6])
    if therm_frame is None:
        therm_frame = np.zeros((288, 384))
    color_frame = thermal_frame_to_color(therm_frame)
    cv.circle(color_frame, (int(tip_therm[0]), int(tip_therm[1])), 5, (0, 0, 255), -1)
    cv.circle(color_frame, (int(defl_adapt.kf.x[4] * thermal_px_per_mm), int(defl_adapt.kf.x[5] * thermal_px_per_mm)), 5, (0, 255, 0), -1)
    cv.circle(color_frame, (int(defl_adapt.kf.x[0] * thermal_px_per_mm), int(defl_adapt.kf.x[1] * thermal_px_per_mm)), 5, (255, 0, 0), -1)
    cv.putText(color_frame, f"Estimated Values: {defl_adapt.kf.x[6]:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv.imshow("Thermal", color_frame)
    # cv.waitKey(10)

filtered_therm_defl = np.array(filtered_therm_defl)
filtered_aruco_defl = np.array(filtered_aruco_defl)
therm_defl = np.linalg.norm(therm_defl - therm_defl[0], axis=1) / thermal_px_per_mm
aruco_defl = np.linalg.norm(aruco_defl - min_x_aruco, axis=1)
aruco_defl /= 124
aruco_defl *= 140
for i, defl in enumerate(aruco_defl):
    if defl > 1.5:
        aruco_defl[i] = aruco_defl[i-1]

error = aruco_defl - filtered_therm_defl
print(f"STD: {np.std(error):.2f} mm")
print(f"RMSE: {np.sqrt(np.mean(error**2)):.2f} mm")
linreg = linregress(aruco_defl, filtered_therm_defl)


plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].plot(therm_defl, '.b', label='Raw Thermal Signal', alpha=0.1)
ax[0].plot(aruco_defl, '--r', label='Aruco Signal', alpha=1)
ax[0].plot(filtered_therm_defl, 'b', label='Filtered Thermal Signal', linewidth=2)
# ax[0].plot(filtered_aruco_defl, 'r', label='Filtered Aruco', linewidth=2)
ax[0].legend()
ax[0].set_ylabel("Deflection [mm]")
ax[1].scatter(aruco_defl, filtered_therm_defl, marker='.', label='Filtered Thermal Signal', alpha=0.25)
x = np.linspace(0, max(aruco_defl), 25)
ax[1].plot(x, x, 'g--', label='y=x')
pval_order = np.floor(np.log10(linreg.pvalue))
pval_digits = linreg.pvalue * 10**-pval_order
# ax[1].plot(x, linreg.slope * x + linreg.intercept, 'r', label=r'$R^2={{{:.2f}}}, p={{{:.1f}}} * 10^{{{:.0f}}}$'.format(linreg.rvalue**2, pval_digits, pval_order))
ax[1].plot(x, linreg.slope * x + linreg.intercept, 'r', label=r'y={:.2f}x + {:.2f}'.format(linreg.slope, linreg.intercept))
ax[1].set_xlabel("Aruco Tag Deflection [mm]")
ax[1].set_ylabel("Thermal Camera Deflection [mm]")
ax[1].legend()
ax[1].set_title("Deflection Comparison")

# ax[0].set_ylim(-0.1, 1.2)
# ax[1].set_ylim(-0.1, 1.2)
ax[0].set_title("Deflection Measurement")

fig, ax = plt.subplots()
params = np.array(params)
ax.plot(params, label='Decay rate [mm/s]')

plt.show()
