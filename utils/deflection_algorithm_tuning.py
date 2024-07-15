import pickle as pkl
import cv2 as cv
import numpy as np
from QuasistaticSource import find_tooltip, OnlineVelocityOptimizer
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from scipy.signal import butter, sosfilt, correlate

filt = butter(10, 2, 'low', fs=24, output='sos')

with open('../logs/deflection_data_2024-07-09-14:44.pkl', 'rb') as f:
    therm_frames, _, aruco_defl = pkl.load(f)

aruco_defl = np.array(aruco_defl)
aruco_defl = aruco_defl

qs = OnlineVelocityOptimizer(des_width=0)
neutral_index = None
thermal_deflections = []

first_x = None
gaus_kernel = cv.getGaussianKernel(3, 0)
sobel_kernel_size = 3
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

deflection_kf = KalmanFilter(dim_x=2, dim_z=1)
deflection_kf.x = np.array([0, 0])
k = 0.
b = 0.
deflection_kf.F = np.eye(2) + np.array([[0, 1], [-k, -b]])
deflection_kf.H = np.array([[1, 0]])
deflection_kf.P *= 1000
deflection_kf.R *= 5
deflection_kf.Q = np.eye(2)
neutral_tip_pos = (183, 355)
neutral_tip_pos = np.array(neutral_tip_pos)

for i, (frame, defl) in enumerate(zip(therm_frames, aruco_defl)):
    norm_frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    qs.send_deflection_to_velopt(0, frame)
    tip = find_tooltip(frame, 50, (neutral_tip_pos[0], neutral_tip_pos[1]))
    if tip is None:
        continue
    tip=np.array(tip)
    meas = np.linalg.norm(tip - neutral_tip_pos) / 5.1337
    deflection_kf.predict()
    deflection_kf.update(meas)
    deflection = deflection_kf.x[0]
    thermal_deflections.append(deflection)
    color_frame = cv.applyColorMap(norm_frame, cv.COLORMAP_HOT)
    cv.putText(color_frame, f"Thermal Deflection: {deflection:.2f} mm", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(color_frame, f"Aruco Deflection: {defl:.2f} mm", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.circle(color_frame, (tip[1], tip[0]), 5, (255, 255, 255), -1)
    # cv.imshow("Thermal Camera", color_frame)
    # key = cv.waitKey(30) & 0xFF
    # if key == ord('q'):
    #     break


thermal_deflections = np.array(thermal_deflections)
aruco_defl = np.array(aruco_defl)[:len(thermal_deflections)]
t = np.arange(len(thermal_deflections)) / 24

# thermal_deflections = sosfilt(filt, thermal_deflections)
# aruco_defl = sosfilt(filt, aruco_defl)
# corr = correlate(thermal_deflections, aruco_defl)
# max_corr = np.argmax(corr)
# aruco_defl = np.roll(aruco_defl, max_corr)
error = (thermal_deflections - aruco_defl)
print(f"RMS Error: {np.sqrt(np.mean(error**2)):.2f} mm")

# thermal_deflections = np.array(thermal_deflections)
plt.plot(t, thermal_deflections, label="Thermal")
plt.plot(t, aruco_defl, label="Aruco")
# plt.plot(t, np.abs(error), label="Error", linestyle="--")
plt.ylabel("Deflection (mm)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()
