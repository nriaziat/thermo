from ExperimentManager import ExperimentManager
from aruco_tracker import ArucoTracker
from T3pro import T3pro
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime

t3 = T3pro(port=0)
camera = cv.VideoCapture(4)
tracker = ArucoTracker()
em = ExperimentManager(camera=camera, thermal_camera=t3, adaptive_velocity=False)

init_pose = None
deflection = 0
scale_factor = None

therm_deflections = []
aruco_deflections = []

em._prepare_experiment()

therm_frames = []

while True:
    ret, frame = em.read_camera()
    ret2, thermal_arr, raw_frame, _ = em.get_t3_frame()
    norm_frame = cv.normalize(thermal_arr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    color_frame = cv.applyColorMap(norm_frame, cv.COLORMAP_HOT)

    if not ret or not ret2:
        continue

    therm_frames.append(thermal_arr)
    corners, ids = tracker.detect(frame)
    if scale_factor is None:
        for i, tag in enumerate(ids):
            if tag == 0:
                scale_factor = np.linalg.norm(corners[i][0][0] - corners[i][0][1]) / 10
                break

    if init_pose is None and tracker.init_kf:
        init_pose = tracker.kf.x[:2]
        cv.imshow("Find tool tip", color_frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
        neutral_point_str = input("Enter neutral pose coordinates x,y: ")
        neutral_point = (int(neutral_point_str.split(',')[1]), int(neutral_point_str.split(',')[0]))
        # print(f"Neutral point: {neutral_point}")
        em.mpc.neutral_tip_pos = neutral_point
        em.set_speed(5)
    else:
        deflection = np.linalg.norm(tracker.kf.x[:2] - init_pose) / scale_factor

    em.update_measurements(0, thermal_arr)
    therm_deflection = em.thermal_deflection
    tool_pos = em.thermal_tool_tip_estimate

    if tool_pos is not None:
        therm_deflections.append(therm_deflection)
        cv.circle(color_frame, (int(tool_pos[1]), int(tool_pos[0])), 5, (255, 0, 0), -1)
        cv.putText(color_frame, f"Thermal Deflection: {therm_deflection:.2f} mm", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                      (255, 255, 255), 1)
        aruco_deflections.append(deflection)

    frame = tracker.draw(frame, corners, ids)
    cv.imshow("Aruco", frame)
    cv.imshow("Thermal", color_frame)

    pos = em.testbed.get_position()
    if pos == -1 or cv.waitKey(1) & 0xFF == ord('q'):
        break

em._end_experiment()
cv.destroyAllWindows()

therm_deflection = np.array(therm_deflections)
aruco_deflection = np.array(aruco_deflections)
plt.plot(therm_deflections, label="Thermal")
plt.plot(aruco_deflections, label="Aruco")

date = datetime.now().strftime("%Y-%m-%d-%H:%M")
with open(f"./logs/deflection_data_{date}.pkl", "wb") as f:
    pkl.dump((therm_frames, therm_deflections, aruco_deflections), f)

plt.legend()
plt.show()