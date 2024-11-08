from aruco_tracker import ArucoTracker
from T3pro import T3pro
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import datetime
from testbed import Testbed
from utils import find_tooltip, thermal_frame_to_color

t3 = T3pro(port=2)
camera = cv.VideoCapture(4)
tracker = ArucoTracker()
tb = Testbed()

therm_positions = []
aruco_positions = []
therm_frames = []

tb.home()
aruco_tag_width_mm = 8
while True:
    ret1, frame = camera.read()
    ret2, raw_frame = t3.read()
    info, lut = t3.info()
    if not ret1 or not ret2:
        print("Error reading frame")
        break
    therm_frame = lut[raw_frame]
    thermal_tooltip = find_tooltip(therm_frame, 60)
    aruco_pos, ids = tracker.detect(frame)
    if thermal_tooltip is not None and ids is not None:
        aruco_tag_width_px = np.linalg.norm(aruco_pos[0][0][0] - aruco_pos[0][0][1])
        aruco_positions.append(aruco_pos[0][0][0] * aruco_tag_width_mm / aruco_tag_width_px)
        therm_positions.append(thermal_tooltip)
    color_frame = thermal_frame_to_color(therm_frame)
    if thermal_tooltip is not None:
        cv.circle(color_frame, (int(thermal_tooltip[0]), int(thermal_tooltip[1])), 5, (0, 0, 255), -1)
    cv.imshow("Thermal", color_frame)
    frame = tracker.draw(frame, aruco_pos, ids)
    cv.imshow("Aruco", frame)
    therm_frames.append(therm_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    tb.set_speed(7)

tb.stop()
camera.release()
t3.release()

plt.plot(therm_positions)
plt.plot(aruco_positions)
plt.show()

with open(f"deflection_test_{datetime.now().strftime('%Y%m%d%H%M')}.pkl", "wb") as f:
    pkl.dump({"therm": therm_positions, "aruco": aruco_positions, "therm_frames": therm_frames}, f)

