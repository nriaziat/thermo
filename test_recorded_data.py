import pickle as pkl
import cv2
from QuasistaticSource import OnlineVelocityOptimizer, cv_isotherm_width
import matplotlib.pyplot as plt
import numpy as np
import ast
import cmapy
from datetime import datetime
import pandas as pd
from filterpy.kalman import KalmanFilter

scale = 5.1337  # px per mm
des_width = 5  # mm

therm_images = pkl.load(open("logs/temp_2024-05-20-14:52.pkl", "rb"))
log_file = open("logs/log_2024-05-20-14:52.log", "r")
log_lines = log_file.readlines()
log_lines = [line for line in log_lines if "Temp" not in line]
log_file.close()
video_save = cv2.VideoWriter(f"logs/processed_2024-05-20-14:52.mp4", cv2.VideoWriter.fourcc(*'mp4v'), 30, (384, 288))
p = 0
ti = None

df = pd.DataFrame(columns=["Time", "Velocity", "Width", "Deflection"])
starting_x = None
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.array([0, 0, 0, 0])
kf.F = np.array([[1, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 1],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0]])
kf.P *= 1000
kf.R = np.array([[0.1, 0],
                    [0, 0.1]])
kf.Q = np.array([[0.001, 0, 0, 0],
                    [0, 0.001, 0, 0],
                    [0, 0, 0.001, 0],
                    [0, 0, 0, 0.001]])
deflection= None

ovo = OnlineVelocityOptimizer()
new_v = None

for i, (line, therm_frame) in enumerate(zip(log_lines, therm_images)):

    dict_str = line.split('DEBUG')[1]
    try:
        data = ast.literal_eval(dict_str)
    except SyntaxError:
        continue

    time = line.split('main')[0].strip()
    tj = datetime.strptime(time, "%H:%M:%S,%f")
    v = data['v']
    # if new_v is None:
    #     new_v = v
    w = data['width'] / scale

    new_v, _ = ovo.send_deflection_to_velopt(v, therm_frame)

    frame = 255 * (therm_frame > 50).astype(np.uint8)
    color_frame = cv2.applyColorMap(therm_frame.astype(np.uint8), cmapy.cmap('hot'))
    cv2.putText(color_frame, f"v: {v:.2f} mm/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(color_frame, f"width: {w:.2f} mm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(color_frame, f"new_v: {new_v:.2f} mm/s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if starting_x is None and (therm_frame > 2*therm_frame.mean()).any():
        starting_x = np.array(np.unravel_index(np.argmax(therm_frame), therm_frame.shape))
        kf.x = np.array([starting_x[1], 0, starting_x[0], 0])
        print("Starting x: ", starting_x)
    elif starting_x is not None:
        hot_point = np.array(np.unravel_index(np.argmax(therm_frame), therm_frame.shape))
        kf.predict()
        kf.update(hot_point)
        hot_point = int(kf.x[0]), int(kf.x[2])
        deflection = np.linalg.norm(hot_point - starting_x) / scale
        cv2.putText(color_frame, f"Deflection: {deflection:.2f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.arrowedLine(color_frame, (starting_x[1], starting_x[0]), (hot_point[1], hot_point[0]), (255, 255, 255), 2)

    df.loc[i] = [tj, v, w, deflection]
    cv2.imshow("Frame", color_frame)
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        break
    video_save.write(color_frame)

df.sort_values("Time", inplace=True)
df["Position"] = np.cumsum(df["Velocity"] * df["Time"].diff().dt.total_seconds())
plt.plot(df["Position"], df["Velocity"], label="Velocity")
plt.xlabel("Position (mm)")
plt.legend()
plt.ylabel("Velocity (mm/s)")
plt.twinx()
plt.plot(df["Position"], df["Width"], color='r', label="Width")
plt.ylabel("Width (mm)")
plt.legend()
plt.twinx()
plt.plot(df["Position"], df["Deflection"], color='g', label="Deflection")
plt.ylabel("Deflection (mm)")
plt.legend()
plt.show()
video_save.release()
