import pickle as pkl
import cv2
from QuasistaticSource import OnlineVelocityOptimizer, cv_isotherm_width
import matplotlib.pyplot as plt
import numpy as np
import ast
import cmapy

scale = 5.1337  # px per mm
des_width = 5  # mm

therm_images = pkl.load(open("logs/temp_2024-05-17-15:04.pkl", "rb"))
log_file = open("logs/log_2024-05-17-15:04.log", "r")
log_lines = log_file.readlines()
log_lines = [line for line in log_lines if "Temp" not in line]
log_file.close()
video_save = cv2.VideoWriter(f"logs/processed_2024-05-17-15:04.mp4", cv2.VideoWriter.fourcc(*'XVID'), 30, (384, 288))
for line, raw_frame in zip(log_lines, therm_images):
    dict_str = line.split('DEBUG')[1]
    data = ast.literal_eval(dict_str)
    v = data['v']
    w = data['width'] / scale
    _, ellipse = cv_isotherm_width(raw_frame, 50)
    frame = cv2.normalize(raw_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    frame = cv2.applyColorMap(frame, cmapy.cmap('hot'))
    cv2.ellipse(frame, ellipse, (255, 255, 255), 1)
    cv2.putText(frame, f"v: {v:.2f} mm/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"width: {w:.2f} mm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    video_save.write(frame)
video_save.release()

qs = OnlineVelocityOptimizer(des_width=des_width * scale)
widths = []
vs = []
v = 2
for frame in therm_images:
    binary_frame = frame > 50
    binary_frame = binary_frame.astype(np.uint8) * 255
    binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2BGR)
    cv2.waitKey(1)
    v, ellipse = qs.update_velocity(v, frame)
    cv2.ellipse(binary_frame, ellipse, (0, 0, 255), 1)
    width = qs.width / scale
    cv2.putText(binary_frame, f"v: {v:.2f} mm/s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(binary_frame, f"width: {width:.2f} mm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("bin frame", binary_frame)
    # cv.ellipse(frame, ellipse, 255, 1)
    width = qs.width / scale
    widths.append(width)
    vs.append(v)
    # print(f"v: {v}, width: {width}")
    plt.pause(0.01)
#
# plt.plot(widths)
# plt.title("Width of the 50C isotherm and Velocity")
# plt.ylabel("Width (mm)")
# plt.hlines(des_width, 0, len(widths), colors='r', linestyles='dashed')
# plt.twinx()
# plt.plot(vs, 'r')
# plt.legend(["Cut Width (mm)", "Velocity (mm/s)"])
# plt.show()
