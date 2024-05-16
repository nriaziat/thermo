import pickle as pkl
from QuasistaticSource import OnlineVelocityOptimizer
import matplotlib.pyplot as plt
import cv2 as cv

data = pkl.load(open("logs/temp_2024-05-16-16:58.pkl", "rb"))
scale = 5.1337  # px per mm
des_width = 5  # mm
qs = OnlineVelocityOptimizer(des_width=des_width * scale)

widths = []
vs = []
for frame in data:
    # plt.imshow(frame, cmap='hot', interpolation='nearest')
    # plt.pause(0.01)
    v, ellipse = qs.update_velocity(frame)
    # cv.ellipse(frame, ellipse, 255, 1)
    plt.imshow(frame, cmap='hot', interpolation='nearest')
    width = qs._width / scale
    widths.append(width)
    vs.append(v)
    # print(f"v: {v}, width: {width}")
    plt.pause(0.01)

plt.plot(widths)
plt.title("Width of the 50C isotherm and Velocity")
plt.ylabel("Width (mm)")
plt.hlines(des_width, 0, len(widths), colors='r', linestyles='dashed')
plt.twinx()
plt.plot(vs, 'r')
plt.legend(["Cut Width (mm)", "Velocity (mm/s)"])
plt.show()
