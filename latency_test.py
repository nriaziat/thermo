from testbed import Testbed
from T3pro import T3pro
from QuasistaticSource import OnlineVelocityOptimizer as OVO
import cv2 as cv
import time
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

testbed = Testbed()
t3 = T3pro()
OVO = OVO(des_width=2 * 5.1337, t_death=15)
with open("logs/temp_2024-05-20-14:52.pkl", "rb") as f:
    temp_arrays = pkl.load(f)
test_frame = temp_arrays[-1]
ts = []
if __name__ == "__main__":
    for i in range(1000):
        t1 = time.time()
        testbed.get_position()
        ret, raw_frame = t3.read()
        if not ret:
            break
        # norm_frame = cv.normalize(raw_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # color_frame = cv.applyColorMap(norm_frame, cv.COLORMAP_HOT)
        # cv.imshow("Thermal Camera", color_frame)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break
        OVO.update_velocity(0, raw_frame)
        testbed.set_speed(0)
        t2 = time.time()
        ts.append(t2 - t1)
    ts = ts[1:]
    median = np.median(ts)
    mean = np.mean(ts)
    std = np.std(ts)
    print(f"Median: {median}, Mean: {mean}, Std: {std}")

    plt.hist(ts, bins=100)
    plt.ylabel("Frequency")
    plt.xlabel("Time (s)")
    plt.vlines(median, 0, 200, colors='r', linestyles='dashed')
    # plt.vlines(mean, 0, 100, colors='g', linestyles='dashed')
    # plt.text(median, 100, f"Median: {median:.2e}", rotation=90)
    # plt.text(mean, 100, f"Mean: {mean:.2e}", rotation=90)
    plt.show()

    # plt.plot(ts)
    # plt.show()
