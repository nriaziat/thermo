from T3pro import T3pro
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import curve_fit
if __name__ == "__main__":
    t3 = T3pro()
    t_data = []
    time0 = time()
    while True:
        ret, frame = t3.read()
        info, lut = t3.info()
        if not ret:
            break
        t_max = info.Tmax_C
        t_max_loc = info.Tmax_point
        t_data.append((time()-time0, t_max))
        frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        frame = cv.applyColorMap(frame, cv.COLORMAP_HOT)
        cv.putText(frame, f"Max Temp: {t_max:.2f} C", t_max_loc, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(frame, f"Max Temp: {t_max:.2f} C", t_max_loc, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv.imshow("frame", frame)
        if cv.waitKey(1) == ord('q'):
            break
    t_data = np.array(t_data)
    t_data[:, 1] -= t_data[:, 1].min()
    t_data[:, 1] /= t_data[:, 1].max()
    t_start = np.argwhere(t_data[:, 1] > 0.1)[0][0]-1
    t_end = np.argwhere(t_data[:, 1] > 0.9)[0][0]+10
    t_data = t_data[t_start:t_end]
    t_data[:, 0] -= t_data[0, 0]
    exp_fit = lambda x, t: 1-np.exp(-x/t)
    popt, pcov = curve_fit(exp_fit, t_data[:, 0], t_data[:, 1])
    tx = np.linspace(0, t_data[-1, 0], 1000)
    fit_data = exp_fit(tx, *popt)
    print(f"Time constant: {popt[0]:.2e} s")
    # tau = np.where(t_data[:, 1] > 0.632)[0][0]
    # print(f"Time constant: {t_data[tau, 0]:.2f} s")
    plt.plot(t_data[:, 0], t_data[:, 1], '.')
    plt.plot(tx, fit_data, '--')
    plt.vlines(popt[0], 0, 1, 'r', '--')
    plt.text(popt[0]+0.01, 0.01, f"Time constant: {popt[0]:.2e} s", rotation=90)
    plt.legend(["Data", "Exponential Fit"])
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Temperature")
    plt.show()


