from T3pro import T3pro
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

t3 = T3pro()  # Change the port number to the correct
ret, raw_frame = t3.read()


def get_error_array(t3):
    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = lut[raw_frame]
    error_arr = np.abs(thermal_arr - 50)
    return ret, error_arr


while ret:
    ret, error_arr = get_error_array(t3)
    error_img = cv.normalize(error_arr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    cv.imshow("Frame", error_img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord("c"):
        t3.calibrate()
    elif key == ord("b"):
        # take a burst of images
        error_arrs = []
        for i in range(100):
            ret, error_arr = get_error_array(t3)
            if (error_arr > 5).any():
                continue
            error_arrs.append(error_arr)
        if len(error_arrs) < 50:
            print("Not enough good frames")
            continue
        print(f"Max Error: {np.max(error_arrs):.2e} C, Mean Error: {np.mean(error_arrs):.2e} C, Std Error: {np.std(error_arrs):.2e} C")
        arr = np.mean(error_arrs, axis=0)
        plt.imshow(arr, cmap='turbo')
        clb = plt.colorbar()
        plt.axis('off')
        plt.title(f"{len(error_arrs)} averaged frames")
        clb.ax.set_title('Temperature Error [C] ', fontsize=8)
        break

t3.release()
# plt.imshow(error_arr, cmap='hot')
# plt.colorbar()
plt.show()