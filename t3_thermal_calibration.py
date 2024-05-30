from T3pro import T3pro
import numpy as np
import cv2 as cv


t3 = T3pro()  # Change the port number to the correct
ret, raw_frame = t3.read()

while ret:
    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = lut[raw_frame]
    error_arr = np.abs(thermal_arr - 50)
    error_img = 255 * (error_arr > 10).astype(np.uint8)
    error_img = cv.applyColorMap(error_img, cv.COLORMAP_JET)
    cv.imshow("Frame", error_img)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
# print(f"Max Error: {np.max(error_arr)}, Mean Error: {np.mean(error_arr)}")
