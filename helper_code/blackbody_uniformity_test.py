from T3pro import T3pro
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

t3 = T3pro()  # Change the port number to the correct
ret, raw_frame = t3.read()
t_ref = 50

if __name__ == "__main__":

    while ret:
        ret, frame = t3.read()
        info, lut = t3.info()
        color_frame = cv.applyColorMap(cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLORMAP_HOT)
        cv.imshow("Frame", color_frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord("c"):
            t3.calibrate()
    t3.release()
    temp_frame = lut[frame]
    error_frame = temp_frame - t_ref
    binary_frame = cv.normalize(error_frame.copy(), None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    binary_frame = cv.threshold(binary_frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    contours = cv.findContours(binary_frame, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = contours[0]
    else:
        print("No contours found")
    rect = cv.boundingRect(contours[0])
    mask = cv.drawContours(np.zeros_like(error_frame), contours, -1, 255, -1)
    max_error = np.max(error_frame[mask == 255])
    mean_error = np.mean(error_frame[mask == 255])
    std_error = np.std(error_frame[mask == 255])
    x, y, w, h = rect
    error_frame = error_frame[y:y+h, x:x+w]
    temp_frame = temp_frame[y:y+h, x:x+w]
    print(f"Max Error: {max_error:.2e} C, Mean Error: {mean_error:.2e} C, Std Error: {std_error:.2e} C")
    cv.imshow("Frame", error_frame)
    cv.waitKey(0)
    plt.imshow(temp_frame, cmap='hot')
    plt.axis('off')
    plt.title(f"{t_ref} C Reference Temperature")
    clb = plt.colorbar()
    clb.ax.set_title('Temperature [C] ', fontsize=8)
    plt.show()
    plt.imshow(error_frame, cmap='coolwarm')
    plt.axis('off')
    plt.title(f"Error from {t_ref} C Reference Temperature")
    # plt.clim(-1, 2)
    clb = plt.colorbar()
    clb.ax.set_title('Temperature Error [C] ', fontsize=8)
    plt.show()
