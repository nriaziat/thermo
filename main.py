import numpy as np
from testbed import Testbed
from QuasistaticSource import AdaptiveVelocityController
import cv2 as cv
from T3pro import T3pro
import matplotlib.pyplot as plt

testbed = Testbed()
t3 = T3pro()

scale = 0.13  # pixels per mm
qs = AdaptiveVelocityController(des_width=10 / scale)

testbed.home()
print("Homing...")
debug = False

video_save = cv.VideoWriter("output.avi", cv.VideoWriter.fourcc(*'XVID'), 30, (384, 288))

while True:
    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = np.array(lut[raw_frame])

    if not ret:
        continue

    if debug:
        tmax = info.Tmax_C
        tmin = info.Tmin_C
        thermal_arr = cv.normalize(thermal_arr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        hist = cv.calcHist([thermal_arr], [0], None, [256], [0, 256])
        bins = np.arange(tmin, tmax, (tmax - tmin) / 256)
        plt.plot(bins, hist)
        plt.pause(0.01)
        plt.cla()

    frame = cv.applyColorMap(raw_frame.astype(np.uint8), cv.COLORMAP_HOT)
    video_save.write(frame)
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        testbed.stop()
        break
    elif key == ord('c'):
        t3.calibrate()

    try:
        v = qs.update_velocity(thermal_arr)
        ret = testbed.set_speed(v)
        if not ret:
            testbed.stop()
            print("Endstop reached")
            break
        pos = testbed.get_position()
        if pos == -1:
            testbed.stop()
            print("Endstop reached")
            break
        if debug:
            print(f"Speed: {v:.2f} mm/s")

    except (KeyboardInterrupt, ValueError) as e:
        testbed.stop()
        video_save.release()
        t3.release()
        print(e)
        exit()

video_save.release()
t3.release()
