import numpy as np
from testbed import Testbed
from QuasistaticSource import OnlineVelocityOptimizer
import cv2 as cv
from T3pro import T3pro
import matplotlib.pyplot as plt
import logging
import datetime
import pickle as pkl

date = datetime.datetime.now()

logging.basicConfig(filename=f"logs/log_{date.strftime('%Y-%m-%d-%H:%M')}.log",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger("main")

testbed = Testbed()
t3 = T3pro()

scale = 5.1337 # px per mm
qs = OnlineVelocityOptimizer(des_width=10 / scale)
debug = False
testbed.home()
input("Press Enter to start the testbed: ")

video_save = cv.VideoWriter(f"logs/output_{date.strftime('%Y-%m-%d-%H:%M')}.avi", cv.VideoWriter.fourcc(*'XVID'), 30, (384, 288))
start = False

testbed.set_speed(qs.v)
cm = plt.get_cmap('hot')

temp_arrays = []
while True:
    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = lut[raw_frame]
    temp_arrays.append(thermal_arr)

    if not ret:
        continue

    tmax = info.Tmax_C
    tmin = info.Tmin_C
    logger.debug(f"Max Temp (C): {tmax}, Min Temp (C): {tmin}")
    if debug:
        thermal_arr = cv.normalize(thermal_arr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        hist = cv.calcHist([thermal_arr], [0], None, [256], [0, 256])
        bins = np.arange(tmin, tmax, (tmax - tmin) / 256)
        plt.plot(bins, hist)
        plt.pause(0.01)
        plt.cla()

    frame = cv.applyColorMap(raw_frame.astype(np.uint8), cv.COLORMAP_HOT)
    # frame = cm(raw_frame)
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
        # ret = testbed.set_speed(v)
        logger.debug(qs.get_loggable_data())
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

pkl.dump(temp_arrays, open(f"logs/temp_{date.strftime('%Y-%m-%d-%H:%M')}.pkl", "wb"))
video_save.release()
t3.release()
