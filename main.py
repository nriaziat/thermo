import numpy as np
from testbed import Testbed
from QuasistaticSource import OnlineVelocityOptimizer
import cv2 as cv
from T3pro import T3pro
import matplotlib.pyplot as plt
import logging
import datetime
import pickle as pkl
import cmapy

date = datetime.datetime.now()

logging.basicConfig(filename=f"logs/log_{date.strftime('%Y-%m-%d-%H:%M')}.log",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger("main")

testbed = Testbed()
t3 = T3pro()

scale = 5.1337  # px per mm
qs = OnlineVelocityOptimizer(des_width=3 * scale)
debug = False
testbed.home()
input("Press Enter to start the testbed: ")

# video_save = cv.VideoWriter(f"logs/output_{date.strftime('%Y-%m-%d-%H:%M')}.avi", cv.VideoWriter.fourcc(*'XVID'), 30,
#                             (384, 288))
start = False

v = qs.v
testbed.set_speed(v)
cm = plt.get_cmap('hot')

temp_arrays = []
widths = []
vs = []
xs = []
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

    # frame = cv.applyColorMap(thermal_arr.astype(np.uint8), cmapy.cmap('hot'))
    cv.imshow("Frame", 255*(thermal_arr>50).astype(np.uint8))
    # video_save.write(frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        testbed.stop()
        break
    elif key == ord('c'):
        t3.calibrate()

    v, _ = qs.update_velocity(v, thermal_arr)
    vs.append(v)
    widths.append(qs.width / scale)
    xs.append(testbed.get_position())
    print(f"v: {v:.2f} mm/s, Width: {qs.width / scale:.2f} mm")
    ret = testbed.set_speed(v)
    logger.debug(qs.get_loggable_data())
    if not ret:
        testbed.stop()
        break
    pos = testbed.get_position()
    if pos == -1:
        testbed.stop()
        break
    if debug:
        print(f"Speed: {v:.2f} mm/s")

pkl.dump(temp_arrays, open(f"logs/temp_{date.strftime('%Y-%m-%d-%H:%M')}.pkl", "wb"))
# video_save.release()
t3.release()
print(f"Avg Width: {np.mean(widths):.2f} mm")
print(f"Avg Velocity: {np.mean(vs):.2f} mm/s")
plt.plot(xs, widths)
plt.title("Width vs position")
plt.xlabel("X (mm)")
plt.ylabel("Width (mm)")
plt.show()
