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
qs = OnlineVelocityOptimizer(des_width=2 * scale)
debug = False
home_input = input("Press Enter to home the testbed or 's' to skip: ")

if home_input != 's':
    print("Homing testbed...")
    testbed.home()
    print("Testbed homed.")
else:
    print("Skipping homing.")

start_input = input("Press Enter to start the experiment or 'q' to quit: ")
if start_input == 'q':
    testbed.stop()
    exit(0)

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
ds = []
while True:
    ret, raw_frame = t3.read()
    info, lut = t3.info()
    thermal_arr = lut[raw_frame]
    temp_arrays.append(thermal_arr)

    if not ret:
        continue

    tmax = info.Tmax_C
    tmin = info.Tmin_C
    tmax_loc = info.Tmax_point
    tmin_loc = info.Tmin_point
    logger.debug(
        f"Max Temp (C): {tmax}, Min Temp (C): {tmin}, Max Temp Location: {tmax_loc}, Min Temp Location: {tmin_loc}")
    if debug:
        thermal_arr = cv.normalize(thermal_arr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        hist = cv.calcHist([thermal_arr], [0], None, [256], [0, 256])
        bins = np.arange(tmin, tmax, (tmax - tmin) / 256)
        plt.plot(bins, hist)
        plt.pause(0.01)
        plt.cla()

    # frame = cv.applyColorMap(thermal_arr.astype(np.uint8), cmapy.cmap('hot'))
    cv.imshow("Frame", 255 * (thermal_arr > 50).astype(np.uint8))
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
    ds.append(qs._deflection / scale)
    print(f"v: {v:.2f} mm/s, Width: {qs.width / scale:.2f} mm, Deflection: {qs._deflection/scale:.2f} mm")
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
fig, axs = plt.subplots(nrows=3, ncols=1)
axs[0].plot(xs, widths)
axs[0].set_title("Width vs Position")
axs[0].set_xlabel("Position (mm)")
axs[0].set_ylabel("Width (mm)")
axs[1].plot(xs, vs)
axs[1].set_title("Velocity vs Position")
axs[1].set_xlabel("Position (mm)")
axs[1].set_ylabel("Velocity (mm/s)")
axs[2].plot(xs, ds)
axs[2].set_title("Deflection vs Position")
axs[2].set_xlabel("Position (mm)")
axs[2].set_ylabel("Deflection (mm)")
plt.show()
