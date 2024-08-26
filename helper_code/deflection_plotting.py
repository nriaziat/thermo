import matplotlib.pyplot as plt
import pickle as pkl
from scipy.signal import correlate
import numpy as np

with open('../logs/deflection_data.pkl', 'rb') as f:
    _, therm_defl, aruco_defl = pkl.load(f)

therm_defl = np.array(therm_defl)
aruco_defl = np.array(aruco_defl) * 10
corr = correlate(therm_defl, aruco_defl, mode='full')
shift = np.argmax(corr) - len(therm_defl)
aruco_defl = np.roll(aruco_defl, shift)


error = np.abs(therm_defl[:len(aruco_defl)] - aruco_defl)

fig, ax = plt.subplots(3, 1)
ax[0].plot(therm_defl, label='Thermal')
ax[0].set_xticks([])
ax[1].plot(aruco_defl, label='Aruco')
ax[0].set_ylabel("Deflection [mm]")
ax[1].set_ylabel("Deflection [mm]")
ax[1].set_xlabel("Frame")
ax[2].plot(error, label='Error')
ax[2].set_ylabel("Error [mm]")

# ax[0].set_ylim(-0.1, 1.2)
# ax[1].set_ylim(-0.1, 1.2)
ax[0].set_title("Thermal Camera Deflection Measurement")
ax[1].set_title("Aruco Tag Deflection Measurement")
plt.show()
