import numpy as np
import pandas as pd
# import hysteresis as hys
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

import matplotlib.pyplot as plt
from scipy.stats import linregress

with open("electrocautery_stiffness.xlsx", "rb") as f:
    df = pd.read_excel(f, "hysteresis")

asc1 = df[["Deflection", "Force Ascending"]]
desc1 = df[["Deflection", "Force Descending"]].dropna()
asc2 = df[["Deflection", "Force Ascending.1"]]
desc2 = df[["Deflection", "Force Descending.1"]].dropna()
asc3 = df[["Deflection", "Force Ascending.2"]]
desc3 = df[["Deflection", "Force Descending.2"]].dropna()

asc = np.vstack((asc1, asc2, asc3))
desc = np.vstack((desc1, desc2, desc3))

# linear_fit_asc = np.polyfit(asc[:, 0], asc[:, 1], 1)
# linear_fit_desc = np.polyfit(desc[:, 0], desc[:, 1], 1)
# linear_fit_mean = np.polyfit(np.hstack((asc[:, 0], desc[:, 0])), np.hstack((asc[:, 1], desc[:, 1])), 1)
linear_fit_asc = linregress(asc[:, 0], asc[:, 1])
linear_fit_desc = linregress(desc[:, 0], desc[:, 1])
linear_fit_mean = linregress(np.hstack((asc[:, 0], desc[:, 0])), np.hstack((asc[:, 1], desc[:, 1])))

print(f"Mean Ascending Fit: {linear_fit_asc[0]:.2f}x + {linear_fit_asc[1]:.2f}, R^2: {linear_fit_asc.rvalue:.2f}")
print(f"Mean Descending Fit: {linear_fit_desc[0]:.2f}x + {linear_fit_desc[1]:.2f}, R^2: {linear_fit_desc.rvalue:.2f}")
print(f"Mean Fit: {linear_fit_mean[0]:.2f}x + {linear_fit_mean[1]:.2f}, R^2: {linear_fit_mean.rvalue:.2f}")

asc_desc = np.vstack((asc, desc))

fig, ax = plt.subplots()
ax.plot(asc[:, 0], asc[:, 1], label="Ascending", color="green", linewidth=0.5, alpha=0.5)
ax.plot(desc[:, 0], desc[:, 1], label="Descending", color="red", linewidth=0.5, alpha=0.5)
ax.plot(asc1["Deflection"], linear_fit_asc.slope * asc1["Deflection"] + linear_fit_asc.intercept, "--",
        label="Linear Ascending Fit", color="green")
ax.plot(desc1["Deflection"], linear_fit_desc.slope * desc1["Deflection"] + linear_fit_desc.intercept, "--",
        label="Linear Descending Fit", color="red")
ax.plot(asc1["Deflection"], linear_fit_mean.slope * asc1["Deflection"] + linear_fit_mean.intercept, "--", label="Linear Mean Fit", color="blue")
ax.text(0.5, 0.05, f"Mean Ascending Fit: {linear_fit_asc.slope:.2f}x + {linear_fit_asc.intercept:.2f}, R^2: {linear_fit_asc.rvalue:.2f}\n"
                    f"Mean Descending Fit: {linear_fit_desc.slope:.2f}x + {linear_fit_desc.intercept:.2f}, R^2: {linear_fit_desc.rvalue:.2f}\n"
                    f"Mean Fit: {linear_fit_mean.slope:.2f}x + {linear_fit_mean.intercept:.2f}, R^2: {linear_fit_mean.rvalue:.2f}",
        transform=ax.transAxes)
# ax.plot(asc[:, 0], asc[:, 1], "gx", label="Ascending")
# ax.plot(desc[:, 0], desc[:, 1], "rx", label="Descending")
ax.legend()
ax.set_xlabel("Deflection (mm)")
ax.set_ylabel("Force (N)")
plt.show()

# myhys = hys.Hysteresis(asc_desc)
# myhys.plot(showReversals=True)
# slope = myhys.slope
# # backbone = hys.envelope.getAvgBackbone(myhys, LPsteps=[3] * 20)
# # backbone.plot()
# # ax.set_title(f"Stiffness: {slope:.2f} N/mm")
# ax.set_xlabel("Deflection (mm)")
# ax.set_ylabel("Force (N)")
# plt.show()
