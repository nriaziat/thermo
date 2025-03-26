import zivid
import cv2 as cv 
import numpy as np

app = zivid.Application()
camera = app.connect_camera()

settings_file = "/home/imerse/ros2_wss/jiawei_ws/src/zivid_settings.yml"
print(f"Loading settings from file: {settings_file}")
settings = zivid.Settings()
settings.acquisitions.append(zivid.Settings.Acquisition())
settings.processing.filters.smoothing.gaussian.enabled = True
settings.processing.filters.smoothing.gaussian.sigma = 1
settings.processing.filters.reflection.removal.enabled = True
settings.processing.filters.reflection.removal.mode = "global"

settings_2d = zivid.Settings2D()
settings_2d.acquisitions.append(zivid.Settings2D.Acquisition())
settings_2d.processing.color.balance.blue = 1.0
settings_2d.processing.color.balance.green = 1.0
settings_2d.processing.color.balance.red = 1.0

settings.color = settings_2d


with camera.capture_2d_3d(settings) as frame:
    frame.save("save.zdf")