import cv2 as cv
import csv
import numpy as np

fx = None
fy = None
PPx = None
PPy = None
with open("../realsense_captures/hydrogel_scan2_Depth_metadata.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        if "Fx" in line:
            fx = float(line[1])
        elif "Fy" in line:
            fy = float(line[1])
        elif "PPx" in line:
            PPx = float(line[1])
        elif "PPy" in line:
            PPy = float(line[1])
        if fx is not None and fy is not None and PPx is not None and PPy is not None:
            break

cal_matrix = np.array([[fx, 0, PPx], [0, fy, PPy], [0, 0, 1]])
rgb = cv.imread("../realsense_captures/hydrogel_scan2_Color.png", cv.IMREAD_COLOR)
hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
hsv = cv.undistort(hsv, cal_matrix, None)
hsv = cv.GaussianBlur(hsv, (5, 5), 0)

def nothing(x):
    pass

cv.namedWindow('HSV Tuning')
cv.createTrackbar('H Min', 'HSV Tuning', 0, 179, nothing)
cv.createTrackbar('S Min', 'HSV Tuning', 0, 255, nothing)
cv.createTrackbar('V Min', 'HSV Tuning', 0, 255, nothing)
cv.createTrackbar('H Max', 'HSV Tuning', 179, 179, nothing)
cv.createTrackbar('S Max', 'HSV Tuning', 255, 255, nothing)
cv.createTrackbar('V Max', 'HSV Tuning', 255, 255, nothing)

while True:
    h_min = cv.getTrackbarPos('H Min', 'HSV Tuning')
    s_min = cv.getTrackbarPos('S Min', 'HSV Tuning')
    v_min = cv.getTrackbarPos('V Min', 'HSV Tuning')
    h_max = cv.getTrackbarPos('H Max', 'HSV Tuning')
    s_max = cv.getTrackbarPos('S Max', 'HSV Tuning')
    v_max = cv.getTrackbarPos('V Max', 'HSV Tuning')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(rgb, rgb, mask=mask)

    # cv.imshow('Original', rgb)
    # cv.imshow('Mask', mask)
    cv.imshow('Result', result)

    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'ESC' to exit
        break

cv.destroyAllWindows()