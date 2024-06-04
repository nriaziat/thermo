import numpy as np
import pickle as pkl
import cv2 as cv

with open('./logs/temp_2024-05-20-14:52.pkl', "rb") as f:
    temp_data = pkl.load(f)
for t_frame in temp_data[50:]:
    color_frame = cv.applyColorMap(cv.normalize(t_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U), cv.COLORMAP_HOT)
    t_death = 50
    binary_frame = (t_frame > t_death).astype(np.uint8)
    binary_frame_blur = cv.medianBlur(binary_frame, 5)
    contours = cv.findContours(binary_frame_blur, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = contours[0]
    else:
        continue
    contours_drawn = cv.drawContours(color_frame.copy(), contours, -1, (255, 255, 255), 2)
    list_of_pts = []
    for ctr in contours:
        if cv.contourArea(ctr) > 50:
            list_of_pts += [pt[0] for pt in ctr]
    ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
    if len(ctr) == 0:
        continue
    if cv.contourArea(ctr) < 1500:
        continue
    comb_contours_drawn = cv.drawContours(color_frame.copy(), [ctr], -1, (255, 255, 255), 2)
    hull = cv.convexHull(ctr)
    hull_drawn = cv.drawContours(color_frame.copy(), [hull], -1, (255, 255, 255), 2)
    ellipse = cv.fitEllipse(hull)
    ellipse_drawn = cv.ellipse(color_frame.copy(), ellipse, (255, 255, 255), 2)
    w = ellipse[1][0]
    imgs = [color_frame, 255*binary_frame, 255*binary_frame_blur, contours_drawn, comb_contours_drawn, hull_drawn, ellipse_drawn]
    for i, img in enumerate(imgs):
        # cv.imshow("img", img)
        #
        # cv.waitKey()
        cv.imwrite(f"logs/cv_step_{i}.png", img)
    break

