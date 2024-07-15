import cv2 as cv
from filterpy.kalman import KalmanFilter
import numpy as np


class ArucoTracker:
    def __init__(self):
        aruco_params = cv.aruco.DetectorParameters()
        self.aruco_detector = cv.aruco.ArucoDetector()
        self.aruco_detector.setDictionary(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50))
        self.aruco_detector.setDetectorParameters(aruco_params)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.R = np.eye(2)
        self.kf.Q = np.eye(4)
        self.kf.x = np.array([0, 0, 0, 0])
        self.init_kf = False

    def detect(self, frame):
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        if ids is not None:
            for i, tag in enumerate(ids):
                if tag == 0:
                    if not self.init_kf:
                        self.init_kf = True
                        self.kf.x = np.array([corners[i][0][0][0], corners[i][0][0][1], 0, 0])
                    else:
                        self.kf.predict()
                        self.kf.update(corners[i][0][0])
        return corners, ids

    def draw(self, frame, corners, ids):
        if ids is not None:
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
        cv.circle(frame, (int(self.kf.x[0]), int(self.kf.x[1])), 5, (0, 0, 255), -1)
        return frame


if __name__ == "__main__":
    tracker = ArucoTracker()
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        corners, ids = tracker.detect(frame)
        frame = tracker.draw(frame, corners, ids)
        cv.imshow("Aruco", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
