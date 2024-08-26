import pyrealsense2 as rs
import numpy as np
import cv2 as cv
from attr import dataclass


@dataclass
class PixelPoint:
    x: float
    y: float

    def __sub__(self, other):
        return PixelPoint(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        return PixelPoint(self.x + other.x, self.y + other.y)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError(
                "`copy=False` isn't supported. A copy is always created."

            )
        return np.array([self.x, self.y], dtype=dtype, copy=True)

@dataclass
class RealPoint:
    x: float
    y: float
    z: float

    def __add__(self, other):
        return RealPoint(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return RealPoint(self.x - other.x, self.y - other.y, self.z - other.z)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError(
                "`copy=False` isn't supported. A copy is always created."

            )
        return np.array([self.x, self.y, self.z], dtype=dtype, copy=True)

class RealsenseDeformationTracker:
    def __init__(self, max_num_points=3):
        self._intrinsics = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.colorizer = rs.colorizer()
        self.old_gray = None
        self.p0 = []
        self.total_def = 0
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._lk_params = dict( winSize  = (15, 15),
                          maxLevel = 2,
                          criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # Start streaming
        profile = self.pipeline.start(self.config)
        self.intr = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.extrinsic = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        self.p0 = []
        self.total_def = 0

        self.lower = np.array([40, 100, 100])
        self.upper = np.array([75, 255, 255])
        self.color_frame = None
        self.aligned_depth_frame = None

        self.relaxed_points: list[RealPoint] = []
        self.current_points: list[RealPoint] = []
        self.current_px_points: list[PixelPoint] = []

        self._max_num_points = max_num_points

    def __del__(self):
        self.pipeline.stop()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        align = rs.align(rs.stream.color)
        frameset = align.process(frames)
        self.aligned_depth_frame = frameset.get_depth_frame()
        self._intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_frame = frames.get_color_frame()
        if not depth_frame or not self.color_frame:
            return False, None
        self.color_frame =np.asanyarray(self.color_frame.get_data())
        return True, self.color_frame.copy()

    def init_points(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower, self.upper)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.dilate(mask, kernel, iterations=2)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_i = min(len(contours), self._max_num_points)
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:max_i]
        self.current_points = []
        self.current_px_points = []
        self.relaxed_points = []
        for contour in contours:
            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            depth2 = self.aligned_depth_frame.get_distance(cX, cY)
            res = rs.rs2_deproject_pixel_to_point(self._intrinsics, [cX, cY], depth2)
            self.current_points.append(RealPoint(*res))
            self.current_px_points.append(PixelPoint(cX, cY))
            self.relaxed_points.append(RealPoint(*res))

    def track_points(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.old_gray is None:
            self.old_gray = frame_gray.copy()
        p0 = np.array([point for point in self.current_px_points], dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, p0, None, **self._lk_params)
        if p1 is not None:
            self.current_points = []
            self.current_px_points = []
            good_new = p1[st == 1]
            for new_point in good_new:
                if not (0 <= new_point[0] < frame.shape[1] and 0 <= new_point[1] < frame.shape[0]):
                    continue
                depth2 = self.aligned_depth_frame.get_distance(int(new_point[0]), int(new_point[1]))
                res2 = rs.rs2_deproject_pixel_to_point(self._intrinsics, [new_point[0], new_point[1]], depth2)
                if len(self.relaxed_points) == 0:
                    self.relaxed_points = [RealPoint(*res2)]
                self.current_points.append(RealPoint(*res2))
                self.current_px_points.append(PixelPoint(*new_point))

        self.old_gray = frame_gray.copy()

    def get_deformation(self, frame):
        if len(self.relaxed_points) == 0:
            self.init_points(frame)
        self.track_points(frame)
        deflection = [np.linalg.norm(point1 - point2) for point1, point2 in zip(self.relaxed_points, self.current_points)]
        return deflection


if __name__ == "__main__":
    tracker = RealsenseDeformationTracker(max_num_points=5)
    success, frame = tracker.read()
    while success:
        success, frame = tracker.read()
        deflection = tracker.get_deformation(frame)
        for point in tracker.current_px_points:
            cv.circle(frame, (int(point.x), int(point.y)), 5, (0, 255, 0), -1)
        cv.imshow("Frame", frame)
        if key:=(cv.waitKey(1) & 0xFF) == ord('q'):
            break
        elif key == ord('r'):
            success, frame = tracker.read()
            tracker.init_points(frame)

