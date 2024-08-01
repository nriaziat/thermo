import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import logging
import datetime
import pickle as pkl
import cmapy
from filterpy.kalman import KalmanFilter
from dataclasses import dataclass, field, astuple, asdict

from numpy.linalg.linalg import LinAlgError


def thermal_frame_to_color(thermal_frame):
    norm_frame = cv.normalize(thermal_frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return cv.applyColorMap(norm_frame, cmapy.cmap('hot'))



@dataclass
class LoggingData:
    widths: list[float] = field(default_factory=list)
    velocities: list[float] = field(default_factory=list)
    deflections: list[float] = field(default_factory=list)
    thermal_frames: list[np.ndarray] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    damping_estimates: list[float] = field(default_factory=list)
    a_hats: list[float] = field(default_factory=list)
    b_hats: list[float] = field(default_factory=list)
    width_estimates: list[float] = field(default_factory=list)

    def __iter__(self):
        return iter(zip(self.velocities, self.widths, self.deflections, self.thermal_frames, self.positions, self.damping_estimates))

    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

class ExperimentManager:

    def __init__(self,
                 testbed,
                 velopt,
                 t3=None,
                 video_save: bool =False,
                 debug: bool=False,
                 adaptive_velocity: bool =True,
                 const_velocity: float or None =None):

        """
        @param testbed: Testbed object
        @param velopt: OnlineVelocityOptimizer object
        @param t3: T3pro object
        @param video_save: Save video of experiment
        @param debug: Print debug info
        @param adaptive_velocity: Use adaptive velocity
        @param const_velocity: if not using adaptive velocity, constant velocity setpoint [mm/s]
        """
        # assert testbed is not None, "Testbed object cannot be None"
        assert velopt is not None, "Velocity Optimizer object cannot be None"
        assert const_velocity is None or adaptive_velocity is False, "Cannot have both adaptive and constant velocity"

        self.testbed = testbed
        self.date = datetime.datetime.now()
        self.debug = False
        self.thermal_px_per_mm = None  # px/mm
        self.adaptive_velocity = adaptive_velocity
        self.const_velocity = const_velocity

        self.vel_opt = velopt

        if video_save:
            self.video_save = cv.VideoWriter(f"logs/output_{self.date.strftime('%Y-%m-%d-%H:%M')}.avi",
                                             cv.VideoWriter.fourcc(*'XVID'),
                                             30,
                                             (384, 288))
        else:
            self.video_save = None

        self.experiment_started = False
        self.data_log = LoggingData()
        self.xs = []
        logging.basicConfig(filename=f"logs/log_{self.date.strftime('%Y-%m-%d-%H:%M')}.log",
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        self.logger = logging.getLogger("main")
        self.debug = debug
        try:
            with open("thermal_calibration.pkl", "rb") as f:
                self.mtx, self.dist = pkl.load(f)
        except FileNotFoundError:
            self.mtx = None
            self.dist = None

        self.t3 = t3

        self.deflection_kf = KalmanFilter(dim_x=2, dim_z=1)
        self.deflection = 0
        self.init_deflection_kf(0.5, 0.8)

    def __del__(self):
        self.testbed.stop()

    def init_deflection_kf(self, k, b):
        self.deflection_kf.x = np.array([0, 0])
        self.deflection_kf.F = np.eye(2) + np.array([[0, 1], [-k, -b]])
        self.deflection_kf.H = np.array([[1, 0]])
        self.deflection_kf.P *= 1000
        self.deflection_kf.R *= 5
        self.deflection_kf.Q = np.eye(2)

    def set_speed(self, speed):
        return self.testbed.set_speed(speed)

    def _prepare_experiment(self):
        assert self.thermal_px_per_mm is not None, "Thermal px per mm not set"
        home_input = input("Press Enter to home the testbed or 's' to skip: ")
        if home_input != 's':
            print("Homing testbed...")
            self.testbed.home()
            print("Testbed homed.")
        else:
            print("Skipping homing.")

        ret, thermal_arr, raw_frame, _ = self.get_t3_frame()
        color_frame = thermal_frame_to_color(thermal_arr)

        start_input = input("Ensure tool is on and touching tissue. Press Enter to start the experiment or 'q' to quit: ")
        if start_input == 'q':
            print("Quitting...")
            self.testbed.stop()
            exit(0)

        self.experiment_started = True

    def send_thermal_frame_to_velopt(self, v, thermal_frame):
        """
        @param v: Current velocity
        @param thermal_frame: Thermal frame
        @return: new_v, ellipse
        """
        self.deflection_kf.predict()
        meas, ddeflection = self.vel_opt.update_tool_deflection(thermal_frame)
        self.deflection_kf.update(meas/self.thermal_px_per_mm)
        self.deflection = self.deflection_kf.x[0]
        return self.vel_opt.update_velocity(v, thermal_frame, deflection=self.deflection)

    @property
    def thermal_deflection(self):
        return self.deflection

    @property
    def thermal_tool_tip_estimate(self):
        return self.vel_opt.tool_tip_pos

    def get_t3_frame(self):
        """
        @return: ret, thermal_arr, raw_frame, info
        """
        if self.t3 is None:
            return False, None, None
        ret, raw_frame = self.t3.read()
        info, lut = self.t3.info()
        thermal_arr = lut[raw_frame]
        if not ret:
            return ret, None, None, None
        # if self.debug:
        #     self.logger.debug(
        #         f"Max Temp (C): {info.Tmax_C}, Min Temp (C): {info.Tmin_C}, Max Temp Location: {info.Tmax_point}, Min Temp Location: {info.Tmin_point}")
        return ret, thermal_arr, raw_frame, info

    @staticmethod
    def plot_histogram(thermal_arr, info):
        norm_thermal_arr = cv.normalize(thermal_arr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        hist = cv.calcHist([norm_thermal_arr], [0], None, [256], [0, 256])
        bins = np.arange(info.Tmax_C, info.Tmin_C, (info.Tmax_C - info.Tmin_C) / 256)
        plt.plot(bins, hist)
        plt.pause(0.01)
        plt.cla()


    def _end_experiment(self):
        self.testbed.stop()
        experiment_type = "adaptive" if self.adaptive_velocity else f"{self.const_velocity}mm-s"
        self.data_log.save(f"logs/data_{experiment_type}_{self.date.strftime('%Y-%m-%d-%H:%M')}.pkl")
        if self.video_save is not None:
            self.video_save.release()
        self.t3.release()
        cv.destroyAllWindows()


    def add_to_data_log(self, thermal_arr):
        self.data_log.widths.append(self.vel_opt.width / self.thermal_px_per_mm)
        if self.adaptive_velocity:
            self.data_log.velocities.append(self.vel_opt.controller_v)
        else:
            self.data_log.velocities.append(self.const_velocity)
        self.data_log.deflections.append(self.deflection)
        self.data_log.thermal_frames.append(thermal_arr)
        self.data_log.damping_estimates.append(self.vel_opt.thermal_controller.d_hat)
        # self.data_log.width_constant_estimates.append(self.vel_opt.thermal_controller.width_constant_estimate)

        self.data_log.a_hats.append(self.vel_opt.thermal_controller.a_hat)
        self.data_log.b_hats.append(self.vel_opt.thermal_controller.b_hat)
        self.data_log.width_estimates.append(self.vel_opt.thermal_controller.width_estimate)

    def save_video_frame(self, color_frame):
        self.video_save.write(color_frame)

    @staticmethod
    def draw_info_on_frame(frame, ellipse, deflection, width, velocity, tool_tip_pos):
        cv.ellipse(frame, ellipse, (0, 255, 0), 2)
        cv.putText(frame, f"Deflection: {deflection:.2f} mm",
                   (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(frame, f"Width: {width:.2f} mm",
                   (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(frame, f"Velocity: {velocity:.2f} mm/s",
                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if tool_tip_pos is not None:
            cv.circle(frame, (int(tool_tip_pos[1]), int(tool_tip_pos[0])), 3, (0, 255, 0), -1)
        return frame

    def run_experiment(self):
        """
        Run the testbed, collect thermal data and control the testbed based on the thermal data. Draws the data
        on the screen.
        """
        self._prepare_experiment()
        print("Starting experiment...")
        while True:
            ret, thermal_arr, raw_frame, _ = self.get_t3_frame()
            color_frame = thermal_frame_to_color(thermal_arr)
            if self.video_save is not None:
                self.save_video_frame(color_frame)

            try:
                _, ellipse = self.send_thermal_frame_to_velopt(self.vel_opt.controller_v, thermal_arr)
            except LinAlgError as e:
                print("LinAlgError:", e)
                break
            if width := (self.vel_opt.width / self.thermal_px_per_mm) < 0.1:
                cv.putText(color_frame, "Warning: Tool may not be touching tissue!", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                self.vel_opt.reset_tool_deflection()
            elif width > 10:
                cv.putText(color_frame, "Warning: Width is greater than 10 mm, excess tissue damage may occur.", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if self.adaptive_velocity:
                ret = self.set_speed(self.vel_opt.controller_v)
                if self.vel_opt.controller_v < 1.5:
                    self.vel_opt.reset_tool_deflection()
            else:
                ret = self.set_speed(self.const_velocity)
            if self.debug:
                self.logger.debug(self.vel_opt.get_loggable_data())

            self.add_to_data_log(thermal_arr)

            if self.debug:
                # print(f"v: {self.qs.v:.2f} mm/s, Width: {self.qs.width / self.thermal_px_per_mm:.2f} mm, Deflection: {self.qs.deflection / self.thermal_px_per_mm:.2f} mm")
                if self.adaptive_velocity:
                    self.logger.debug(f"Velocity: {self.vel_opt.controller_v:.2f} mm/s")
                else:
                    self.logger.debug(f"Velocity: {self.const_velocity} mm/s")
            # print(f"Tool pos: {self.vel_opt.tool_tip_pos}")
            color_frame = self.draw_info_on_frame(color_frame,
                                                  ellipse,
                                                  self.deflection,
                                                  self.vel_opt.width / self.thermal_px_per_mm,
                                                  self.vel_opt.controller_v,
                                                  self.vel_opt.tool_tip_pos)
            cv.imshow("Thermal Camera", color_frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                self.testbed.stop()
                break

            pos = self.testbed.get_position()
            self.data_log.positions.append(pos)
            if not ret or pos == -1:
                print(ret, pos)
                break

        print(f"Avg Width: {np.mean(self.data_log.widths):.2f} mm")
        print(f"Avg Velocity: {np.mean(self.data_log.velocities):.2f} mm/s")
        self._end_experiment()

    def plot(self):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        axs[0].plot(self.data_log.widths)
        axs[0].set_title("Width vs Time")
        axs[0].set_xlabel("Time (samples)")
        axs[0].set_ylabel("Width (mm)")
        axs[1].plot(self.data_log.velocities)
        axs[1].set_title("Velocity vs Time")
        axs[1].set_xlabel("Time (samples)")
        axs[1].set_ylabel("Velocity (mm/s)")
        axs[2].plot(self.data_log.deflections)
        axs[2].set_title("Deflection vs Time")
        axs[2].set_ylabel("Deflection (mm)")

        fig, axs = plt.subplots(nrows=4, ncols=1)
        axs[0].plot(self.data_log.damping_estimates)
        axs[0].set_ylabel("Damping Estimate")
        axs[1].plot(self.data_log.a_hats)
        axs[1].set_title("a_hat")
        axs[1].set_ylabel("a_hat")
        axs[2].plot(self.data_log.b_hats)
        axs[2].set_title("b_hat")
        axs[2].set_ylabel("b_hat")
        axs[3].plot(self.data_log.width_estimates)
        axs[3].set_title("Width Estimates")
        axs[3].set_ylabel("Width Estimate (mm)")

        plt.show()

class VirtualExperimentManager(ExperimentManager):
    def __init__(self, data_save: LoggingData, velopt, debug: bool=False, adaptive_velocity: bool =True, const_velocity: float or None =None):
        super().__init__(None, velopt, None, False, debug, adaptive_velocity, const_velocity)
        self.data_save: LoggingData = data_save
        self.a_hats = [self.vel_opt.thermal_controller.a_hat]
        self.b_hats = [self.vel_opt.thermal_controller.b_hat]
        self.d_hats = [self.vel_opt.thermal_controller.d_hat]
        self.width_predictions = [self.vel_opt.thermal_controller.width_estimate]
        self.optimized_vels = []

    def run_experiment(self):
        for vel, width, deflection, thermal_frame, pos, damping_estimate in self.data_save:
            self.vel_opt.thermal_controller.update(vel, deflection, width)
            self.width_predictions.append(self.vel_opt.thermal_controller.width_estimate)
            self.optimized_vels.append(self.vel_opt.thermal_controller.v)
            self.a_hats.append(self.vel_opt.thermal_controller.a_hat)
            self.b_hats.append(self.vel_opt.thermal_controller.b_hat)
            self.d_hats.append(self.vel_opt.thermal_controller.d_hat)

    def plot(self):
        fig, axs = plt.subplots(nrows=3, ncols=1)
        axs[0].plot(self.data_save.velocities)
        axs[0].plot(self.optimized_vels)
        axs[0].legend(["Actual Velocity", "Optimized Velocity"])
        axs[0].set_title("Velocity vs Time")
        axs[0].set_ylabel("Velocity (mm/s)")
        axs[1].plot(self.data_save.widths)
        axs[1].set_title("Width vs Time")
        axs[1].set_ylabel("Width (mm)")
        axs[2].plot(self.data_save.deflections)
        axs[2].set_title("Deflection vs Time")
        axs[2].set_ylabel("Deflection (mm)")
        fig, axs = plt.subplots(nrows=3, ncols=1)
        axs[0].plot(self.data_save.damping_estimates)
        axs[0].set_title("Damping Estimates vs Time")
        axs[0].set_ylabel("Damping Estimate")
        deflection_pred = np.array(self.data_save.damping_estimates) * np.array(self.data_save.velocities)
        axs[1].plot(deflection_pred)
        axs[1].set_title("Deflection Prediction vs Time")
        axs[1].set_ylabel("Deflection Prediction")
        axs[2].plot(deflection_pred - self.data_save.deflections)
        axs[2].set_title("Deflection Error vs Time")
        axs[2].set_ylabel("Deflection")
        fig, axs = plt.subplots(nrows=4, ncols=1)
        axs[0].plot(self.a_hats)
        axs[0].set_title("a")
        axs[1].plot(self.b_hats)
        axs[1].set_title("b")
        axs[2].plot(self.d_hats)
        axs[2].set_title("d")
        axs[3].plot(self.width_predictions)
        axs[3].set_title("Width Predictions vs Time")
        axs[3].set_ylabel("Width Prediction")

        # fig, ax = plt.subplots()
        # ax.plot(1/np.array(self.data_save.velocities))
        # ax.plot(np.array(self.data_save.widths))
        # ax.plot(np.array(self.data_save.velocities) * np.array(self.data_save.widths))
        # plt.legend(["1/Velocity", "Width", "Velocity * Width"])
        plt.show()

    def __del__(self):
        pass