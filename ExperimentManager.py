from typing import Optional
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import logging
import datetime
import pickle as pkl
from matplotlib import rcParams
from utils import thermal_frame_to_color, Plotter, cv_isotherm_width, draw_info_on_frame
from models import ElectrosurgeryMPCModel, ToolTipKF
from AdaptiveID import ScalarLinearAlgabraicAdaptation, ScalarFirstOrderAdaptation
import do_mpc

class ExperimentManager:
    qw = 1  # width cost
    qd = 1 # deflection cost
    r = 0.01  # control change cost

    setup_mpc = {
        'n_horizon': 10,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 1 / 24,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True,
        # Use MA57 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'MA57'}
    }

    def __init__(self,
                 testbed,
                 model: ElectrosurgeryMPCModel,
                 adaptive_deflection_model: Optional[ScalarLinearAlgabraicAdaptation] = None,
                 adaptive_thermal_model: Optional[ScalarFirstOrderAdaptation] = None,
                 thermal_camera=None,
                 video_save: bool =False,
                 debug: bool=False,
                 adaptive_velocity: bool =True,
                 const_velocity: float or None =None,
                 v_bounds: tuple[float, float] = (1, 10)):

        """
        @param testbed: Testbed object
        @param model: ElectrosurgeryModel object
        @param adaptive_deflection_model: Optional adaptive deflection model
        @param adaptive_thermal_model: Optional adaptive thermal model
        @param thermal_camera: thermal camera object, like T3pro
        @param video_save: Save video of experiment
        @param debug: Print debug info
        @param adaptive_velocity: Use adaptive velocity
        @param const_velocity: if not using adaptive velocity, constant velocity setpoint [mm/s]
        @param v_bounds: velocity bounds [min, max], [mm/s]
        """
        assert const_velocity is None or adaptive_velocity is False, "Cannot have both adaptive and constant velocity"
        assert len(v_bounds) == 2, "v_bounds must be a tuple of length 2"
        assert all(v > 0 for v in v_bounds), "v_bounds must be positive"

        plt.ion()
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'
        rcParams['axes.grid'] = True
        rcParams['lines.linewidth'] = 2.0
        rcParams['axes.labelsize'] = 'xx-large'
        rcParams['xtick.labelsize'] = 'xx-large'
        rcParams['ytick.labelsize'] = 'xx-large'

        self.testbed = testbed
        self.date = datetime.datetime.now()
        self.debug = False
        self.thermal_px_per_mm = None  # px/mm
        self.adaptive_velocity = adaptive_velocity
        self.const_velocity = const_velocity

        self.model = model
        self.model.set_cost_function(self.qw, self.qd)
        self.model.setup()
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**self.setup_mpc)
        self.mpc.set_objective(mterm=self.model.aux['cost'], lterm=self.model.aux['cost'])
        self.mpc.set_rterm(u=self.r)

        self.v_min, self.v_max = v_bounds
        self.mpc.bounds['lower', '_u', 'u'] = self.v_min
        self.mpc.bounds['upper', '_u', 'u'] = self.v_max
        for i in range(model.n_isotherms):
            self.mpc.scaling['_x', f'x_{i}'] = 1
            self.mpc.bounds['lower', '_x', f'x_{i}'] = 1e-1
        self.mpc.scaling['_z', 'deflection'] = 0.1
        self.mpc.bounds['lower', '_z', 'deflection'] = 0

        self._tool_tip_kf = ToolTipKF(0.7)
        self._adaptive_deflection_model = adaptive_deflection_model
        self._adaptive_thermal_model = adaptive_thermal_model

        self.plotter = Plotter(self.mpc.data)
        if video_save:
            self.video_save = cv.VideoWriter(f"logs/output_{self.date.strftime('%Y-%m-%d-%H:%M')}.avi",
                                             cv.VideoWriter.fourcc(*'XVID'),
                                             30,
                                             (384, 288))
        else:
            self.video_save = None

        self.experiment_started = False
        self.data_log = dict()
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

        self.t3 = thermal_camera
        self.deflection_mm = 0
        self.widths_mm = []

    def __del__(self):
        self.testbed.stop()
        plt.ioff()

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
        thermal_frame_to_color(thermal_arr)

        start_input = input("Ensure tool is on and touching tissue. Press Enter to start the experiment or 'q' to quit: ")
        if start_input == 'q':
            print("Quitting...")
            self.testbed.stop()
            exit(0)

        self.experiment_started = True

    def update_measurements(self, thermal_frame):
        """
        @param v: Current velocity
        @param thermal_frame: Thermal frame
        @return: new_v
        """

        meas_px, _ = self._tool_tip_kf.update_with_measurement(thermal_frame)
        self.deflection_mm = meas_px / self.thermal_px_per_mm
        self.widths_mm = []
        for isotherm_temp in self.model.isotherm_temps:
            self.widths_mm.append(cv_isotherm_width(thermal_frame, isotherm_temp)[0] / self.thermal_px_per_mm)

        measurement = np.array([self.widths_mm, self.deflection_mm])
        v = self.mpc.make_step(measurement)
        return v

    @property
    def thermal_deflection(self):
        return self.deflection_mm

    @property
    def thermal_tool_tip_estimate(self):
        return self._tool_tip_kf.x[0], self._tool_tip_kf.x[2]

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
        return ret, thermal_arr, raw_frame, info

    def _end_experiment(self):
        self.testbed.stop()
        experiment_type = "adaptive" if self.adaptive_velocity else f"{self.const_velocity}mm-s"
        with open(f"logs/data_{experiment_type}_{self.date.strftime('%Y-%m-%d-%H:%M')}.pkl", "wb") as f:
            pkl.dump(self.data_log, f)
        if self.video_save is not None:
            self.video_save.release()
        self.t3.release()
        cv.destroyAllWindows()

    def add_to_data_log(self, thermal_arr):
        self.data_log['widths_mm'].append(self.widths_mm)
        if self.adaptive_velocity:
            self.data_log['velocities'].append(self.mpc.u0)
        else:
            self.data_log['velocities'].append(self.const_velocity)
        self.data_log['deflections_mm'].append(self.deflection_mm)
        self.data_log['thermal_frames'].append(thermal_arr)
        self.data_log['damping_estimates'].append(self._adaptive_deflection_model.b)
        self.data_log['a_hats'].append(self._adaptive_thermal_model.a)
        self.data_log['alpha_hats'].append(self._adaptive_thermal_model.b)
        self.data_log['width_estimates'].append(self._adaptive_thermal_model.state_estimate)

    def save_video_frame(self, color_frame):
        self.video_save.write(color_frame)

    def run_experiment(self):
        """
        Run the testbed, collect thermal data and control the testbed based on the thermal data. Draws the data
        on the screen.
        """
        self._prepare_experiment()
        print("Starting experiment...")
        n_loops = 0
        try:
            while True:
                ret, thermal_arr, raw_frame, _ = self.get_t3_frame()
                color_frame = thermal_frame_to_color(thermal_arr)

                self.update_measurements(thermal_arr)

                if all(w < 0.1 for w in self.widths_mm):
                    # cv.putText(color_frame, "Warning: Tool may not be touching tissue!", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    self._tool_tip_kf._init = False
                elif all(w > 10 for w in self.widths_mm):
                    print(f"All isotherms greater than 10 mm, quitting...")
                    break
                    # cv.putText(color_frame, "Warning: Width is greater than 10 mm, excess tissue damage may occur.", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                elif self.adaptive_velocity:
                    if self.deflection_mm > (d_lim := 15):
                        # print(f"Deflection greater than {d_lim} mm, moving backwards...")
                        v = np.clip(self.mpc.controller_v_mm_s - self.v_max / d_lim * self.deflection_mm, -self.v_max, self.v_max)
                        self.mpc.v = v
                        ret = self.testbed.set_speed(v)
                    else:
                        ret = self.testbed.set_speed(self.mpc.controller_v_mm_s)
                else:
                    if self.deflection_mm > (d_lim := 15):
                        print(f"Deflection greater than {d_lim} mm, quitting...")
                        ret = self.testbed.set_speed(0)
                        break
                    self.mpc.v = self.const_velocity
                    ret = self.testbed.set_speed(self.const_velocity)

                self.add_to_data_log(thermal_arr)

                if self.debug:
                    if self.adaptive_velocity:
                        self.logger.debug(f"Velocity: {self.mpc.controller_v_mm_s:.2f} mm/s")
                    else:
                        self.logger.debug(f"Velocity: {self.const_velocity} mm/s")
                color_frame = draw_info_on_frame(color_frame,
                                                      self.deflection_mm,
                                                      self.width_mm,
                                                      self.mpc.controller_v_mm_s,
                                                      (self._tool_tip_kf.x[0], self._tool_tip_kf.x[2]))


                pos = self.testbed.get_position()
                self.data_log['positions_mm'].append(pos)
                self.plotter.plot()
                plt.pause(0.0001)
                # color_frame = np.zeros((288, 384, 3), dtype=np.uint8)
                cv.imshow("Thermal Camera", color_frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if not ret or pos == -1:
                    break
                n_loops += 1
        except KeyboardInterrupt:
            pass

        self.testbed.stop()
        plt.show()
        # plt.savefig(f"logs/plot_{self.date.strftime('%Y-%m-%d-%H:%M')}.png")
        self._end_experiment()


class VirtualExperimentManager(ExperimentManager):
    def __init__(self,
                 data_save: dict,
                 model: ElectrosurgeryMPCModel,
                 adaptive_deflection_model: Optional[ScalarLinearAlgabraicAdaptation] = None,
                 adaptive_thermal_model: Optional[ScalarFirstOrderAdaptation] = None,
                 debug: bool=False,
                 adaptive_velocity: bool =True,
                 const_velocity: float or None =None):
        super().__init__(None, model, adaptive_deflection_model, adaptive_thermal_model, debug, adaptive_velocity, None, const_velocity)
        self.data_save: dict = data_save

    def run_experiment(self):
        for vel, width, deflection, thermal_frame, pos, damping_estimate in    self.data_save:
            self.mpc.update_velocity(vel, thermal_frame, deflection_mm=deflection)
            self.plotter.plot()
            plt.pause(0.001)

        plt.show()

    def __del__(self):
        pass