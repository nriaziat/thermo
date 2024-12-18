import tkinter as tk
from tkinter import filedialog, messagebox
import serial.serialutil
from control_experiment import ExperimentType, main, RunConfig, Devices, ControlMode
from models import PseudoStaticModel, humanTissue, hydrogelPhantom, SteadyStateMinimizationModel, ElectrosurgeryMPCModel
import do_mpc
from T3pro import T3pro
from testbed import Testbed
import TKinterModernThemes as TKMT
import pygame

class ControlExperimentUI(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__("Control Experiment", "azure", "light")
        pygame.init()
        self.init_vars()
        self.create_widgets()
        self.run()

    def init_vars(self):
        self.exp_type = tk.Variable(value=ExperimentType.REAL)
        self.constant_velocity = tk.DoubleVar(value=7)
        self.material_name = tk.StringVar(value="hydrogel")
        self.control_mode = tk.Variable(value=ControlMode.AUTONOMOUS)
        # self.n_horizons = tk.IntVar(value=10)
        self.qd = tk.IntVar(value=1)
        self.qw = tk.IntVar(value=1)
        self.r = tk.DoubleVar(value=0.005)
        self.save_dir = tk.StringVar(value="./logs/")
        self.logFile = tk.StringVar()
        self.home = tk.BooleanVar(value=True)
        self.plot_adaptive_params = tk.BooleanVar(value=False)


    def create_widgets(self):
        self.files_frame = self.addLabelFrame("Log Files")
        self.exp_type_frame = self.addLabelFrame("Experiment Type")
        self.control_mode_frame = self.addLabelFrame("Model Type")

        self.nextCol()
        self.material_type_frame = self.addLabelFrame("Material Type")
        self.numerical_params_frame = self.addLabelFrame("MPC Settings")
        self.connected_devices_frame = self.addLabelFrame("Connected Devices")
        self.AccentButton("Start Experiment", command=self.start_experiment)

        self.exp_type_frame.Radiobutton("Real", variable=self.exp_type, value=ExperimentType.REAL, command=self.exp_type_selection)
        self.exp_type_frame.Radiobutton("Pre-recorded", variable=self.exp_type, value=ExperimentType.PRERECORDED, command=self.exp_type_selection)
        self.exp_type_frame.Radiobutton("Simulated", variable=self.exp_type, value=ExperimentType.SIMULATED, command=self.exp_type_selection)
        self.homing_button = self.exp_type_frame.SlideSwitch("Home", variable=self.home)
        self.plot_adaptive_params_button = self.exp_type_frame.SlideSwitch("Plot Adaptive Params", variable=self.plot_adaptive_params)

        self.control_mode_frame.Radiobutton("Autonomous", variable=self.control_mode, value=ControlMode.AUTONOMOUS, command=self.control_mode_selection)
        # self.model_type_frame.Radiobutton("MPC", variable=self.model_name, value="pseudostatic", command=self.model_selection)
        self.control_mode_frame.Radiobutton("Constant Velocity", variable=self.control_mode, value=ControlMode.CONSTANT_VELOCITY, command=self.control_mode_selection)
        self.control_mode_frame.Radiobutton("Shared Control", variable=self.control_mode, value=ControlMode.SHARED_CONTROL, command=self.control_mode_selection)
        self.control_mode_frame.Radiobutton("Joystick Control", variable=self.control_mode, value=ControlMode.TELEOPERATED, command=self.control_mode_selection)

        self.material_type_frame.Radiobutton("Human Tissue", variable=self.material_name, value="human")
        self.material_type_frame.Radiobutton("Hydrogel Phantom", variable=self.material_name, value="hydrogel")

        # self.numerical_params_frame.Label("Number of Horizons")
        # self.n_horizons_entry = self.numerical_params_frame.Entry(textvariable=self.n_horizons)
        self.numerical_params_frame.Label("Qd")
        self.qd_entry = self.numerical_params_frame.Entry(textvariable=self.qd)
        self.numerical_params_frame.Label("Qw")
        self.qw_entry = self.numerical_params_frame.Entry(textvariable=self.qw)
        self.numerical_params_frame.Label("R")
        self.r_entry = self.numerical_params_frame.Entry(textvariable=self.r)
        self.numerical_params_frame.Label("Constant Velocity")
        self.constant_velocity_entry = self.numerical_params_frame.Entry(textvariable=self.constant_velocity)
        self.constant_velocity_entry.config(state="disabled")

        try:
            with T3pro(port=2):
                self.connected_devices_frame.Label("T3pro connected")
        except serial.serialutil.SerialException:
            self.connected_devices_frame.Label("T3pro not connected")
        try:
            with Testbed():
                self.connected_devices_frame.Label("Testbed connected")
        except serial.serialutil.SerialException:
            self.connected_devices_frame.Label("Testbed not connected")
        try:
            joy = pygame.joystick.Joystick(0)
            self.connected_devices_frame.Label("Joystick connected")
            joy.quit()
        except pygame.error:
            self.connected_devices_frame.Label("Joystick not connected")

        self.save_indicator = self.files_frame.Button("Log Save Directory", command=self.folderDialog)
        self.load_indicator = self.files_frame.Button("Load Log File", command=self.loadLogDialog)

    def control_mode_selection(self):
        control_mode = ControlMode(self.control_mode.get())
        self.set_control_mode_params("normal" if control_mode is ControlMode.CONSTANT_VELOCITY else "disabled",
                                      "normal" if control_mode in [ControlMode.AUTONOMOUS, ControlMode.SHARED_CONTROL] else "disabled",
                                      "normal" if control_mode in [ControlMode.AUTONOMOUS, ControlMode.SHARED_CONTROL] else "disabled",
                                      "normal" if control_mode in [ControlMode.AUTONOMOUS, ControlMode.SHARED_CONTROL] else "disabled")

    def set_control_mode_params(self, constant_velocity_state, r_state, qd_state, qw_state):
        # if n_horizons is not None:
        #     self.n_horizons.set(n_horizons)
        # self.n_horizons_entry.config(state=n_horizons_state)
        self.constant_velocity_entry.config(state=constant_velocity_state)
        self.r_entry.config(state=r_state)
        self.qd_entry.config(state=qd_state)
        self.qw_entry.config(state=qw_state)

    def exp_type_selection(self):
        exp_type = ExperimentType(self.exp_type.get())
        if exp_type is ExperimentType.REAL:
            self.set_exp_type_params("normal", "disabled", "normal")
        elif exp_type is ExperimentType.PRERECORDED:
            self.set_exp_type_params("disabled", "normal", "disabled")
            self.home.set(False)
        elif exp_type is ExperimentType.SIMULATED:
            self.set_exp_type_params("disabled", "disabled", "disabled")
            self.home.set(False)

    def set_exp_type_params(self, save_indicator_state, load_indicator_state, homing_button_state):
        self.save_indicator.config(state=save_indicator_state)
        self.load_indicator.config(state=load_indicator_state)
        self.homing_button.config(state=homing_button_state)

    def folderDialog(self):
        filename = filedialog.askdirectory(initialdir="./logs/")
        if filename:
            self.save_dir.set(filename)
            self.save_indicator.config(text=f"Save to: {filename}")
            self.exp_type.set(ExperimentType.REAL)
            self.exp_type_selection()

    def loadLogDialog(self):
        filename = filedialog.askopenfilename(filetypes=[("Log Files", "*.pkl")])
        self.logFile.set(filename)
        if filename:
            self.load_indicator.config(text=f"Load from: {filename.split('/')[-1]}")
            self.exp_type.set(ExperimentType.PRERECORDED)
            self.exp_type_selection()


    def start_experiment(self):
        material = humanTissue if self.material_name.get() == "human" else hydrogelPhantom
        model, adaptive_velocity = self.get_control_mode()
        # mpc = self.setup_mpc(model) if self.model_name.get() == "pseudostatic" else None
        mpc = None
        t3, tb, joy = self.setup_hardware()

        if not self.validate_experiment(t3, tb):
            return

        run_conf = RunConfig(
            exp_type=ExperimentType(self.exp_type.get()),
            control_mode=ControlMode(self.control_mode.get()),
            adaptive_velocity=isinstance(adaptive_velocity, bool),
            constant_velocity=adaptive_velocity if isinstance(adaptive_velocity, float) else None,
            log_save_dir=self.save_dir.get(),
            log_file_to_load=self.logFile.get(),
            mpc=mpc is not None,
            home=self.home.get(),
            plot_adaptive_params=self.plot_adaptive_params.get(),
            material=material
        )

        devices = Devices(t3, tb, joy)
        main(model, run_conf, devices, mpc)

    def get_control_mode(self) -> tuple[SteadyStateMinimizationModel | ElectrosurgeryMPCModel, float | bool]:
        control_mode = ControlMode(self.control_mode.get())
        if control_mode in [ControlMode.AUTONOMOUS, ControlMode.SHARED_CONTROL, ControlMode.TELEOPERATED]:
            model = SteadyStateMinimizationModel(qw=self.qw.get(), qd=self.qd.get(), r=self.r.get())
            return model, True
        # elif control_mode == "pseudostatic":
        #     model = PseudoStaticModel(material)
        #     return model, True
        elif control_mode is ControlMode.CONSTANT_VELOCITY:
            model = SteadyStateMinimizationModel()
            return model, float(self.constant_velocity.get())
        else:
            raise ValueError(f"Invalid control mode: {control_mode}")

    # def setup_mpc(self, model):
    #     model.set_cost_function(self.qd.get(), self.qw.get())
    #     model.setup()
    #     mpc = do_mpc.controller.MPC(model=model)
    #     mpc.settings.n_horizon = self.n_horizons.get()
    #     mpc.settings.n_robust = 0
    #     mpc.settings.open_loop = 0
    #     mpc.settings.t_step = 1 / 24
    #     mpc.settings.state_discretization = 'collocation'
    #     mpc.settings.collocation_type = 'radau'
    #     mpc.settings.collocation_deg = 2
    #     mpc.settings.collocation_ni = 2
    #     mpc.settings.store_full_solution = True
    #     mpc.settings.nlpsol_opts = {'ipopt.linear_solver': 'MA97'}
    #     return mpc

    def setup_hardware(self):
        t3, tb, joy = None, None, None
        exp_type = ExperimentType(self.exp_type.get())
        if exp_type is ExperimentType.REAL:
            try:
                t3 = T3pro(port=4)
            except serial.serialutil.SerialException:
                messagebox.showerror("Error", "No T3pro found")
            try:
                tb = Testbed()
            except serial.serialutil.SerialException:
                messagebox.showerror("Error", "No Testbed found")
            if ControlMode(self.control_mode.get()) in [ControlMode.TELEOPERATED, ControlMode.SHARED_CONTROL]:
                try:
                    joy = pygame.joystick.Joystick(0)
                except pygame.error:
                    messagebox.showerror("Error", "No joystick found")
        return t3, tb, joy

    def validate_experiment(self, t3, tb):
        exp_type = ExperimentType(self.exp_type.get())
        if exp_type is ExperimentType.PRERECORDED and not self.logFile.get():
            messagebox.showerror("Warning", "No log file selected")
            return False
        if exp_type is ExperimentType.SIMULATED and self.control_mode.get() == ControlMode.AUTONOMOUS:
            messagebox.showerror("Warning", "Minimization model cannot be simulated")
            return False
        if exp_type is ExperimentType.REAL and (not t3 or not tb):
            return False
        if not self.save_dir.get() and exp_type is ExperimentType.REAL:
            messagebox.showinfo("Warning", "No save directory selected, data will not be saved")
        return True

if __name__ == "__main__":
    app = ControlExperimentUI()
