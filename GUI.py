import tkinter as tk
from tkinter import filedialog, messagebox
import serial.serialutil
from control_experiment import ExperimentType, main, RunConfig
from models import PseudoStaticModel, humanTissue, hydrogelPhantom, SteadyStateMinimizationModel
import do_mpc
from T3pro import T3pro
from testbed import Testbed
import TKinterModernThemes as TKMT

class ControlExperimentUI(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__("Control Experiment", "azure", "light")
        self.init_vars()
        self.create_widgets()
        self.run()

    def init_vars(self):
        self.exp_type = tk.StringVar(value=ExperimentType.REAL)
        self.constant_velocity = tk.DoubleVar(value=7)
        self.material_name = tk.StringVar(value="hydrogel")
        self.model_name = tk.StringVar(value="minimization")
        self.n_horizons = tk.IntVar(value=10)
        self.qd = tk.IntVar(value=1)
        self.qw = tk.IntVar(value=1)
        self.r = tk.DoubleVar(value=0.1)
        self.save_dir = tk.StringVar(value="./logs/")
        self.logFile = tk.StringVar()
        self.home = tk.BooleanVar(value=True)
        self.plot_adaptive_params = tk.BooleanVar(value=False)


    def create_widgets(self):
        self.files_frame = self.addLabelFrame("Log Files")
        self.exp_type_frame = self.addLabelFrame("Experiment Type")
        self.model_type_frame = self.addLabelFrame("Model Type")

        self.nextCol()
        self.material_type_frame = self.addLabelFrame("Material Type")
        self.numerical_params_frame = self.addLabelFrame("MPC Settings")
        self.AccentButton("Start Experiment", command=self.start_experiment)

        self.exp_type_frame.Radiobutton("Real", variable=self.exp_type, value=ExperimentType.REAL, command=self.exp_type_selection)
        self.exp_type_frame.Radiobutton("Pre-recorded", variable=self.exp_type, value=ExperimentType.PRERECORDED, command=self.exp_type_selection)
        self.exp_type_frame.Radiobutton("Simulated", variable=self.exp_type, value=ExperimentType.SIMULATED, command=self.exp_type_selection)
        self.homing_button = self.exp_type_frame.SlideSwitch("Home", variable=self.home)
        self.plot_adaptive_params_button = self.exp_type_frame.SlideSwitch("Plot Adaptive Params", variable=self.plot_adaptive_params)

        self.model_type_frame.Radiobutton("Minimization", variable=self.model_name, value="minimization", command=self.model_selection)
        self.model_type_frame.Radiobutton("MPC", variable=self.model_name, value="pseudostatic", command=self.model_selection)
        self.model_type_frame.Radiobutton("Constant Velocity", variable=self.model_name, value="constant_velocity", command=self.model_selection)

        self.material_type_frame.Radiobutton("Human Tissue", variable=self.material_name, value="human")
        self.material_type_frame.Radiobutton("Hydrogel Phantom", variable=self.material_name, value="hydrogel")

        self.numerical_params_frame.Label("Number of Horizons")
        self.n_horizons_entry = self.numerical_params_frame.Entry(textvariable=self.n_horizons)
        self.numerical_params_frame.Label("Qd")
        self.qd_entry = self.numerical_params_frame.Entry(textvariable=self.qd)
        self.numerical_params_frame.Label("Qw")
        self.qw_entry = self.numerical_params_frame.Entry(textvariable=self.qw)
        self.numerical_params_frame.Label("R")
        self.r_entry = self.numerical_params_frame.Entry(textvariable=self.r)
        self.numerical_params_frame.Label("Constant Velocity")
        self.constant_velocity_entry = self.numerical_params_frame.Entry(textvariable=self.constant_velocity)
        self.constant_velocity_entry.config(state="disabled")

        self.save_indicator = self.files_frame.Button("Log Save Directory", command=self.folderDialog)
        self.load_indicator = self.files_frame.Button("Load Log File", command=self.loadLogDialog)

    def model_selection(self):
        model_name = self.model_name.get()
        if model_name == "minimization":
            self.set_model_params(0, "disabled", "disabled", "normal", "normal", "normal")
        elif model_name == "pseudostatic":
            self.set_model_params(10, "normal", "disabled", "normal", "normal", "normal")
        elif model_name == "constant_velocity":
            self.set_model_params(None, "disabled", "normal", "disabled", "disabled", "disabled")

    def set_model_params(self, n_horizons, n_horizons_state, constant_velocity_state, r_state, qd_state, qw_state):
        if n_horizons is not None:
            self.n_horizons.set(n_horizons)
        self.n_horizons_entry.config(state=n_horizons_state)
        self.constant_velocity_entry.config(state=constant_velocity_state)
        self.r_entry.config(state=r_state)
        self.qd_entry.config(state=qd_state)
        self.qw_entry.config(state=qw_state)

    def exp_type_selection(self):
        exp_type = self.exp_type.get()
        if exp_type == ExperimentType.REAL:
            self.set_exp_type_params("normal", "disabled", "normal")
        elif exp_type == ExperimentType.PRERECORDED:
            self.set_exp_type_params("disabled", "normal", "disabled")
            self.home.set(False)
        elif exp_type == ExperimentType.SIMULATED:
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
        model, adaptive_velocity, constant_velocity = self.get_model(material)
        mpc = self.setup_mpc(model) if self.model_name.get() == "pseudostatic" else None
        t3, tb = self.setup_hardware()

        if not self.validate_experiment(t3, tb):
            return

        run_conf = RunConfig(
            exp_type=ExperimentType(self.exp_type.get()),
            adaptive_velocity=adaptive_velocity,
            constant_velocity=constant_velocity,
            log_save_dir=self.save_dir.get(),
            log_file_to_load=self.logFile.get(),
            mpc=mpc is not None,
            home=self.home.get(),
            plot_adaptive_params=self.plot_adaptive_params.get(),
            material=material
        )

        main(model, run_conf, mpc, t3, tb, self.r.get())

    def get_model(self, material):
        model_name = self.model_name.get()
        if model_name == "minimization":
            model = SteadyStateMinimizationModel(qw=self.qw.get(), qd=self.qd.get())
            return model, True, None
        elif model_name == "pseudostatic":
            model = PseudoStaticModel(material)
            return model, True, None
        elif model_name == "constant_velocity":
            model = SteadyStateMinimizationModel()
            return model, False, float(self.constant_velocity.get())

    def setup_mpc(self, model):
        model.set_cost_function(self.qd.get(), self.qw.get())
        model.setup()
        mpc = do_mpc.controller.MPC(model=model)
        mpc.settings.n_horizon = self.n_horizons.get()
        mpc.settings.n_robust = 0
        mpc.settings.open_loop = 0
        mpc.settings.t_step = 1 / 24
        mpc.settings.state_discretization = 'collocation'
        mpc.settings.collocation_type = 'radau'
        mpc.settings.collocation_deg = 2
        mpc.settings.collocation_ni = 2
        mpc.settings.store_full_solution = True
        mpc.settings.nlpsol_opts = {'ipopt.linear_solver': 'MA97'}
        return mpc

    def setup_hardware(self):
        t3, tb = None, None
        if self.exp_type.get() == ExperimentType.REAL:
            try:
                t3 = T3pro(port=3)
            except serial.serialutil.SerialException:
                messagebox.showerror("Error", "No T3pro found")
            try:
                tb = Testbed()
            except serial.serialutil.SerialException:
                messagebox.showerror("Error", "No Testbed found")
        return t3, tb

    def validate_experiment(self, t3, tb):
        exp_type = self.exp_type.get()
        if exp_type == ExperimentType.PRERECORDED and not self.logFile.get():
            messagebox.showerror("Warning", "No log file selected")
            return False
        if exp_type == ExperimentType.SIMULATED and self.model_name.get() == "minimization":
            messagebox.showerror("Warning", "Minimization model cannot be simulated")
            return False
        if exp_type == ExperimentType.REAL and (not t3 or not tb):
            return False
        if not self.save_dir.get() and exp_type == ExperimentType.REAL:
            messagebox.showinfo("Warning", "No save directory selected, data will not be saved")
        return True

if __name__ == "__main__":
    app = ControlExperimentUI()
