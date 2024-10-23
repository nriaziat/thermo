import tkinter as tk
from tkinter import filedialog, messagebox
import serial.serialutil
from control_experiment import ExperimentType, main, RunningParams
from models import PseudoStaticModel, humanTissue, hydrogelPhantom, SteadyStateMinimizationModel
import do_mpc
from T3pro import T3pro
from testbed import Testbed
import TKinterModernThemes as TKMT

class ControlExperimentUI(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__("Control Experiment", "azure", "light")
        self.files_frame = self.addLabelFrame("Log Files")
        self.exp_type_frame = self.addLabelFrame("Experiment Type")
        self.model_type_frame = self.addLabelFrame("Model Type")

        self.nextCol()
        self.material_type_frame = self.addLabelFrame("Material Type")
        self.numerical_params_frame = self.addLabelFrame("MPC Settings")
        self.AccentButton("Start Experiment", command=self.start_experiment)

        self.exp_type = tk.StringVar(value=ExperimentType.REAL)
        self.adaptive_velocity = tk.StringVar(value="y")
        self.constant_velocity = tk.DoubleVar(value=7)
        self.material_name = tk.StringVar(value="human")
        self.model_name = tk.StringVar(value="pseudostatic")
        self.n_horizons = tk.IntVar(value=10)
        self.qd = tk.IntVar(value=1)
        self.qw = tk.IntVar(value=25)
        self.r = tk.DoubleVar(value=0.1)
        self.save_dir = tk.StringVar()
        self.logFile = tk.StringVar()
        self.home = tk.BooleanVar(value=True)
        self.create_widgets()
        self.run()

    def create_widgets(self):

        self.exp_type_frame.Radiobutton("Real", variable=self.exp_type, value=ExperimentType.REAL, command=self.exp_type_selection)
        self.exp_type_frame.Radiobutton("Pre-recorded", variable=self.exp_type, value=ExperimentType.PRERECORDED, command=self.exp_type_selection)
        self.exp_type_frame.Radiobutton("Simulated", variable=self.exp_type, value=ExperimentType.SIMULATED, command=self.exp_type_selection)
        self.homing_button = self.exp_type_frame.SlideSwitch("Home", variable=self.home)

        self.model_type_frame.Radiobutton("MPC", variable=self.model_name, value="pseudostatic", command=self.model_selection)
        self.model_type_frame.Radiobutton("Minimization", variable=self.model_name, value="minimization", command=self.model_selection)
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
        if self.model_name.get() == "minimization":
            self.n_horizons.set(0)
            self.n_horizons_entry.config(state="disabled")
            self.constant_velocity_entry.config(state="disabled")
            self.r_entry.config(state="normal")
            self.qd_entry.config(state="normal")
            self.qw_entry.config(state="normal")

        elif self.model_name.get() == "pseudostatic":
            self.n_horizons.set(10)
            self.n_horizons_entry.config(state="normal")
            self.constant_velocity_entry.config(state="disabled")
            self.r_entry.config(state="normal")
            self.qd_entry.config(state="normal")
            self.qw_entry.config(state="normal")

        elif self.model_name.get() == "constant_velocity":
            self.constant_velocity_entry.config(state="normal")
            self.r_entry.config(state="disabled")
            self.qd_entry.config(state="disabled")
            self.qw_entry.config(state="disabled")

    def exp_type_selection(self):
        if self.exp_type.get() == ExperimentType.REAL:
            self.save_indicator.config(state="normal")
            self.load_indicator.config(state="disabled")
            self.homing_button.config(state="normal")
        elif self.exp_type.get() == ExperimentType.PRERECORDED:
            self.save_indicator.config(state="disabled")
            self.load_indicator.config(state="normal")
            self.home.set(False)
            self.homing_button.config(state="disabled")
        elif self.exp_type.get() == ExperimentType.SIMULATED:
            self.save_indicator.config(state="disabled")
            self.load_indicator.config(state="disabled")
            self.home.set(False)
            self.homing_button.config(state="disabled")

    def folderDialog(self):
        filename = filedialog.askdirectory()
        if filename == "":
            return
        self.save_dir.set(filename)
        self.save_indicator.config(text=f"Save to: {filename}")

    def loadLogDialog(self):
        filename = filedialog.askopenfilename(filetypes=[("Log Files", "*.pkl")])
        self.logFile.set(filename)
        self.load_indicator.config(text=f"Load from: {filename.split('/')[-1]}")
        self.exp_type.set(ExperimentType.PRERECORDED)

    def start_experiment(self):

        adaptive_velocity = self.adaptive_velocity.get() == "y"
        constant_velocity = float(self.constant_velocity.get()) if not adaptive_velocity else None
        material = humanTissue if self.material_name.get() == "human" else hydrogelPhantom

        if (name := self.model_name.get()) == "minimization":
            model = SteadyStateMinimizationModel(material)
        elif name == "pseudostatic":
            model = PseudoStaticModel(material)
        elif name == "constant_velocity":
            messagebox.showerror("Constant velocity is not implemented yet.")
            return

        if name == "minimization":
            mpc = None
            model.qd = self.qd.get()
            model.qw = self.qw.get()
        else:
            model.set_cost_function(self.qd.get(), self.qw.get())
            model.setup()
            mpc = do_mpc.controller.MPC(model=model)
            ##############################
            mpc.settings.n_horizon = self.n_horizons.get()
            mpc.settings.n_robust = 0
            mpc.settings.open_loop = 0
            mpc.settings.t_step = 1 / 24
            mpc.settings.state_discretization = 'collocation'
            mpc.settings.collocation_type = 'radau'
            mpc.settings.collocation_deg = 2
            mpc.settings.collocation_ni = 2
            mpc.settings.store_full_solution = True
            mpc.settings.nlpsol_opts = {'ipopt.linear_solver': 'MA97',
                                        # 'ipopt.bound_relax_factor': 0,
                                        # 'ipopt.mu_strategy': 'adaptive',
                                        # 'ipopt.ma57_automatic_scaling': 'yes',
                                        # 'ipopt.check_derivatives_for_naninf': 'yes',
                                        # 'ipopt.expect_infeasible_problem': 'yes',
                                        # 'monitor': 'nlp_g'
                                    }
        # mpc.settings.supress_ipopt_output()
        exp_type = ExperimentType(self.exp_type.get())
        t3 = None
        tb = None

        if exp_type == ExperimentType.PRERECORDED:
            if self.logFile.get() == "":
                messagebox.showerror("Warning", "No log file selected")
                return

        elif exp_type == ExperimentType.SIMULATED:
            if self.model_name.get() == "minimization":
                messagebox.showerror("Warning", "Minimization model cannot be simulated")
                return

        # elif exp_type == ExperimentType.REAL:
        #     try:
        #         t3 = T3pro()
        #     except serial.serialutil.SerialException:
        #         messagebox.showerror("Error", "No T3pro found")
        #         return
        #     try:
        #         tb = Testbed()
        #     except serial.serialutil.SerialException:
        #         messagebox.showerror("Error", "No Testbed found")
        #         return


        if (save_dir:=self.save_dir.get()) == "" and exp_type == ExperimentType.REAL:
            messagebox.showinfo("Warning", "No save directory selected, data will not be saved")

        params = RunningParams(exp_type=exp_type,
                               adaptive_velocity=adaptive_velocity,
                               constant_velocity=constant_velocity,
                               log_save_dir=save_dir,
                               log_file_to_load=self.logFile.get(),
                               mpc=mpc is not None,
                               home=self.home.get())

        main(model, params, mpc, t3, tb, self.r.get())

if __name__ == "__main__":
    app = ControlExperimentUI()
