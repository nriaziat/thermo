import tkinter as tk
from tkinter import filedialog, messagebox
import serial.serialutil
from control_experiment import ExperimentType, main
from models import MultiIsothermMPCModel, PseudoStaticMPCModel, humanTissue, hydrogelPhantom, SteadyStateMinimizationModel
import do_mpc
from T3pro import T3pro
from testbed import Testbed
import TKinterModernThemes as TKMT

class ControlExperimentUI(TKMT.ThemedTKinterFrame):
    def __init__(self):
        super().__init__("Control Experiment", "azure", "light")
        self.exp_type_frame = self.addLabelFrame("Experiment Type")
        self.velocity_type_frame = self.addLabelFrame("Velocity Type")
        self.model_type_frame = self.addLabelFrame("Model Type")
        self.material_type_frame = self.addLabelFrame("Material Type")
        self.nextCol()
        self.numerical_params_frame = self.addLabelFrame("MPC Settings")
        self.files_frame = self.addLabelFrame("Log Files")


        self.exp_type = tk.StringVar(value=ExperimentType.REAL)
        self.adaptive_velocity = tk.StringVar(value="y")
        self.constant_velocity = tk.StringVar()
        self.model_name = tk.StringVar(value="multiisotherm")
        self.material_name = tk.StringVar(value="human")
        self.n_isotherms = tk.IntVar(value=1)
        self.n_horizons = tk.IntVar(value=10)
        self.qd = tk.IntVar(value=1)
        self.qw = tk.IntVar(value=25)
        self.save_dir = tk.StringVar()
        self.logFile = tk.StringVar()
        self.create_widgets()
        self.run()

    def create_widgets(self):
        self.AccentButton("Start Experiment", command=self.start_experiment)

        self.exp_type_frame.Radiobutton("Real", variable=self.exp_type, value=ExperimentType.REAL)
        self.exp_type_frame.Radiobutton("Pre-recorded", variable=self.exp_type, value=ExperimentType.PRERECORDED)
        self.exp_type_frame.Radiobutton("Simulated", variable=self.exp_type, value=ExperimentType.SIMULATED)

        self.velocity_type_frame.Radiobutton("MPC", variable=self.adaptive_velocity, value="y")
        self.velocity_type_frame.Radiobutton("Constant Velocity", variable=self.adaptive_velocity, value="n")

        self.model_type_frame.Radiobutton("Multi-Isotherm", variable=self.model_name, value="multiisotherm")
        self.model_type_frame.Radiobutton("MPC Pseudo-Static", variable=self.model_name, value="pseudostatic")
        self.model_type_frame.Radiobutton("Minimization", variable=self.model_name, value="minimization")

        self.material_type_frame.Radiobutton("Human Tissue", variable=self.material_name, value="human")
        self.material_type_frame.Radiobutton("Hydrogel Phantom", variable=self.material_name, value="hydrogel")

        self.numerical_params_frame.Label("Number of Isotherms")
        self.numerical_params_frame.Entry(textvariable=self.n_isotherms)
        self.numerical_params_frame.Label("Number of Horizons")
        self.numerical_params_frame.Entry(textvariable=self.n_horizons)
        self.numerical_params_frame.Label("Qd")
        self.numerical_params_frame.Entry(textvariable=self.qd)
        self.numerical_params_frame.Label("Qw")
        self.numerical_params_frame.Entry(textvariable=self.qw)

        self.save_indicator = self.files_frame.Button("Log Save Directory", command=self.folderDialog)
        self.load_indicator = self.files_frame.Button("Load Log File", command=self.loadLogDialog)

    def model_selection(self):
        if self.model_name.get() == "minimization":
            self.numerical_params_frame.Label("Number of Horizons", disabled=True)
            self.numerical_params_frame.Entry(disabled=True)
            self.numerical_params_frame.Label("Qd", disabled=True)
            self.numerical_params_frame.Entry(disabled=True)
        else:
            self.numerical_params_frame.Label("Number of Horizons", disabled=False)
            self.numerical_params_frame.Entry(disabled=False)
            self.numerical_params_frame.Label("Qd", disabled=False)
            self.numerical_params_frame.Entry(disabled=False)

    def folderDialog(self):
        filename = filedialog.askdirectory()
        self.save_dir.set(filename)
        self.save_indicator.config(text=f"Save to: {filename}")

    def loadLogDialog(self):
        filename = filedialog.askopenfilename(filetypes=[("Log Files", "*.pkl")])
        self.logFile.set(filename)
        self.load_indicator.config(text=f"Load from: {filename.split('/')[-1]}")


    def start_experiment(self):

        adaptive_velocity = self.adaptive_velocity.get() == "y"
        constant_velocity = float(self.constant_velocity.get()) if not adaptive_velocity else None
        material = humanTissue if self.material_name.get() == "human" else hydrogelPhantom
        if self.model_name.get() == "multiisotherm":
            if self.n_isotherms.get() < 3:
                messagebox.showerror("Warning", "Multi-Isotherm model requires at least 3 isotherms")
                return

        if (name := self.model_name.get()) == "minimization":
            model = SteadyStateMinimizationModel(material)
        elif name == "multiisotherm":
            model = MultiIsothermMPCModel(n_isotherms=self.n_isotherms.get(), material=material)
        elif name == "pseudostatic":
            model = PseudoStaticMPCModel(material)
        else:
            raise ValueError(f"Invalid model name: {name}")

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

        elif exp_type == ExperimentType.REAL:
            try:
                t3 = T3pro()
            except serial.serialutil.SerialException:
                messagebox.showerror("Error", "No T3pro found")
                return
            try:
                tb = Testbed()
            except serial.serialutil.SerialException:
                messagebox.showerror("Error", "No Testbed found")
                return


        if (save_dir:=self.save_dir.get()) == "":
            messagebox.showinfo("Warning", "No save directory selected, data will not be saved")

        main(model=model, mpc=mpc, exp_type=exp_type, adaptive_velocity=adaptive_velocity, constant_velocity=constant_velocity,
            log_save_dir=save_dir, log_file_to_load=self.logFile.get(), t3=t3, tb=tb)
        if t3 is not None:
            t3.release()
        if tb is not None:
            tb.stop()



if __name__ == "__main__":
    app = ControlExperimentUI()
