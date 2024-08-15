from ExperimentManager import VirtualExperimentManager
import pickle as pkl
from control_experiment import velopt

thermal_px_per_mm = 5.1337 # px/mm

if __name__ == "__main__":
    with open("logs/data_adaptive_2024-08-14-15:26.pkl", "rb") as f:
        data = pkl.load(f)
    em = VirtualExperimentManager(
            data_save=data,
            velopt=velopt,
            debug=False,
            adaptive_velocity=True)
    em.thermal_px_per_mm = thermal_px_per_mm
    em.run_experiment()
    # em.plot()