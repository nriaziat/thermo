from ExperimentManager import VirtualExperimentManager
from ThermalProcessing import OnlineVelocityOptimizer
import pickle as pkl
import matplotlib.pyplot as plt

thermal_px_per_mm = 5.1337 # px/mm
velopt = OnlineVelocityOptimizer(t_death=45,
                                 v_min=1,
                                 v_max=15)

if __name__ == "__main__":
    with open("logs/data_adaptive_2024-08-06-15:43.pkl", "rb") as f:
        data = pkl.load(f)
    em = VirtualExperimentManager(
            data_save=data,
            velopt=velopt,
            debug=False,
            adaptive_velocity=True)
    em.thermal_px_per_mm = thermal_px_per_mm
    em.run_experiment()
    # em.plot()