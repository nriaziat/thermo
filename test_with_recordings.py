from ExperimentManager import VirtualExperimentManager
from ThermalProcessing import OnlineVelocityOptimizer
import pickle as pkl

thermal_px_per_mm = 5.1337 # px/mm
velopt = OnlineVelocityOptimizer(des_width=3 * thermal_px_per_mm,
                                 t_death=50,
                                 v_min=0.2,
                                 v_max=10)

if __name__ == "__main__":
    with open("logs/data_adaptive_2024-07-26-14:48.pkl", "rb") as f:
        data = pkl.load(f)
    adaptive_velocity = input("Adaptive Velocity? (y/n): ").lower().strip()
    constant_velocity = None
    if adaptive_velocity == "n":
        constant_velocity = float(input("Enter constant velocity: "))
        print(f"Constant velocity: {constant_velocity} mm/s")
    em = VirtualExperimentManager(
            data_save=data,
            velopt=velopt,
            debug=False,
            adaptive_velocity=adaptive_velocity == "y",
            const_velocity=constant_velocity)
    em.thermal_px_per_mm = thermal_px_per_mm
    em.run_experiment()
    em.plot()