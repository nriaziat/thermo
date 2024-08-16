from ExperimentManager import ExperimentManager
from T3pro import T3pro
from testbed import Testbed
from ThermalProcessing import OnlineVelocityOptimizer

thermal_px_per_mm = 5.1337 # px/mm

velopt = OnlineVelocityOptimizer(
                                 t_death_c=45,
                                 v_min=1,
                                 v_max=15)

if __name__ == "__main__":
    t3 = T3pro(port=0)
    tb = Testbed()

    adaptive_velocity = input("Adaptive Velocity? (y/n): ").lower().strip()
    constant_velocity = None
    if adaptive_velocity == "n":
        constant_velocity = float(input("Enter constant velocity: "))
        print(f"Constant velocity: {constant_velocity} mm/s")
    em = ExperimentManager(testbed=tb,
                           velopt=velopt,
                           video_save=True,
                           debug=False,
                           t3=t3,
                           adaptive_velocity=adaptive_velocity == "y",
                           const_velocity=constant_velocity)
    em.thermal_px_per_mm = thermal_px_per_mm
    em.run_experiment()
    # em.plot()