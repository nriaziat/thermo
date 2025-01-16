import pandas as pd
from datetime import datetime

class DataLogger:
    def __init__(self, log_save_dir: str, adaptive_velocity: bool, constant_velocity: float):
        self.log_save_dir = log_save_dir
        self.adaptive_velocity = adaptive_velocity
        self.constant_velocity = constant_velocity
        self.data_log = pd.DataFrame(
            columns=['time_sec', 'position_mm', 'widths_mm', 'velocities', 'vstar', 'thermal_deflections_mm', 'aruco_pos_mm', 'thermal_frames', 'deflection_estimates', 'thermal_estimates']
        )
        self.start_time = datetime.now()

    def log_data(self, pos, w_mm, u0, vstar, defl_mm, thermal_arr, defl_adaptation, therm_adaptation, aruco_defl):
        dt = (datetime.now() - self.start_time).total_seconds()
        self.data_log.loc[len(self.data_log)] = [
            dt, pos, w_mm, u0, vstar, defl_mm, aruco_defl, thermal_arr,
            {'c_defl': defl_adaptation.c_defl},
            {'q': therm_adaptation.q, 'Cp': therm_adaptation.Cp, 'k': therm_adaptation.k, 'rho': therm_adaptation.rho}
        ]

    def save_log(self):
        if len(self.data_log['widths_mm']) > 0 and self.log_save_dir != "":
            date = datetime.now()
            mode = "adaptive" if self.adaptive_velocity else f"{self.constant_velocity:.0f}mm_s"
            fname = f"{self.log_save_dir}/data_{mode}_{date.strftime('%Y-%m-%d-%H:%M')}.pkl"
            self.data_log.to_pickle(fname)
