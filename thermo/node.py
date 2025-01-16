from .astr_ros2 import Devices, RunConfig, ControlMode
from .astr_ros2 import main as thermo
from .T3pro import T3pro
from .models import SteadyStateMinimizationModel, hydrogelPhantom, humanTissue

def main():
    t3 = T3pro(port=4)
    model = SteadyStateMinimizationModel(qw=1, qd=1, r=1)
    devices = Devices(t3, None)
    material = hydrogelPhantom
    run_conf = RunConfig(
        control_mode=ControlMode.AUTONOMOUS,
        adaptive_velocity=True,
        log_save_dir='./thermo_logs',
        material=material,
    )
    thermo(model=model, devices=devices, run_conf=run_conf)