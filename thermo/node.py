from .astr_ros2 import Devices, RunConfig, ControlMode
from .astr_ros2 import main as thermo
from .T3pro import T3pro
from .models import SteadyStateMinimizationModel, hydrogelPhantom, humanTissue
import click

@click.command()
@click.argument('trajectory_path', type=str)
def main(trajectory_path: str):
    t3=None,
    model = SteadyStateMinimizationModel(qw=1, qd=1, r=1)
    devices = Devices(None, None)
    material = hydrogelPhantom
    run_conf = RunConfig(
        control_mode=ControlMode.CONSTANT_VELOCITY,
        adaptive_velocity=True,
        log_save_dir='./thermo_logs',
        material=material,
    )
    thermo(model=model, devices=devices, run_conf=run_conf, trajectory_path=trajectory_path)