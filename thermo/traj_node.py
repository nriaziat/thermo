from .trajectory_planner import Parameters
from .trajectory_planner import main as thermo
import click
import json

@click.command()
@click.argument('trajectory_path', type=str)
@click.argument('params_path', type=str)
def main(trajectory_path: str, params_path: str):
    """
    Main function to run the experiment UR10 arm.
    TRAJECTORY_PATH: Path to the trajectory
    PARAMS_PATH: Path to the parameters file
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
        params = Parameters(**params)

    thermo(params=params, trajectory_path=trajectory_path)