from .trajectory_planner import Parameters
from .trajectory_planner import main as thermo
import click
import json

@click.command()
@click.option('--trajectory_path', type=str, help="path to trajectory directory containing points.npy and normals.npy.")
@click.option('--params_path', type=str, help="path to params.json.")
@click.option('--smooth', is_flag=True, default=True, help="Apply exponential smoothing to orientations")
@click.option('--interpolation_factor', type=int, default=1, help="how many points to interpolate between given points.")
@click.option('--cut_depth_m', type=float, default=0, help="depth of cut in m.")
def main(trajectory_path: str, params_path: str, smooth: bool, interpolation_factor: int, cut_depth_m: float):
    """
    Main function to run the experiment UR10 arm.
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
        params = Parameters(**params)

    thermo(params=params, trajectory_path=trajectory_path, smooth_traj=smooth, interpolation_factor=interpolation_factor, cut_depth_m=cut_depth_m)