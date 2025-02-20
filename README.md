# ASTR ThERMO Spinup

## Prerequisites

1. Source astr workspace, thermo workspace, and zivid workspace.
2. Ensure zivid is turned on via powerstrip.
3. Mount tissue sample under robot arm and position such that zivid camera can see it. 

## Capture Point Cloud
1. Run astr: ```ros2 launch astr_driver astr_bringup.launch.py launch_rviz:=true use_fake_hardware:=true description_file_electrocautery_arm:=electrocautery_arm_mounted_with_zivid_and_tool.urdf.xacro```
2. Run deployer: ```deployer -s /home/imerse/ros2_wss/astr2_ws/install/astr_controller/share/astr_controller/scripts/astr_mid_level_controller.ops```
3. Run ```ros2 run nephrectomy_vision pc_helper``` for converting to numpy and downsampling. 
4. Start zivid driver: ```ros2 run zivid_camera zivid_camera --ros-args -p settings_file_path:=/home/imerse/red_zivid.yml```. Point file path to yaml saved from Zivid studio. For red tongue, use red subsample. 
5. Run ```ros2 launch nephrectomy_vision camera_tf_broadcasters.launch.py```. Ensure ```zivid_link``` exists in RVIZ. 
6. Trigger capture with ```ros2 service call /get_point_cloud std_srvs/srv/Trigger``` and wait for resampling and saving. 

## Analyze PC and get Trajectory

1. Move .ply file from nephrectomy_ws to ethan_ws. 
2. Run ethan's planner: ```python3 manual_planner.py```. Outputs normals.npy and points.npy (and verification pyplot). Double check normals and make sure they are similarly oriented, otherwise yell at ethan and remove those points manually. 
3. Move normals and points to speed_ws and place in trajectory folder. 

## Preplan and visualize path. 

1. Go to speed_ws and source ros and python env. 
2. Run ros node: ```ros2 run thermo node ./src/trajectory``` and point to trajectory directory. 
