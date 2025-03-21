# ASTR ThERMO Spinup

## Prerequisites

1. Source astr workspace, thermo workspace, and zivid workspace.
2. Ensure zivid is turned on via powerstrip.
3. Mount tissue sample under robot arm and position such that zivid camera can see it. 

## Capture Point Cloud
1. Run astr: ```ros2 launch astr_driver astr_bringup.launch.py launch_rviz:=true use_fake_hardware:=false description_file_electrocautery_arm:=electrocautery_arm_mounted_with_zivid_and_tool.urdf.xacro```
2. Run orocos deployer: ```source orocos_ws/install/setup.bash && deployer -s /home/imerse/ros2_wss/astr2_ws/install/astr_controller/share/astr_controller/scripts/astr_mid_level_controller.ops```.
3. Run ```ros2 run nephrectomy_vision pc_helper``` for converting to numpy and downsampling. 
4. Start zivid driver: ```source ~/ros2_wss/nephrectomy_ws/install/setup.bash && ros2 run zivid_camera zivid_camera --ros-args -p settings_file_path:=/home/imerse/fortongue.yml```. Point file path to yaml saved from Zivid studio. For red tongue, use red subsample. 
5. Run ```ros2 launch nephrectomy_vision camera_tf_broadcasters.launch.py```. Ensure ```zivid_link``` exists in RVIZ. 
6. Trigger capture with ```ros2 service call /get_point_cloud std_srvs/srv/Trigger``` and wait for resampling and saving. 

## Analyze PC and get Trajectorys

1. Move .ply file to ethan_ws. 
2. Run ethan's planner: ```python3 manual_planner_circle.py --pcd observed_pointcloud.ply --visualize True```. Outputs normals.npy and points.npy (and verification pyplot). Double check normals and make sure they are similarly oriented, otherwise yell at ethan and remove those points manually. 
3. Move normals and points to speed_ws and place in trajectory folder. 

## Preplan and visualize path. 

To visualize the PCD, use ```ros2 run cao_planner cf_get_seg_and_publish``` and edit the code to point to your pcd. 
1. Go to speed_ws and source ros and python env. 
2. On the teach pendant, run either ```urcap``` or ```external_control```.
2. Run ros node: ```ros2 run thermo node ./src/trajectory ./src/thermo/thermo/params.json``` and point to trajectory directory. 


Run deployer with ```robot1.SetToolFrameOffset(-4, 2.5, 0)``` and adjust z (+) if its not contacting. The deployer may need to ```cd robot1``` if not set already.

