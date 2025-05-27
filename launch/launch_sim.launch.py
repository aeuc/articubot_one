import os
import math

from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node



def generate_launch_description():


    # Include the robot_state_publisher launch file, provided by our own package. Force sim time to be enabled
    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!

    package_name='articubot_one' #<--- CHANGE ME

    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','rsp.launch.py'
                )]), launch_arguments={'use_sim_time': 'true', 'use_ros2_control': 'true'}.items()
    )

    joystick = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','joystick.launch.py'
                )]), launch_arguments={'use_sim_time': 'true'}.items()
    )

    twist_mux_params = os.path.join(get_package_share_directory(package_name),'config','twist_mux.yaml')
    twist_mux = Node(
            package="twist_mux",
            executable="twist_mux",
            parameters=[twist_mux_params, {'use_sim_time': True}],
            remappings=[('/cmd_vel_out','/diff_cont/cmd_vel_unstamped')]
        )

    gazebo_params_file = os.path.join(get_package_share_directory(package_name),'config','gazebo_params.yaml')

    # Include the Gazebo launch file, provided by the gazebo_ros package
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
                    launch_arguments={'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file}.items()
             )

    # Run the spawner node from the gazebo_ros package. The entity name doesn't really matter if you only have a single robot.
    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'my_bot'],
                        output='screen')


    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont"],
    )

    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broad"],
    )

    kinect_tilt_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["kinect_tilt_controller"],
    )

    kinect_pan_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["kinect_pan_controller"],
    )



    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[('cloud_in', '/kinect/kinect_depth/points'), # Input PointCloud2 topic
                    ('scan', '/scan')],                   # Output LaserScan topic
        parameters=[{
            'target_frame': 'base_footprint', # Or 'kinect_depth_sensor_optical_link' or 'base_link'
                                                        # This is the frame in which the scan will be generated.
                                                        # Usually the frame of the sensor or a frame aligned with robot base.
            'transform_tolerance': 0.1,    # Time (s) to wait for transform to target_frame
            'min_height': 0.05,            # Minimum Z height of points to consider (relative to target_frame)
            'max_height': 0.40,             # Maximum Z height of points to consider (relative to target_frame)
                                            # This creates a 20cm thick horizontal slice. Adjust as needed.
            'angle_min': -1.089/2.0,    # Start angle of the scan (rad), e.g., -90 degrees
            'angle_max': 1.089/2.0,     # End angle of the scan (rad), e.g., +90 degrees
                                            # This gives a 180-degree forward-facing scan.
            'angle_increment': 1.089/360.0, # Angular resolution (rad), e.g., 0.5 degree increments
            'scan_time': 0.1,               # Time between scans (s), 1/update_rate
            'range_min': 0.20,              # Minimum range of the generated scan (m)
            'range_max': 4.0,               # Maximum range of the generated scan (m)
            'use_inf': True,                # Whether to use +/- INF for out-of-range points
            'inf_epsilon': 1.0,
            # 'concurrency_level': 1,       # Number of threads to use
            'use_sim_time': True    # Pass the use_sim_time parameter
        }]
    )


    # Code for delaying a node (I haven't tested how effective it is)
    # 
    # First add the below lines to imports
    # from launch.actions import RegisterEventHandler
    # from launch.event_handlers import OnProcessExit
    #
    # Then add the following below the current diff_drive_spawner
    # delayed_diff_drive_spawner = RegisterEventHandler(
    #     event_handler=OnProcessExit(
    #         target_action=spawn_entity,
    #         on_exit=[diff_drive_spawner],
    #     )
    # )
    #
    # Replace the diff_drive_spawner in the final return with delayed_diff_drive_spawner



    # Launch them all!
    return LaunchDescription([
        rsp,
        joystick,
        twist_mux,
        gazebo,
        spawn_entity,
        diff_drive_spawner,
        joint_broad_spawner,
        kinect_tilt_spawner,
        kinect_pan_spawner,
        pointcloud_to_laserscan_node
    ])
