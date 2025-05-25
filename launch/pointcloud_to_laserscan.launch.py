from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration # If using launch args
from launch.actions import DeclareLaunchArgument     # If using launch args
import math # For pi

# ... other parts of your launch file ...

def generate_launch_description():
    # ... (your existing launch descriptions) ...

    # Declare launch arguments for flexibility (optional, but good practice)
    # You might already have use_sim_time declared
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true')
    use_sim_time = LaunchConfiguration('use_sim_time')


    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        remappings=[('cloud_in', '/kinect/kinect_depth/points'), # Input PointCloud2 topic
                    ('scan', '/scan')],                   # Output LaserScan topic
        parameters=[{
            'target_frame': 'kinect_depth_sensor_link', # Or 'kinect_depth_sensor_optical_link' or 'base_link'
                                                        # This is the frame in which the scan will be generated.
                                                        # Usually the frame of the sensor or a frame aligned with robot base.
            'transform_tolerance': 0.03,    # Time (s) to wait for transform to target_frame
            'min_height': -0.10,            # Minimum Z height of points to consider (relative to target_frame)
            'max_height': 0.10,             # Maximum Z height of points to consider (relative to target_frame)
                                            # This creates a 20cm thick horizontal slice. Adjust as needed.
            'angle_min': -math.pi / 2.0,    # Start angle of the scan (rad), e.g., -90 degrees
            'angle_max': math.pi / 2.0,     # End angle of the scan (rad), e.g., +90 degrees
                                            # This gives a 180-degree forward-facing scan.
            'angle_increment': math.pi / 360.0, # Angular resolution (rad), e.g., 0.5 degree increments
            'scan_time': 0.1,               # Time between scans (s), 1/update_rate
            'range_min': 0.20,              # Minimum range of the generated scan (m)
            'range_max': 5.0,               # Maximum range of the generated scan (m)
            'use_inf': True,                # Whether to use +/- INF for out-of-range points
            'inf_epsilon': 1.0,
            # 'concurrency_level': 1,       # Number of threads to use
            'use_sim_time': use_sim_time    # Pass the use_sim_time parameter
        }]
    )

    return LaunchDescription([
        declare_use_sim_time_arg,
        # ... (your existing launch actions like rsp, gazebo, spawners, etc.) ...
        pointcloud_to_laserscan_node # Add this node
    ])