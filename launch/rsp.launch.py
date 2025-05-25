import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command, FindExecutable, PathJoinSubstitution # Added FindExecutable, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue # ** IMPORT THIS **

# import xacro # Not strictly needed if using Command for xacro processing

def generate_launch_description():

    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_ros2_control = LaunchConfiguration('use_ros2_control')

    # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('articubot_one'))
    xacro_file_name = 'robot.urdf.xacro' # Main URDF/XACRO file that includes robot_core.xacro, etc.
    xacro_file_path = os.path.join(pkg_path,'description', xacro_file_name)

    # Use PathJoinSubstitution and FindExecutable for robustness with xacro command
    robot_description_content = Command([ # Renamed for clarity from robot_description_config
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        xacro_file_path,
        ' use_ros2_control:=', use_ros2_control,
        ' sim_mode:=', use_sim_time
        # Add any other xacro arguments your robot.urdf.xacro needs, e.g.:
        # ' camera_enabled:=true',
        # ' lidar_enabled:=true'
    ])
    
    # Create a robot_state_publisher node
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description_content, value_type=str), # ** THE FIX **
            'use_sim_time': use_sim_time
        }]
    )

    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use sim time if true'),
        DeclareLaunchArgument(
            'use_ros2_control',
            default_value='true',
            description='Use ros2_control if true'),

        node_robot_state_publisher
    ])