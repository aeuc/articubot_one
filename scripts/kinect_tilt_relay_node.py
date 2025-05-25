import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension # ** ADD MultiArrayDimension IMPORT **
from rcl_interfaces.msg import ParameterDescriptor

class KinectTiltRelayNode(Node):
    def __init__(self): # Corrected from _init
        super().__init__('kinect_tilt_relay_node') # Corrected from _init

        self.declare_parameter('goal_angle_topic', '/kinect_tilt_angle_goal',
                                ParameterDescriptor(description='Topic to subscribe for desired tilt angle (Float64).'))
        self.declare_parameter('controller_command_topic', '/kinect_tilt_controller/commands',
                                ParameterDescriptor(description='Topic to publish controller commands (Float64MultiArray).'))
        self.declare_parameter('joint_name', 'kinect_tilt_joint',
                                ParameterDescriptor(description='Name of the tilt joint.'))

        goal_topic = self.get_parameter('goal_angle_topic').get_parameter_value().string_value
        cmd_topic = self.get_parameter('controller_command_topic').get_parameter_value().string_value
        self.joint_name = self.get_parameter('joint_name').get_parameter_value().string_value

        self.subscription = self.create_subscription(
            Float64,
            goal_topic,
            self.goal_callback,
            10)
        self.publisher_ = self.create_publisher(Float64MultiArray, cmd_topic, 10)
        self.get_logger().info(f"Relay node initialized. Listening on '{goal_topic}', publishing to '{cmd_topic}'.")

    def goal_callback(self, msg: Float64):
        controller_command = Float64MultiArray()

        # Create a proper MultiArrayDimension object
        dim = MultiArrayDimension()
        dim.label = self.joint_name
        dim.size = 1  # Number of elements in this dimension (usually 1 for a single joint position)
        dim.stride = 1 # Total number of elements if it were a multi-dimensional array after this dim.
                       # For a 1D array of size N, dim[0].stride = N, dim[0].size = N.
                       # For a single value like this, size=1, stride=1 is common.
                       # The controller might also just care about controller_command.data and ignore layout for single values.

        controller_command.layout.dim.append(dim)
        controller_command.layout.data_offset = 0 # Usually 0 for simple arrays
        
        controller_command.data = [msg.data] # Target position for the kinect_tilt_joint

        self.publisher_.publish(controller_command)
        self.get_logger().debug(f"Relayed tilt command: {msg.data:.3f} rad for joint '{self.joint_name}'")

def main(args=None):
    rclpy.init(args=args)
    node = KinectTiltRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        self.get_logger().info("Keyboard interrupt, shutting down relay node.") # Added logger
    finally:
        if node and rclpy.ok(): # Check if node exists and rclpy is ok
            node.destroy_node()
        if rclpy.ok(): # Check rclpy status again before shutting down
            rclpy.shutdown()

if __name__ == '__main__':
    main()