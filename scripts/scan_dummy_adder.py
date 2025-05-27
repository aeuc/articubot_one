import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import copy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy # Import QoS classes

class ScanRepublisherNode(Node):
    def __init__(self):
        super().__init__('scan_republisher_node')
        
        self.declare_parameter('input_scan_topic', '/scan')
        self.declare_parameter('output_scan_topic', '/scan_modified')

        input_topic = self.get_parameter('input_scan_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_scan_topic').get_parameter_value().string_value

        ##### NEW/MODIFIED: Define QoS Profile for Subscription #####
        # Standard QoS for sensor data: Best Effort, Volatile, Keep Last (depth 1 is common for best effort)
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1, # For BEST_EFFORT, often depth 1 is fine, or match publisher's depth
            durability=DurabilityPolicy.VOLATILE # Sensor data is typically volatile
        )
        ##### END NEW/MODIFIED #####

        self.subscription = self.create_subscription(
            LaserScan,
            input_topic,
            self.scan_callback,
            qos_profile=sensor_qos_profile # ##### NEW/MODIFIED: Apply QoS to subscriber #####
        )
        # For the publisher, the default QoS might be fine, or you can also specify one:
        self.publisher = self.create_publisher(LaserScan, output_topic, qos_profile=sensor_qos_profile) # Optionally set QoS for publisher too
        
        self.get_logger().info(f"Scan Republisher Node started. Subscribing to '{input_topic}' (Best Effort), publishing to '{output_topic}'.")

    def scan_callback(self, msg: LaserScan):
        # ... (rest of your scan_callback logic is likely fine) ...
        modified_msg = copy.deepcopy(msg)

        if len(modified_msg.ranges) == 360:
            last_valid_range = None
            for r_val in reversed(modified_msg.ranges):
                if not math.isinf(r_val) and not math.isnan(r_val):
                    last_valid_range = r_val
                    break
            if last_valid_range is None:
                last_valid_range = modified_msg.ranges[-1] if modified_msg.ranges else msg.range_max

            new_ranges = list(modified_msg.ranges)
            new_ranges.append(last_valid_range)
            modified_msg.ranges = new_ranges

            if len(modified_msg.ranges) == 361:
                modified_msg.angle_max = modified_msg.angle_min + (360 * modified_msg.angle_increment)
            
            # self.get_logger().debug(f"Modified scan: {len(modified_msg.ranges)} ranges, new angle_max: {modified_msg.angle_max}")

        elif len(modified_msg.ranges) == 361:
            # self.get_logger().debug("Scan already has 361 ranges, republishing as is.")
            pass
        else:
            self.get_logger().warn(
                f"Received scan with {len(modified_msg.ranges)} ranges, not 360 or 361. Republishing as is."
            )
        self.publisher.publish(modified_msg)

# ... (main function remains the same) ...
def main(args=None):
    rclpy.init(args=args)
    node = ScanRepublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()