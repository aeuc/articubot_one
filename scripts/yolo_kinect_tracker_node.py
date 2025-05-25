#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter # Keep if you use dynamic reconfigure elsewhere
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String, Float64 # ##### NEW/MODIFIED #####
from sensor_msgs.msg import CompressedImage # ##### NEW/MODIFIED #####
# from cv_bridge import CvBridge # Not strictly needed for imdecode, but good to have if you use raw Image
import cv2
from ultralytics import YOLO
import time
import threading # Keep if other parts of your original code relied on it, but frame_reader_thread is removed
import traceback
import math
import sys
import numpy as np # ##### NEW/MODIFIED #####

# --- Constants ---
DEFAULT_MODEL_NAME = "/home/rosegarden/ros2_ws/yolo11l.pt"
##### NEW/MODIFIED DEFAULTS #####
DEFAULT_KINECT_IMAGE_TOPIC = '/kinect/kinect_depth/image_raw/compressed' # Assuming this is compressed RGB
DEFAULT_TILT_GOAL_TOPIC = '/kinect_tilt_angle_goal'
# Default Vertical FoV (radians) - PLEASE VERIFY AND SET ACCURATELY FOR YOUR SENSOR
DEFAULT_KINECT_VERTICAL_FOV_RADIANS = math.radians(45.0)
DEFAULT_TILT_P_GAIN = 0.3  # Proportional gain for tilt control
DEFAULT_MAX_TILT_INCREMENT_RAD_PER_FRAME = math.radians(1.5) # Max tilt speed per frame
DEFAULT_MIN_TILT_LIMIT_RAD = math.radians(-40.0) # Match URDF kinect_tilt_joint lower limit
DEFAULT_MAX_TILT_LIMIT_RAD = math.radians(25.0)  # Match URDF kinect_tilt_joint upper limit
##### END NEW/MODIFIED DEFAULTS #####

DEFAULT_NODE_NAME = 'yolo_kinect_tracker_node'
DEFAULT_YOLO_PUBLISH_TOPIC = '/yolo_detections'
DEFAULT_TIMER_RATE_HZ = 10.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DISPLAY_WINDOW = True
RESIZE_WIDTH = 320
RESIZE_HEIGHT = 240
# READER_THREAD_SLEEP_S = 0.001 # Not needed anymore
TARGET_CLASS_NAME = "person"

class YoloKinectTrackerNode(Node):
    def __init__(self):
        super().__init__(DEFAULT_NODE_NAME)

        # --- Declare Parameters ---
        self.declare_parameter('model_name', DEFAULT_MODEL_NAME)
        self.declare_parameter('yolo_publish_topic', DEFAULT_YOLO_PUBLISH_TOPIC)
        self.declare_parameter('timer_rate_hz', DEFAULT_TIMER_RATE_HZ)
        self.declare_parameter('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        self.declare_parameter('display_window', DISPLAY_WINDOW)
        self.declare_parameter('resize_width', RESIZE_WIDTH)
        self.declare_parameter('resize_height', RESIZE_HEIGHT)

        ##### NEW/MODIFIED: Declare new parameters #####
        self.declare_parameter('kinect_image_topic', DEFAULT_KINECT_IMAGE_TOPIC)
        self.declare_parameter('tilt_goal_topic', DEFAULT_TILT_GOAL_TOPIC)
        self.declare_parameter('kinect_vertical_fov_radians', DEFAULT_KINECT_VERTICAL_FOV_RADIANS)
        self.declare_parameter('tilt_p_gain', DEFAULT_TILT_P_GAIN)
        self.declare_parameter('max_tilt_increment_rad_per_frame', DEFAULT_MAX_TILT_INCREMENT_RAD_PER_FRAME)
        self.declare_parameter('min_tilt_limit_rad', DEFAULT_MIN_TILT_LIMIT_RAD)
        self.declare_parameter('max_tilt_limit_rad', DEFAULT_MAX_TILT_LIMIT_RAD)
        ##### END NEW/MODIFIED #####

        self.add_on_set_parameters_callback(self.parameters_callback)
        self._load_parameters() # Load all parameters including new ones

        self.get_logger().info(f"--- Initializing {DEFAULT_NODE_NAME} ---")

        ##### NEW/MODIFIED: Threading Setup & Image Handling #####
        self.latest_compressed_frame_data = None # Will store bytes from CompressedImage
        self.frame_lock = threading.Lock()
        # self.reader_thread_running = False # Not needed
        self.current_commanded_tilt_angle = 0.0 # Internal state for tilt
        ##### END NEW/MODIFIED #####

        # --- ROS Publishers ---
        self.yolo_publisher = self.create_publisher(String, self.yolo_publish_topic, 10)
        ##### NEW/MODIFIED: Add tilt goal publisher #####
        self.tilt_goal_publisher = self.create_publisher(Float64, self.tilt_goal_topic, 10)
        ##### END NEW/MODIFIED #####

        # --- YOLO Model ---
        self.person_class_index = None
        try:
            self.model = YOLO(self.model_name)
            self.get_logger().info(f"Model '{self.model_name}' loaded successfully.")
            if hasattr(self.model, 'names'):
                names_map = {name.lower(): index for index, name in self.model.names.items()}
                if TARGET_CLASS_NAME.lower() in names_map:
                    self.person_class_index = names_map[TARGET_CLASS_NAME.lower()]
                    self.get_logger().info(f"Found class '{TARGET_CLASS_NAME}' with index: {self.person_class_index}")
                else:
                    self.get_logger().error(f"Could not find class '{TARGET_CLASS_NAME}' in model names: {list(self.model.names.values())}. Tracking will fail!")
            else:
                self.get_logger().warn(f"Model does not have a 'names' attribute. Cannot filter by class index.")
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"Failed to load YOLO model '{self.model_name}': {e}\nTraceback:\n{tb_str}")
            self.create_timer(0.1, lambda: self.safe_shutdown("YOLO model load failed"))
            return

        ##### NEW/MODIFIED: ROS Subscriber for Compressed Image #####
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            self.kinect_image_topic,
            self.compressed_image_callback,
            10  # QoS depth
        )
        self.get_logger().info(f"Subscribed to compressed image topic: {self.kinect_image_topic}")

        self.timer = None
        self._create_processing_timer()

        self.frame_count_display = 0
        self.start_time_display = time.time()
        self.get_logger().info(f"--- {DEFAULT_NODE_NAME} Initialization Complete ---")

    def _load_parameters(self):

        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.yolo_publish_topic = self.get_parameter('yolo_publish_topic').get_parameter_value().string_value
        self.timer_rate_hz = self.get_parameter('timer_rate_hz').get_parameter_value().double_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.display_window = self.get_parameter('display_window').get_parameter_value().bool_value
        self.resize_width = self.get_parameter('resize_width').get_parameter_value().integer_value
        self.resize_height = self.get_parameter('resize_height').get_parameter_value().integer_value

        ##### NEW/MODIFIED: Load new parameters #####
        self.kinect_image_topic = self.get_parameter('kinect_image_topic').get_parameter_value().string_value
        self.tilt_goal_topic = self.get_parameter('tilt_goal_topic').get_parameter_value().string_value
        self.kinect_vertical_fov_radians = self.get_parameter('kinect_vertical_fov_radians').get_parameter_value().double_value
        self.tilt_p_gain = self.get_parameter('tilt_p_gain').get_parameter_value().double_value
        self.max_tilt_increment_rad_per_frame = self.get_parameter('max_tilt_increment_rad_per_frame').get_parameter_value().double_value
        self.min_tilt_limit_rad = self.get_parameter('min_tilt_limit_rad').get_parameter_value().double_value
        self.max_tilt_limit_rad = self.get_parameter('max_tilt_limit_rad').get_parameter_value().double_value
        ##### END NEW/MODIFIED #####

    def parameters_callback(self, params):
        success = True
        require_timer_recreate = False
        for param in params:
            try:
                if param.name == 'timer_rate_hz':
                    self.timer_rate_hz = param.value.double_value
                    require_timer_recreate = True
                    self.get_logger().info(f"Updated timer rate to: {self.timer_rate_hz:.1f} Hz")
                elif param.name == 'confidence_threshold':
                    self.confidence_threshold = param.value.double_value
                    self.get_logger().info(f"Updated confidence threshold to: {self.confidence_threshold:.2f}")
                ##### NEW/MODIFIED: Handle new dynamic parameters #####
                elif param.name == 'tilt_p_gain':
                    self.tilt_p_gain = param.value.double_value
                    self.get_logger().info(f"Updated tilt_p_gain to: {self.tilt_p_gain:.3f}")
                elif param.name == 'max_tilt_increment_rad_per_frame':
                    self.max_tilt_increment_rad_per_frame = param.value.double_value
                # Add more if you want kinect_vertical_fov_radians, min/max_tilt_limit_rad to be dynamic
                ##### END NEW/MODIFIED #####
            except Exception as e:
                self.get_logger().error(f"Error processing parameter update for {param.name}: {e}")
                success = False

        if require_timer_recreate:
            self._create_processing_timer()
        return SetParametersResult(successful=success)

    def _create_processing_timer(self):
        if hasattr(self, 'timer') and self.timer is not None and not self.timer.is_canceled():
            self.timer.cancel()
        timer_period = 1.0 / self.timer_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Processing timer (re)created with period: {timer_period:.3f} s ({self.timer_rate_hz:.1f} Hz)")

    ##### REMOVE _frame_reader_loop method entirely #####
    # def _frame_reader_loop(self): ...

    ##### NEW/MODIFIED: Callback for compressed images #####
    def compressed_image_callback(self, msg: CompressedImage):
        """Callback for ROS CompressedImage messages."""
        with self.frame_lock:
            self.latest_compressed_frame_data = msg.data
    ##### END NEW/MODIFIED #####

    def timer_callback(self):
        ##### NEW/MODIFIED: Get and decompress frame #####
        compressed_data = None
        with self.frame_lock:
            if self.latest_compressed_frame_data is not None:
                compressed_data = self.latest_compressed_frame_data
                self.latest_compressed_frame_data = None # Process only once

        if compressed_data is None:
            # self.get_logger().debug("Timer cb: No new compressed frame.")
            return

        frame_to_process = None
        try:
            np_arr = np.frombuffer(compressed_data, np.uint8)
            frame_to_process = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Assumes BGR
            if frame_to_process is None:
                self.get_logger().warn("cv2.imdecode failed. Check compressed image format.")
                return
        except Exception as e:
            self.get_logger().error(f"Error decompressing image: {e}\n{traceback.format_exc()}")
            return
        ##### END NEW/MODIFIED #####

        try:
            frame_display = cv2.resize(frame_to_process, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().warn(f"Error resizing frame: {e}\nTraceback:\n{tb_str}. Skipping frame.")
            return

        results = None
        track_args = {
            "persist": True, "verbose": False, "conf": self.confidence_threshold,
            "device": "cuda", # or "cpu"
            "classes": [self.person_class_index] if self.person_class_index is not None else None
        }
        try:
            frame_for_yolo = frame_display.copy() # YOLO might modify input
            results = self.model.track(frame_for_yolo, **track_args)
        except Exception as e:
            tb_str = traceback.format_exc()
            self.get_logger().error(f"Error during model tracking: {e}\nTraceback:\n{tb_str}")
            results = None

        person_detection_data_list = []
        ##### NEW/MODIFIED: Variables for tilt target #####
        target_person_y_center = None
        target_person_detected = False
        best_box_area = -1 # For selecting largest person if multiple
        ##### END NEW/MODIFIED #####

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                    is_person = (cls_id == self.person_class_index) if self.person_class_index is not None else False
                    if not is_person:
                        continue
                    
                    target_person_detected = True # A person is detected

                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = box.conf[0].item() if box.conf is not None else 0.0
                    track_id = int(box.id[0].item()) if box.id is not None and box.id[0] is not None and not math.isnan(box.id[0].item()) else -1

                    class_name = TARGET_CLASS_NAME
                    detection_str = (
                        f"TrackID: {track_id}, Class: {class_name}, Conf: {conf:.2f}, "
                        f"BBox: [{x1},{y1},{x2},{y2}]"
                    )
                    person_detection_data_list.append(detection_str)

                    ##### NEW/MODIFIED: Select target for tilting (e.g., largest person) #####
                    current_area = (x2 - x1) * (y2 - y1)
                    if current_area > best_box_area:
                        best_box_area = current_area
                        target_person_y_center = (y1 + y2) / 2.0
                    ##### END NEW/MODIFIED #####

                    if self.display_window:
                        label = f"ID:{track_id} {class_name} {conf:.2f}"
                        color = (0, 255, 0)
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame_display, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                        # ROI rectangle already there, no change
                except Exception as e:
                    tb_str = traceback.format_exc()
                    self.get_logger().warn(f"Error processing a detection box: {e}\nTraceback:\n{tb_str}")
                    continue

        if person_detection_data_list:
            yolo_msg = String()
            yolo_msg.data = "\n".join(person_detection_data_list)
            self.yolo_publisher.publish(yolo_msg)

        ##### NEW/MODIFIED: Kinect Tilt Control Logic #####
        if target_person_detected and target_person_y_center is not None:
            image_center_y = self.resize_height / 2.0
            pixel_error_y = target_person_y_center - image_center_y # Positive if person below center

            # Proportional control for tilt adjustment
            normalized_pixel_error = pixel_error_y / (self.resize_height / 2.0) # Error from -1 to 1
            tilt_adjustment_rad = self.tilt_p_gain * normalized_pixel_error
            
            # Limit max tilt speed per frame
            tilt_adjustment_rad = np.clip(tilt_adjustment_rad,
                                          -self.max_tilt_increment_rad_per_frame,
                                          self.max_tilt_increment_rad_per_frame)

            new_target_tilt_rad = self.current_commanded_tilt_angle + tilt_adjustment_rad
            new_target_tilt_rad = np.clip(new_target_tilt_rad,
                                           self.min_tilt_limit_rad,
                                           self.max_tilt_limit_rad)

            tilt_msg = Float64()
            tilt_msg.data = new_target_tilt_rad
            self.tilt_goal_publisher.publish(tilt_msg)
            self.current_commanded_tilt_angle = new_target_tilt_rad # Update internal state
            
            self.get_logger().debug(f"TargetY: {target_person_y_center:.1f}, PxErr: {pixel_error_y:.1f}, TiltCmd: {math.degrees(new_target_tilt_rad):.1f}deg")
        ##### END NEW/MODIFIED #####

        if self.display_window:
            self.frame_count_display += 1
            elapsed_time = time.time() - self.start_time_display
            if elapsed_time >= 1.0:
                fps = self.frame_count_display / elapsed_time
                self.current_fps_display = f"Proc FPS: {fps:.1f}"
                self.start_time_display = time.time()
                self.frame_count_display = 0
            if hasattr(self, 'current_fps_display'):
                cv2.putText(frame_display, self.current_fps_display, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            # Draw image center for reference
            cv2.circle(frame_display, (self.resize_width // 2, self.resize_height // 2), 3, (0,0,255), -1)


            try:
                cv2.imshow('YOLO Kinect Tracker', frame_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Quit key pressed. Shutting down.")
                    self.safe_shutdown("User requested quit")
            except Exception as e:
                tb_str = traceback.format_exc()
                self.get_logger().error(f"Error during CV2 display: {e}\nTraceback:\n{tb_str}")

    def safe_shutdown(self, reason="Unknown"):
        if not rclpy.ok(): return
        self.get_logger().warn(f"Initiating shutdown... Reason: {reason}")

        ##### NEW/MODIFIED: Remove reader_thread_running #####
        # self.reader_thread_running = False
        ##### END NEW/MODIFIED #####
        if hasattr(self, 'timer') and self.timer is not None and not self.timer.is_canceled():
            self.timer.cancel()

        if rclpy.ok():
            try:
                self.create_timer(0.2, lambda: rclpy.try_shutdown(context=rclpy.get_default_context()))
                self.get_logger().info(f"Scheduled rclpy shutdown. Reason: {reason}")
            except Exception as e:
                tb_str = traceback.format_exc()
                self.get_logger().error(f"Failed to schedule shutdown timer: {e}\nTraceback:\n{tb_str}")
                if rclpy.ok(): rclpy.try_shutdown(context=rclpy.get_default_context())

    def destroy_node(self):
        self.get_logger().info("Destroying node and releasing resources...")
        ##### NEW/MODIFIED: Remove reader_thread and cap handling #####
        # self.reader_thread_running = False
        # time.sleep(0.05) # Can likely remove if no other threads need it

        if hasattr(self, 'timer') and self.timer is not None and not self.timer.is_canceled():
            self.timer.cancel()
            self.get_logger().info("Processing timer canceled.")

        # if hasattr(self, 'reader_thread') and self.reader_thread.is_alive():
        #     self.get_logger().info("Waiting for reader thread to exit...")
        #     self.reader_thread.join(timeout=2.0)
        #     if self.reader_thread.is_alive():
        #         self.get_logger().warn("Reader thread did not exit cleanly after timeout.")

        # if hasattr(self, 'cap') and self.cap is not None:
        #     if self.cap.isOpened():
        #         self.cap.release()
        #         self.get_logger().info("Video capture explicitly released.")
        #     self.cap = None
        ##### END NEW/MODIFIED #####
        try:
            if self.display_window and cv2.getWindowProperty('YOLO Kinect Tracker', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyAllWindows()
                self.get_logger().info("OpenCV windows closed.")
        except cv2.error:
            self.get_logger().debug("OpenCV window likely already closed or not created.")
        except Exception as e:
            self.get_logger().warn(f"Error closing OpenCV windows during destroy: {e}")

        super().destroy_node()
        self.get_logger().info("Node destruction complete.")

def main(args=None):
    rclpy.init(args=args)
    node = None
    exit_code = 0
    try:
        node = YoloKinectTrackerNode()
        init_success = True
        if not hasattr(node, 'model') or node.model is None: init_success = False
        ##### NEW/MODIFIED: Remove cap and reader_thread checks #####
        # if not hasattr(node, 'cap') or node.cap is None or not node.cap.isOpened():
        #     node.get_logger().warn("VideoCapture not ready after init, image subscriber will attempt connection.")
            # init_success = False # Don't fail init if image topic isn't immediately available
        # if not hasattr(node, 'reader_thread') or not node.reader_thread.is_alive():
        #    node.get_logger().error("Reader thread failed to start.") # Not used
        #    init_success = False # Not used
        ##### END NEW/MODIFIED #####

        if init_success:
            rclpy.spin(node)
        else:
            node.get_logger().error("Node initialization failed or critical components missing. Shutting down early.")
            exit_code = 1
            if node:
                # node.reader_thread_running = False # Not used
                if hasattr(node, 'timer') and node.timer is not None: node.timer.cancel()

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, initiating shutdown...")
        if node: node.safe_shutdown("KeyboardInterrupt")
    except Exception as e:
        exit_code = 1
        tb_str = traceback.format_exc()
        print(f"Unhandled exception in main:\n{tb_str}", file=sys.stderr)
        if node:
            node.get_logger().fatal(f"Unhandled exception in main loop: {e}\nTraceback:\n{tb_str}")
            node.safe_shutdown("Unhandled exception")
        time.sleep(0.5) # Allow logs to flush
    finally:
        if node is not None:
            if hasattr(node, 'destroy_node'): # Check if full init happened
                node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
            print("rclpy shutdown complete.")
        else:
            print("rclpy context already shut down or error during shutdown.")
        # sys.exit(exit_code) # Optional: propagate exit code

if __name__ == '__main__':
    main()