#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult # Parameter was unused, keep if needed
from std_msgs.msg import String, Float64
from sensor_msgs.msg import CompressedImage
import cv2
from ultralytics import YOLO
import time
import threading
import traceback
import math
import sys
import numpy as np

# --- Constants ---
DEFAULT_MODEL_NAME = "/home/rosegarden/ros2_ws/yolo11l.pt" # Ensure this path is correct
DEFAULT_KINECT_IMAGE_TOPIC = '/kinect/kinect_depth/image_raw/compressed' # Assuming this is compressed RGB
DEFAULT_NODE_NAME = 'yolo_kinect_tracker_node'
DEFAULT_YOLO_PUBLISH_TOPIC = '/yolo_detections'

DEFAULT_TILT_GOAL_TOPIC = '/kinect_tilt_angle_goal'
DEFAULT_PAN_GOAL_TOPIC = '/kinect_pan_angle_goal' # New topic for pan

DEFAULT_TIMER_RATE_HZ = 10.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DISPLAY_WINDOW = True
RESIZE_WIDTH = 320*2
RESIZE_HEIGHT = 240*2
TARGET_CLASS_NAME = "person"

# --- Tracking Box Parameters ---
# Defines the central target box as a fraction of image width/height
DEFAULT_TARGET_BOX_WIDTH_FRAC = 0.2  # e.g., person should occupy 30% of image width
DEFAULT_TARGET_BOX_HEIGHT_FRAC = 0.2 # e.g., person should occupy 60% of image height

# --- Field of View (FoV) Parameters - CRITICAL TO SET ACCURATELY ---
DEFAULT_KINECT_HORIZONTAL_FOV_RADIANS = math.radians(57.0) # EXAMPLE - PLEASE VERIFY
DEFAULT_KINECT_VERTICAL_FOV_RADIANS = math.radians(43.0)   # EXAMPLE - PLEASE VERIFY

# --- Pan Control Parameters ---
DEFAULT_PAN_P_GAIN = 0.4  # Proportional gain for pan control
DEFAULT_MAX_PAN_INCREMENT_RAD_PER_FRAME = math.radians(2.0) # Max pan speed per frame
DEFAULT_MIN_PAN_LIMIT_RAD = math.radians(-80.0) # Match URDF kinect_pan_joint lower limit (e.g. -pi/2)
DEFAULT_MAX_PAN_LIMIT_RAD = math.radians(80.0)  # Match URDF kinect_pan_joint upper limit (e.g. pi/2)

# --- Tilt Control Parameters ---
DEFAULT_TILT_P_GAIN = 0.4  # Proportional gain for tilt control
DEFAULT_MAX_TILT_INCREMENT_RAD_PER_FRAME = math.radians(1.5) # Max tilt speed per frame
DEFAULT_MIN_TILT_LIMIT_RAD = math.radians(-40.0) # Match URDF kinect_tilt_joint lower limit
DEFAULT_MAX_TILT_LIMIT_RAD = math.radians(25.0)  # Match URDF kinect_tilt_joint upper limit


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
        self.declare_parameter('kinect_image_topic', DEFAULT_KINECT_IMAGE_TOPIC)

        # Tilt parameters
        self.declare_parameter('tilt_goal_topic', DEFAULT_TILT_GOAL_TOPIC)
        self.declare_parameter('kinect_vertical_fov_radians', DEFAULT_KINECT_VERTICAL_FOV_RADIANS)
        self.declare_parameter('tilt_p_gain', DEFAULT_TILT_P_GAIN)
        self.declare_parameter('max_tilt_increment_rad_per_frame', DEFAULT_MAX_TILT_INCREMENT_RAD_PER_FRAME)
        self.declare_parameter('min_tilt_limit_rad', DEFAULT_MIN_TILT_LIMIT_RAD)
        self.declare_parameter('max_tilt_limit_rad', DEFAULT_MAX_TILT_LIMIT_RAD)

        # Pan parameters
        self.declare_parameter('pan_goal_topic', DEFAULT_PAN_GOAL_TOPIC)
        self.declare_parameter('kinect_horizontal_fov_radians', DEFAULT_KINECT_HORIZONTAL_FOV_RADIANS)
        self.declare_parameter('pan_p_gain', DEFAULT_PAN_P_GAIN)
        self.declare_parameter('max_pan_increment_rad_per_frame', DEFAULT_MAX_PAN_INCREMENT_RAD_PER_FRAME)
        self.declare_parameter('min_pan_limit_rad', DEFAULT_MIN_PAN_LIMIT_RAD)
        self.declare_parameter('max_pan_limit_rad', DEFAULT_MAX_PAN_LIMIT_RAD)

        # Target box parameters
        self.declare_parameter('target_box_width_frac', DEFAULT_TARGET_BOX_WIDTH_FRAC)
        self.declare_parameter('target_box_height_frac', DEFAULT_TARGET_BOX_HEIGHT_FRAC)


        self.add_on_set_parameters_callback(self.parameters_callback)
        self._load_parameters()

        self.get_logger().info(f"--- Initializing {DEFAULT_NODE_NAME} ---")

        self.latest_compressed_frame_data = None
        self.frame_lock = threading.Lock()
        self.current_commanded_tilt_angle = 0.0
        self.current_commanded_pan_angle = 0.0 # New state for pan

        # --- ROS Publishers ---
        self.yolo_publisher = self.create_publisher(String, self.yolo_publish_topic, 10)
        self.tilt_goal_publisher = self.create_publisher(Float64, self.tilt_goal_topic, 10)
        self.pan_goal_publisher = self.create_publisher(Float64, self.pan_goal_topic, 10) # New publisher for pan

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

        self.image_subscriber = self.create_subscription(
            CompressedImage,
            self.kinect_image_topic,
            self.compressed_image_callback,
            10
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
        self.kinect_image_topic = self.get_parameter('kinect_image_topic').get_parameter_value().string_value

        self.tilt_goal_topic = self.get_parameter('tilt_goal_topic').get_parameter_value().string_value
        self.kinect_vertical_fov_radians = self.get_parameter('kinect_vertical_fov_radians').get_parameter_value().double_value
        self.tilt_p_gain = self.get_parameter('tilt_p_gain').get_parameter_value().double_value
        self.max_tilt_increment_rad_per_frame = self.get_parameter('max_tilt_increment_rad_per_frame').get_parameter_value().double_value
        self.min_tilt_limit_rad = self.get_parameter('min_tilt_limit_rad').get_parameter_value().double_value
        self.max_tilt_limit_rad = self.get_parameter('max_tilt_limit_rad').get_parameter_value().double_value

        self.pan_goal_topic = self.get_parameter('pan_goal_topic').get_parameter_value().string_value
        self.kinect_horizontal_fov_radians = self.get_parameter('kinect_horizontal_fov_radians').get_parameter_value().double_value
        self.pan_p_gain = self.get_parameter('pan_p_gain').get_parameter_value().double_value
        self.max_pan_increment_rad_per_frame = self.get_parameter('max_pan_increment_rad_per_frame').get_parameter_value().double_value
        self.min_pan_limit_rad = self.get_parameter('min_pan_limit_rad').get_parameter_value().double_value
        self.max_pan_limit_rad = self.get_parameter('max_pan_limit_rad').get_parameter_value().double_value

        self.target_box_width_frac = self.get_parameter('target_box_width_frac').get_parameter_value().double_value
        self.target_box_height_frac = self.get_parameter('target_box_height_frac').get_parameter_value().double_value

    def parameters_callback(self, params):
        success = True
        require_timer_recreate = False
        for param in params:
            try:
                if param.name == 'timer_rate_hz':
                    self.timer_rate_hz = param.value.double_value
                    require_timer_recreate = True
                elif param.name == 'confidence_threshold':
                    self.confidence_threshold = param.value.double_value
                elif param.name == 'tilt_p_gain':
                    self.tilt_p_gain = param.value.double_value
                elif param.name == 'pan_p_gain':
                    self.pan_p_gain = param.value.double_value
                # Add others if you want them to be dynamically reconfigurable
            except Exception as e:
                self.get_logger().error(f"Error processing parameter update for {param.name}: {e}")
                success = False
        if require_timer_recreate: self._create_processing_timer()
        return SetParametersResult(successful=success)

    def _create_processing_timer(self):
        if hasattr(self, 'timer') and self.timer is not None and not self.timer.is_canceled():
            self.timer.cancel()
        timer_period = 1.0 / self.timer_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Processing timer (re)created with period: {timer_period:.3f} s ({self.timer_rate_hz:.1f} Hz)")

    def compressed_image_callback(self, msg: CompressedImage):
        with self.frame_lock:
            self.latest_compressed_frame_data = msg.data

    def timer_callback(self):
        compressed_data = None
        with self.frame_lock:
            if self.latest_compressed_frame_data is not None:
                compressed_data = self.latest_compressed_frame_data
                self.latest_compressed_frame_data = None

        if compressed_data is None: return

        frame_to_process = None
        try:
            np_arr = np.frombuffer(compressed_data, np.uint8)
            frame_to_process = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame_to_process is None:
                self.get_logger().warn("cv2.imdecode failed.")
                return
        except Exception as e:
            self.get_logger().error(f"Error decompressing image: {e}\n{traceback.format_exc()}")
            return

        try:
            frame_display = cv2.resize(frame_to_process, (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.get_logger().warn(f"Error resizing frame: {e}. Skipping frame.")
            return

        results = None
        track_args = {
            "persist": True, "verbose": False, "conf": self.confidence_threshold,
            "device": "cuda",
            "classes": [self.person_class_index] if self.person_class_index is not None else None
        }
        try:
            frame_for_yolo = frame_display.copy()
            results = self.model.track(frame_for_yolo, **track_args)
        except Exception as e:
            self.get_logger().error(f"Error during model tracking: {e}\n{traceback.format_exc()}")
            results = None

        person_detection_data_list = []
        target_person_x_center = None # For Pan
        target_person_y_center = None # For Tilt
        target_person_detected = False
        best_box_area = -1

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                    is_person = (cls_id == self.person_class_index) if self.person_class_index is not None else False
                    if not is_person: continue
                    
                    target_person_detected = True
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    current_area = (x2 - x1) * (y2 - y1)
                    if current_area > best_box_area: # Select largest person
                        best_box_area = current_area
                        target_person_x_center = (x1 + x2) / 2.0
                        target_person_y_center = (y1 + y2) / 2.0
                    
                    # (Your existing yolo detection string publishing logic)
                    conf = box.conf[0].item() if box.conf is not None else 0.0
                    track_id = int(box.id[0].item()) if box.id is not None and box.id[0] is not None and not math.isnan(box.id[0].item()) else -1
                    class_name = TARGET_CLASS_NAME
                    detection_str = f"TrackID: {track_id}, Class: {class_name}, Conf: {conf:.2f}, BBox: [{x1},{y1},{x2},{y2}]"
                    person_detection_data_list.append(detection_str)

                    if self.display_window:
                        label = f"ID:{track_id} {class_name} {conf:.2f}"
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(frame_display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
                except Exception as e:
                    self.get_logger().warn(f"Error processing a detection box: {e}")
                    continue
        
        if person_detection_data_list:
            yolo_msg = String(); yolo_msg.data = "\n".join(person_detection_data_list)
            self.yolo_publisher.publish(yolo_msg)

        # --- Target Box Definition ---
        target_box_w = self.resize_width * self.target_box_width_frac
        target_box_h = self.resize_height * self.target_box_height_frac
        target_box_x1 = (self.resize_width - target_box_w) / 2.0
        target_box_y1 = (self.resize_height - target_box_h) / 2.0
        target_box_x2 = target_box_x1 + target_box_w
        target_box_y2 = target_box_y1 + target_box_h

        if target_person_detected and target_person_x_center is not None and target_person_y_center is not None:
            # --- Pan Control ---
            # Error: positive if person is to the RIGHT of target box center, negative if to the LEFT
            target_box_center_x = (target_box_x1 + target_box_x2) / 2.0
            pixel_error_x = target_person_x_center - target_box_center_x
            
            # Only pan if person is outside the horizontal bounds of the target box
            if target_person_x_center < target_box_x1 or target_person_x_center > target_box_x2:
                normalized_pixel_error_x = pixel_error_x / (self.resize_width / 2.0)
                pan_adjustment_rad = -self.pan_p_gain * normalized_pixel_error_x
                pan_adjustment_rad = np.clip(pan_adjustment_rad,
                                             -self.max_pan_increment_rad_per_frame,
                                             self.max_pan_increment_rad_per_frame)
                new_target_pan_rad = self.current_commanded_pan_angle + pan_adjustment_rad
                new_target_pan_rad = np.clip(new_target_pan_rad,
                                               self.min_pan_limit_rad,
                                               self.max_pan_limit_rad)
                pan_msg = Float64(); pan_msg.data = new_target_pan_rad
                self.pan_goal_publisher.publish(pan_msg)
                self.current_commanded_pan_angle = new_target_pan_rad
                self.get_logger().debug(f"Pan TargetX: {target_person_x_center:.1f}, PxErrX: {pixel_error_x:.1f}, PanCmd: {math.degrees(new_target_pan_rad):.1f}deg")

            # --- Tilt Control ---
            # Error: positive if person is BELOW target box center, negative if ABOVE
            target_box_center_y = (target_box_y1 + target_box_y2) / 2.0
            pixel_error_y = target_person_y_center - target_box_center_y

            # Only tilt if person is outside the vertical bounds of the target box
            if target_person_y_center < target_box_y1 or target_person_y_center > target_box_y2:
                normalized_pixel_error_y = pixel_error_y / (self.resize_height / 2.0)
                tilt_adjustment_rad = self.tilt_p_gain * normalized_pixel_error_y
                tilt_adjustment_rad = np.clip(tilt_adjustment_rad,
                                              -self.max_tilt_increment_rad_per_frame,
                                              self.max_tilt_increment_rad_per_frame)
                new_target_tilt_rad = self.current_commanded_tilt_angle + tilt_adjustment_rad
                new_target_tilt_rad = np.clip(new_target_tilt_rad,
                                               self.min_tilt_limit_rad,
                                               self.max_tilt_limit_rad)
                tilt_msg = Float64(); tilt_msg.data = new_target_tilt_rad
                self.tilt_goal_publisher.publish(tilt_msg)
                self.current_commanded_tilt_angle = new_target_tilt_rad
                self.get_logger().debug(f"Tilt TargetY: {target_person_y_center:.1f}, PxErrY: {pixel_error_y:.1f}, TiltCmd: {math.degrees(new_target_tilt_rad):.1f}deg")
        else:
            # No person, do nothing (hold current pan/tilt)
            pass

        if self.display_window:
            # Draw the target box
            cv2.rectangle(frame_display,
                          (int(target_box_x1), int(target_box_y1)),
                          (int(target_box_x2), int(target_box_y2)),
                          (255, 0, 0), 1) # Blue box

            # Your FPS display logic
            self.frame_count_display += 1
            elapsed_time = time.time() - self.start_time_display
            if elapsed_time >= 1.0:
                fps = self.frame_count_display / elapsed_time
                self.current_fps_display = f"Proc FPS: {fps:.1f}"
                self.start_time_display = time.time()
                self.frame_count_display = 0
            if hasattr(self, 'current_fps_display'):
                cv2.putText(frame_display, self.current_fps_display, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            
            # Draw image center
            # cv2.circle(frame_display, (self.resize_width // 2, self.resize_height // 2), 3, (0,0,255), -1) # Red dot

            try:
                cv2.imshow('YOLO Kinect Tracker', frame_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("Quit key pressed. Shutting down.")
                    self.safe_shutdown("User requested quit")
            except Exception as e:
                self.get_logger().error(f"Error during CV2 display: {e}")

    def safe_shutdown(self, reason="Unknown"):
        if not rclpy.ok(): return
        self.get_logger().warn(f"Initiating shutdown... Reason: {reason}")
        if hasattr(self, 'timer') and self.timer is not None and not self.timer.is_canceled():
            self.timer.cancel()
        if rclpy.ok():
            try: self.create_timer(0.2, lambda: rclpy.try_shutdown(context=rclpy.get_default_context()))
            except Exception: pass # Ignore if already shutting down
            finally:
                if rclpy.ok(): rclpy.try_shutdown(context=rclpy.get_default_context())


    def destroy_node(self):
        self.get_logger().info("Destroying node and releasing resources...")
        if hasattr(self, 'timer') and self.timer is not None and not self.timer.is_canceled():
            self.timer.cancel()
        try:
            if self.display_window and cv2.getWindowProperty('YOLO Kinect Tracker', cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyAllWindows()
        except: pass # Ignore errors on destroy
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
        if init_success:
            rclpy.spin(node)
        else:
            node.get_logger().error("Node initialization failed. Shutting down early.")
            exit_code = 1
            if node and hasattr(node, 'timer') and node.timer is not None: node.timer.cancel()
    except KeyboardInterrupt:
        if node: node.safe_shutdown("KeyboardInterrupt")
    except Exception as e:
        exit_code = 1
        if node:
            node.get_logger().fatal(f"Unhandled exception in main loop: {e}\n{traceback.format_exc()}")
            node.safe_shutdown("Unhandled exception")
        else:
            print(f"Unhandled exception before node init: {e}\n{traceback.format_exc()}", file=sys.stderr)
        time.sleep(0.5)
    finally:
        if node is not None and hasattr(node, 'destroy_node'):
            node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        # sys.exit(exit_code) # Optional

if __name__ == '__main__':
    main()