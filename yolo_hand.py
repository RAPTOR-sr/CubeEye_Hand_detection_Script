import queue
import sys
import CubeEye as cu
import cv2
import numpy as np
import ctypes
import open3d as o3d
import threading
import os
import time
import pythoncom
from ultralytics import YOLO

# Initialize COM library for the current thread - Changed threading model
pythoncom.CoInitializeEx(pythoncom.COINIT_APARTMENTTHREADED)

# ==== Global variables ====
DATASET_ROOT = "Dataset"
frame_counter = 0
recording = False
CLASS_ID = ""
USER_ID = ""

# Hand detection and point cloud parameters
HAND_DEPTH_MIN = 0.1  # Minimum depth in meters
HAND_DEPTH_MAX = 0.8  # Maximum depth in meters
HAND_BOX_PADDING = 20  # Padding pixels around detected hand
HAND_OUTLIER_NEIGHBORS = 20  # Number of neighbors for outlier removal
HAND_OUTLIER_STD = 2.0  # Standard deviation threshold for outlier removal

# ==== Queues ====
frame_sets_queue = queue.Queue(maxsize=5)

current_frame_set = {
    'depth': None,
    'amplitude': None,
    'point_cloud': None,
    'timestamp': None
}

# ==== Open3D setup ====
_o3d_vis_start = False
_hand_vis_start = False  # Flag for hand visualization

# ==== YOLO Hand detector setup ====
yolo_model = YOLO("D:\\Projects\\CubeEye\\best.pt")  # Update with your model path
yolo_model.to("cuda")  # Use GPU for inference, change to "cpu" if needed
CONF_THRESHOLD = 0.5  # Confidence threshold for detections

def create_dataset_dirs():
    for dtype in ["depth", "amplitude", "point_cloud"]:
        dir_path = os.path.join(DATASET_ROOT, dtype, CLASS_ID, USER_ID)
        os.makedirs(dir_path, exist_ok=True)

def o3d_vis_escape_key_callback(vis):
    global _o3d_vis_start
    _o3d_vis_start = False
    
def o3d_r_key_callback(vis):
    global recording, frame_counter, CLASS_ID, USER_ID
    
    if recording:
        # Stop recording
        recording = False
        print(f"Recording stopped. Captured {frame_counter} frames.")
    else:
        # Use terminal for text input
        class_id = input("Enter Class ID (e.g., class_01, press Enter for auto-generated): ")
        user_id = input("Enter User ID (e.g., user_01, press Enter for auto-generated): ")
        
        # Set the global variables
        if class_id:
            CLASS_ID = class_id
        else:
            CLASS_ID = f"class_{time.strftime('%Y%m%d_%H%M%S')}"
            
        if user_id:
            USER_ID = user_id
        else:
            USER_ID = f"user_{time.strftime('%Y%m%d_%H%M%S')}"
        
        create_dataset_dirs()
        frame_counter = 0
        recording = True
        print(f"Starting recording for Class ID: {CLASS_ID}, User ID: {USER_ID}")
    
    return False  # Don't close visualizer

# ==== CubeEye Sink Callback ====
class _CubeEyePythonSink(cu.Sink):
    def __init__(self):
        cu.Sink.__init__(self)

    def name(self):
        return "_CubeEyePythonSink"

    def onCubeEyeCameraState(self, name, serial_number, uri, state):
        # print(f"Camera ({name}/{serial_number}) state: {state}")
        pass

    def onCubeEyeCameraError(self, name, serial_number, uri, error):
        # print(f"Camera ({name}/{serial_number}) error: {error}")
        pass

    def onCubeEyeFrameList(self, name, serial_number, uri, frames):
        global frame_sets_queue, current_frame_set

        if frames is not None:
            timestamp = time.time()
            frame_set_updated = False

            for _frame in frames:
                if _frame.isBasicFrame():
                    if cu.DataType_U16 == _frame.dataType():
                        _u16_frame = cu.frame_cast_basic16u(_frame)
                        _u16_data_ptr = ctypes.c_uint16 * _u16_frame.dataSize()
                        _u16_data_ptr = _u16_data_ptr.from_address(int(_u16_frame.dataPtr()))
                        _u16_data_arr = np.ctypeslib.as_array(_u16_data_ptr)

                        if cu.FrameType_Depth == _u16_frame.frameType():
                            _u8_bgr = np.zeros((_frame.height(), _frame.width(), 3), dtype=np.uint8)
                            cu.convert2bgr(_u16_data_arr, _u8_bgr)
                            current_frame_set['depth'] = _u8_bgr
                            current_frame_set['timestamp'] = timestamp
                            frame_set_updated = True

                        elif cu.FrameType_Amplitude == _u16_frame.frameType():
                            _u8_gray = np.zeros((_frame.height(), _frame.width()), dtype=np.uint8)
                            cu.convert2gray(_u16_data_arr, _u8_gray)
                            current_frame_set['amplitude'] = _u8_gray
                            current_frame_set['timestamp'] = timestamp
                            frame_set_updated = True

                elif cu.FrameType_PointCloud == _frame.frameType():
                    if cu.DataType_F32 == _frame.dataType():
                        _f32_frame = cu.frame_cast_pcl32f(_frame)

                        _x_ptr = ctypes.c_float * _f32_frame.dataXsize()
                        _y_ptr = ctypes.c_float * _f32_frame.dataYsize()
                        _z_ptr = ctypes.c_float * _f32_frame.dataZsize()

                        _x_arr = np.ctypeslib.as_array(_x_ptr.from_address(int(_f32_frame.dataXptr())))
                        _y_arr = np.ctypeslib.as_array(_y_ptr.from_address(int(_f32_frame.dataYptr())))
                        _z_arr = np.ctypeslib.as_array(_z_ptr.from_address(int(_f32_frame.dataZptr())))

                        current_frame_set['point_cloud'] = (
                            np.array(_x_arr),
                            np.array(_y_arr),
                            np.array(_z_arr)
                        )
                        current_frame_set['timestamp'] = timestamp
                        frame_set_updated = True

            if (frame_set_updated and 
                current_frame_set['depth'] is not None and 
                current_frame_set['amplitude'] is not None and 
                current_frame_set['point_cloud'] is not None):

                if frame_sets_queue.full():
                    try:
                        frame_sets_queue.get_nowait()
                    except queue.Empty:
                        pass

                frame_sets_queue.put(current_frame_set.copy())

                current_frame_set = {
                    'depth': None,
                    'amplitude': None,
                    'point_cloud': None,
                    'timestamp': None
                }

# ==== Main ====
if __name__ == "__main__":
    print("Starting CubeEye capture...")
    print("Press 'R' to start/stop recording, ESC to exit")
    print("When recording starts, position your hand in the frame for detection")

    cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Amplitude', cv2.WINDOW_AUTOSIZE)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    _o3d_vis = o3d.visualization.VisualizerWithKeyCallback()
    _o3d_vis.create_window("CubeEye PCL", 640, 480)
    _o3d_vis.get_render_option().background_color = np.asarray([0, 0, 0])
    _o3d_vis.get_render_option().point_size = 1.0
    _o3d_vis.register_key_callback(256, o3d_vis_escape_key_callback)  # ESC key
    _o3d_vis.register_key_callback(ord('R'), o3d_r_key_callback)      # R key
    
    # Create a separate visualizer for hand point cloud
    _hand_vis = o3d.visualization.Visualizer()
    _hand_vis.create_window("Hand Point Cloud", 400, 400)
    _hand_vis.get_render_option().background_color = np.asarray([0, 0, 0])
    _hand_vis.get_render_option().point_size = 2.0  # Larger points for better visibility
    _hand_vis_start = True
    _hand_pcd = o3d.geometry.PointCloud()
    _first_hand_frame = True
    
    _source_list = cu.search_camera_source()
    if _source_list is None or _source_list.size() <= 0:
        print("No CubeEye camera found.")
        sys.exit(1)

    _camera = cu.create_camera(_source_list[0])
    if _camera is None:
        print("Failed to create camera.")
        sys.exit(1)

    _sink = _CubeEyePythonSink()
    _camera.addSink(_sink)
    _camera.setProperty(cu.make_property_8u("framerate", 30))
    if _camera.prepare() != cu.Result_Success:
        print("Camera preparation failed.")
        cu.destroy_camera(_camera)
        sys.exit(1)

    if _camera.run(38) != cu.Result_Success:
        print("Camera run failed.")
        cu.destroy_camera(_camera)
        sys.exit(1)

    _o3d_vis_start = True
    _first_frame = True
    _pcd = o3d.geometry.PointCloud()

    # Main loop - keep running until ESC is pressed
    while _o3d_vis_start:
        if not frame_sets_queue.empty():
            frame_set = frame_sets_queue.get_nowait()
            depth_img = frame_set['depth']
            amplitude_img = frame_set['amplitude']
            pcl_data = frame_set['point_cloud']

            # Process point cloud for visualization
            x, y, z = pcl_data
            xyz = np.transpose(np.vstack((x, y, z)))
            _pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # Apply rotation
            R_x = _pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
            _pcd.rotate(R_x, center=[0, 0, 0])

            # Always convert amplitude to BGR for hand detection
            display_amp = cv2.cvtColor(amplitude_img, cv2.COLOR_GRAY2BGR)
            
            # Replace MediaPipe hand detection with YOLO
            results = yolo_model(display_amp, conf=CONF_THRESHOLD)
            
            # Prepare depth image for display
            display_depth = depth_img.copy()
            
            # Draw bounding boxes on display_amp (optional)
            if results and len(results) > 0:
                # Annotate the amplitude image
                annotated_amp = results[0].plot()
                display_amp = annotated_amp
            
            # If hand detected, show the bounding box on depth image and update hand point cloud
            hands_detected = False
            if results and len(results[0]) > 0 and len(results[0].boxes) > 0:
                hands_detected = True
                # Get the first detected hand with highest confidence
                box = results[0].boxes[0]
                
                # Extract bounding box coordinates (YOLO uses x1,y1,x2,y2 format)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_crop, y_crop = int(x1), int(y1)
                w_crop, h_crop = int(x2 - x1), int(y2 - y1)
                
                # Add padding to bounding box to capture full hand
                x_crop = max(0, x_crop - HAND_BOX_PADDING)
                y_crop = max(0, y_crop - HAND_BOX_PADDING)
                w_crop += HAND_BOX_PADDING * 2
                h_crop += HAND_BOX_PADDING * 2
                
                cv2.rectangle(display_depth, (x_crop, y_crop), 
                             (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)
                
                # Make sure the crop is within image boundaries
                h_max, w_max = amplitude_img.shape
                x_crop = max(0, x_crop)
                y_crop = max(0, y_crop)
                if x_crop + w_crop > w_max:
                    w_crop = w_max - x_crop
                if y_crop + h_crop > h_max:
                    h_crop = h_max - y_crop
                
                # Reshape point cloud arrays to match image dimensions
                pcl_x_reshaped = np.array(x).reshape(amplitude_img.shape)
                pcl_y_reshaped = np.array(y).reshape(amplitude_img.shape)
                pcl_z_reshaped = np.array(z).reshape(amplitude_img.shape)
                
                # Extract hand depth information (center point of detection)
                hand_center_x = int((x1 + x2) / 2)
                hand_center_y = int((y1 + y2) / 2)
                hand_depth_center = None
                
                if (0 <= hand_center_y < pcl_z_reshaped.shape[0] and 
                    0 <= hand_center_x < pcl_z_reshaped.shape[1]):
                    hand_depth_center = pcl_z_reshaped[hand_center_y, hand_center_x]
                
                # Crop the point cloud data using the bounding box
                cropped_pcl_x = pcl_x_reshaped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop].flatten()
                cropped_pcl_y = pcl_y_reshaped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop].flatten()
                cropped_pcl_z = pcl_z_reshaped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop].flatten()
                
                # Create a new Open3D point cloud with the cropped points
                cropped_xyz = np.vstack((cropped_pcl_x, cropped_pcl_y, cropped_pcl_z)).T
                
                # Filter out any invalid points (inf, nan, etc.)
                valid_indices = np.logical_and.reduce([
                    np.isfinite(cropped_xyz[:, 0]),
                    np.isfinite(cropped_xyz[:, 1]),
                    np.isfinite(cropped_xyz[:, 2])
                ])
                cropped_xyz = cropped_xyz[valid_indices]

                # Apply depth filtering
                if hand_depth_center is not None and np.isfinite(hand_depth_center):
                    # Dynamic depth thresholding based on detected hand depth
                    min_depth = hand_depth_center - 0.15
                    max_depth = hand_depth_center + 0.15
                else:
                    # Static depth thresholding
                    min_depth = HAND_DEPTH_MIN
                    max_depth = HAND_DEPTH_MAX
                
                depth_indices = np.logical_and(
                    cropped_xyz[:, 2] >= min_depth,
                    cropped_xyz[:, 2] <= max_depth
                )
                
                # Combine all filtering conditions
                cropped_xyz = cropped_xyz[depth_indices]
                
                # Update hand point cloud with current raw points
                _hand_pcd.points = o3d.utility.Vector3dVector(cropped_xyz)
                
                # Apply statistical outlier removal if enough points
                if len(_hand_pcd.points) > HAND_OUTLIER_NEIGHBORS:
                    try:
                        # ================================================================= #
                        # FIXED: For live view, do not re-assign the _hand_pcd object.      #
                        # REASON: Re-assigning breaks the reference the visualizer holds.   #
                        # Instead, run the filter and copy the resulting points back        #
                        # to the original _hand_pcd object.                                 #
                        # ================================================================= #
                        cleaned_pcd, ind = _hand_pcd.remove_statistical_outlier(
                            nb_neighbors=HAND_OUTLIER_NEIGHBORS,
                            std_ratio=HAND_OUTLIER_STD
                        )
                        _hand_pcd.points = cleaned_pcd.points
                    except Exception as e:
                        print(f"Statistical outlier removal failed: {e}")
                
                # Apply the same rotation as the original point cloud
                R_x_hand = _hand_pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
                _hand_pcd.rotate(R_x_hand, center=[0, 0, 0])
                
                # Colorize the hand point cloud for better visualization
                _hand_pcd.paint_uniform_color([0.8, 0.2, 0.2])  # Reddish color
                
                # Update the hand visualization
                if _first_hand_frame:
                    _hand_vis.add_geometry(_hand_pcd)
                    _first_hand_frame = False
                else:
                    _hand_vis.update_geometry(_hand_pcd)
                
                _hand_vis.poll_events()
                _hand_vis.update_renderer()
            
            # Save data if recording
            if recording and frame_counter < 15:  # Limit to 15 frames per recording
                if hands_detected:  # If hand detected by YOLO
                    # Extract box again for clarity
                    box = results[0].boxes[0]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x_crop, y_crop = int(x1), int(y1)
                    w_crop, h_crop = int(x2 - x1), int(y2 - y1)
                    
                    # Add padding to bounding box
                    x_crop = max(0, x_crop - HAND_BOX_PADDING)
                    y_crop = max(0, y_crop - HAND_BOX_PADDING)
                    w_crop += HAND_BOX_PADDING * 2
                    h_crop += HAND_BOX_PADDING * 2
                    
                    # Crop amplitude and depth images
                    cropped_amplitude = amplitude_img[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
                    cropped_depth = depth_img[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop]
                    
                    # Crop the point cloud data
                    # Point cloud arrays match the image dimensions when reshaped
                    pcl_x_reshaped = np.array(x).reshape(amplitude_img.shape)
                    pcl_y_reshaped = np.array(y).reshape(amplitude_img.shape)
                    pcl_z_reshaped = np.array(z).reshape(amplitude_img.shape)
                    
                    # Crop the point cloud data using the same bounding box
                    cropped_pcl_x = pcl_x_reshaped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop].flatten()
                    cropped_pcl_y = pcl_y_reshaped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop].flatten()
                    cropped_pcl_z = pcl_z_reshaped[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop].flatten()
                    
                    # Create a new Open3D point cloud with the cropped points
                    cropped_xyz = np.vstack((cropped_pcl_x, cropped_pcl_y, cropped_pcl_z)).T
                    cropped_pcd = o3d.geometry.PointCloud()
                    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_xyz)
                    
                    # Apply the same rotation as the original point cloud
                    R_x_crop = cropped_pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
                    cropped_pcd.rotate(R_x_crop, center=[0, 0, 0])
                    
                    # Save cropped data
                    depth_path = os.path.join(DATASET_ROOT, "depth", CLASS_ID, USER_ID, f"frame_{frame_counter:04d}.png")
                    amplitude_path = os.path.join(DATASET_ROOT, "amplitude", CLASS_ID, USER_ID, f"frame_{frame_counter:04d}.png")
                    cv2.imwrite(depth_path, cropped_depth)
                    cv2.imwrite(amplitude_path, cropped_amplitude)
                    
                    # Save cropped point cloud
                    ply_path = os.path.join(DATASET_ROOT, "point_cloud", CLASS_ID, USER_ID, f"frame_{frame_counter:04d}.ply")
                    o3d.io.write_point_cloud(ply_path, cropped_pcd)
                    
                    frame_counter += 1
                    print(f"Saved frame {frame_counter}/15 with hand detection")
                    
                    # Stop recording if we've reached 15 frames
                    if frame_counter >= 15:
                        recording = False
                        print(f"Recording complete. Captured {frame_counter} frames with hand detection.")
                else:
                    print("No hand detected in this frame, skipping...")
            
            # Visualize
            if _first_frame:
                _first_frame = False
                _o3d_vis.add_geometry(_pcd)
            else:
                _o3d_vis.update_geometry(_pcd)

            _o3d_vis.poll_events()
            _o3d_vis.update_renderer()
            
            # Add recording indicator if recording
            if recording:
                # Add recording text to both displays
                cv2.putText(display_depth, "RECORDING", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_amp, "RECORDING", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Always show the processed images with hand detection
            cv2.imshow("Depth", display_depth)
            cv2.imshow("Amplitude", display_amp)

        if cv2.waitKey(1) == 27:  # ESC key
            _o3d_vis_start = False
            break

    print("Stopping camera...")
    _camera.stop()
    cu.destroy_camera(_camera)
    _o3d_vis.destroy_window()
    _hand_vis.destroy_window()  # Clean up the hand visualization window
    cv2.destroyAllWindows()
    print("Bye bye~")
    sys.exit(0)
# End of the script
# Note: This script is designed to run with the CubeEye SDK and Open3D library.