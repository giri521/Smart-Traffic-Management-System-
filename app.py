"""
AI Smart Traffic Management System - Enhanced Version with History & Analytics
Flask-based application with YOLOv8 for vehicle detection and data persistence
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import threading
import time
from collections import defaultdict
import logging
import os
from datetime import datetime
import atexit
import json
import hashlib
from cryptography.fernet import Fernet
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with development configuration
app = Flask(__name__)

# Enable template auto-reloading
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# API Keys for encryption
API_KEYS = [
    "728E339A-93A2-46C8-B82D-7601700F3638",
    "8DC5F86F-F938-4C3A-B9F2-B16D3F964117"
]

# Create encryption key from API keys
def create_encryption_key():
    """Create Fernet encryption key from API keys"""
    combined = "".join(API_KEYS).encode()
    key = base64.urlsafe_b64encode(hashlib.sha256(combined).digest())
    return key

ENCRYPTION_KEY = create_encryption_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

# Global configuration
FRAME_SKIP = 2
ROAD_MASK_UPDATE_FREQ = 10
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Data storage configuration
DATA_STORAGE_INTERVAL = 300  # 5 minutes in seconds

# Global state management with thread safety
class TrafficState:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera = None
        self.processing_active = False
        self.current_frame = None
        self.processed_frame = None
        self.frame_count = 0
        self.current_mode = "STANDBY"  # STANDBY, VIDEO, CAMERA, IMAGE
        
        # Vehicle counts - updated by detection
        self.total_vehicles = 0
        self.vehicle_counts = defaultdict(int)
        self.density_level = "LOW"
        
        # Signal state - continuous independent thread
        self.signal_state = "RED"
        self.signal_timer = 20  # Current remaining time
        self.base_duration = 20  # Base duration for current signal
        self.last_switch_time = time.time()
        self.green_start_time = None
        self.min_green_time = 10
        
        # Emergency detection
        self.emergency_detected = False
        self.emergency_type = None
        self.emergency_timeout = 0
        self.emergency_vehicle_count = 0
        
        # Detection models
        self.det_model = None
        self.seg_model = None
        self.road_mask = None
        self.last_road_update = 0
        
        # Processing thread for video/camera
        self.processing_thread = None
        # Signal controller thread (runs forever)
        self.signal_thread = None
        self.signal_running = True
        
        # Data storage
        self.last_storage_time = time.time()
        self.current_session_data = []
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

traffic_state = TrafficState()

def load_models():
    """Load YOLO models with error handling"""
    try:
        traffic_state.det_model = YOLO('yolov8n.pt')
        traffic_state.seg_model = YOLO('yolov8n-seg.pt')
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def encrypt_data(data):
    """Encrypt sensitive data using API keys"""
    try:
        json_str = json.dumps(data)
        encrypted = cipher_suite.encrypt(json_str.encode())
        return base64.b64encode(encrypted).decode()
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return None

def decrypt_data(encrypted_data):
    """Decrypt data using API keys"""
    try:
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted = cipher_suite.decrypt(encrypted_bytes)
        return json.loads(decrypted.decode())
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return None

def store_traffic_data():
    """Store current traffic data with encryption"""
    with traffic_state.lock:
        current_time = time.time()
        
        # Check if it's time to store data
        if current_time - traffic_state.last_storage_time >= DATA_STORAGE_INTERVAL:
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'total_vehicles': traffic_state.total_vehicles,
                'vehicle_counts': dict(traffic_state.vehicle_counts),
                'density_level': traffic_state.density_level,
                'signal_state': traffic_state.signal_state,
                'signal_timer': traffic_state.signal_timer,
                'base_duration': traffic_state.base_duration,
                'emergency_detected': traffic_state.emergency_detected,
                'emergency_type': traffic_state.emergency_type,
                'session_id': traffic_state.session_id
            }
            
            traffic_state.current_session_data.append(data_point)
            traffic_state.last_storage_time = current_time
            logger.info(f"Traffic data stored at {data_point['timestamp']}")
            
            # Keep only last 100 data points in memory
            if len(traffic_state.current_session_data) > 100:
                traffic_state.current_session_data = traffic_state.current_session_data[-100:]

def get_traffic_history():
    """Get encrypted traffic history"""
    with traffic_state.lock:
        return encrypt_data(traffic_state.current_session_data)

def reset_system():
    """
    Reset system but preserve signal state
    Signal continues running independently
    """
    with traffic_state.lock:
        logger.info("Resetting system (signal continues)...")
        
        # Stop processing but keep signal running
        traffic_state.processing_active = False
        time.sleep(0.5)
        
        # Release camera/video capture
        if traffic_state.camera:
            traffic_state.camera.release()
            traffic_state.camera = None
        
        # Clear frame buffers
        traffic_state.current_frame = None
        traffic_state.processed_frame = None
        traffic_state.frame_count = 0
        traffic_state.road_mask = None
        
        # Reset vehicle statistics
        traffic_state.total_vehicles = 0
        traffic_state.vehicle_counts = defaultdict(int)
        traffic_state.density_level = "LOW"
        
        # Reset emergency detection
        traffic_state.emergency_detected = False
        traffic_state.emergency_type = None
        traffic_state.emergency_vehicle_count = 0
        
        # Reset mode
        traffic_state.current_mode = "STANDBY"
        
        logger.info("System reset complete - signal continues running")

def get_signal_duration(density, signal_type):
    """Calculate signal duration based on traffic density"""
    if signal_type == 'YELLOW':
        return 5  # Fixed 5 seconds for yellow
    
    if signal_type == 'RED':
        if density == 'LOW':
            return 30
        elif density == 'MEDIUM':
            return 15
        elif density == 'HIGH':
            return 10
    
    elif signal_type == 'GREEN':
        if density == 'LOW':
            return 10
        elif density == 'MEDIUM':
            return 15
        elif density == 'HIGH':
            return 30
    
    return 20  # Default fallback

def detect_emergency_vehicles(frame, detections):
    """Detect emergency vehicles (ambulance, fire truck, police car)"""
    # This is a simplified version - in production, use specialized model
    emergency_count = 0
    emergency_type = None
    
    # Check for red/blue colors that might indicate emergency vehicles
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color range (for emergency lights)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Blue color range
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Count red and blue pixels (simplified emergency detection)
    red_pixels = cv2.countNonZero(red_mask)
    blue_pixels = cv2.countNonZero(blue_mask)
    
    if red_pixels > 1000 and blue_pixels > 1000:
        emergency_count = 1
        emergency_type = "AMBULANCE/FIRE"
    elif red_pixels > 1000:
        emergency_count = 1
        emergency_type = "EMERGENCY (RED)"
    elif blue_pixels > 1000:
        emergency_count = 1
        emergency_type = "POLICE"
    
    return emergency_count, emergency_type

def update_signal_state():
    """
    State machine for traffic signal control
    Runs continuously independent of detection
    """
    with traffic_state.lock:
        current_time = time.time()
        elapsed = current_time - traffic_state.last_switch_time
        
        # Emergency vehicle override
        if traffic_state.emergency_detected:
            if traffic_state.signal_state != "GREEN":
                traffic_state.signal_state = "GREEN"
                traffic_state.green_start_time = current_time
                traffic_state.signal_timer = traffic_state.emergency_timeout
                traffic_state.last_switch_time = current_time
            return
        
        # Check if minimum green time has been satisfied
        if (traffic_state.signal_state == "GREEN" and 
            traffic_state.green_start_time and 
            current_time - traffic_state.green_start_time < traffic_state.min_green_time):
            return
        
        # Check if it's time to switch to next state
        if elapsed >= traffic_state.signal_timer:
            # State transition logic
            if traffic_state.signal_state == "RED":
                traffic_state.signal_state = "GREEN"
                traffic_state.green_start_time = current_time
                traffic_state.base_duration = get_signal_duration(
                    traffic_state.density_level, 'GREEN'
                )
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"Signal switched to GREEN, duration: {traffic_state.signal_timer}s (density: {traffic_state.density_level})")
                
            elif traffic_state.signal_state == "GREEN":
                traffic_state.signal_state = "YELLOW"
                traffic_state.base_duration = get_signal_duration(None, 'YELLOW')
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"Signal switched to YELLOW, duration: {traffic_state.signal_timer}s")
                
            elif traffic_state.signal_state == "YELLOW":
                traffic_state.signal_state = "RED"
                traffic_state.base_duration = get_signal_duration(
                    traffic_state.density_level, 'RED'
                )
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"Signal switched to RED, duration: {traffic_state.signal_timer}s (density: {traffic_state.density_level})")
                
            traffic_state.last_switch_time = current_time

def signal_controller_loop():
    """
    Dedicated thread for signal control
    Runs forever independent of detection
    """
    logger.info("Signal controller thread started - continuous signal simulation")
    
    while traffic_state.signal_running:
        update_signal_state()
        store_traffic_data()  # Store data periodically
        time.sleep(0.1)  # Update every 100ms for smooth countdown
    
    logger.info("Signal controller thread stopped")

def update_density_level():
    """
    Update traffic density based on vehicle count
    Dynamically updates signal duration WITHOUT resetting the cycle
    """
    with traffic_state.lock:
        old_density = traffic_state.density_level
        
        if traffic_state.total_vehicles <= 7:
            new_density = "LOW"
        elif 8 <= traffic_state.total_vehicles <= 15:
            new_density = "MEDIUM"
        else:  # vehicles > 15
            new_density = "HIGH"
        
        # Update density level
        traffic_state.density_level = new_density
        
        # If density changed and we're in RED or GREEN (not YELLOW)
        if old_density != new_density and traffic_state.signal_state in ['RED', 'GREEN']:
            # Calculate elapsed time in current state
            current_time = time.time()
            elapsed = current_time - traffic_state.last_switch_time
            
            # Get new base duration for current signal based on new density
            new_duration = get_signal_duration(traffic_state.density_level, traffic_state.signal_state)
            old_duration = traffic_state.base_duration
            
            if new_duration != old_duration:
                # Store the elapsed time proportion
                progress_ratio = elapsed / old_duration if old_duration > 0 else 0
                
                # Update base duration
                traffic_state.base_duration = new_duration
                
                # Calculate new remaining time based on progress
                remaining = max(1, new_duration * (1 - min(progress_ratio, 0.95)))
                traffic_state.signal_timer = remaining
                
                logger.info(f"Density changed: {old_density}->{new_density}: "
                          f"Updated {traffic_state.signal_state} duration to {new_duration}s, "
                          f"remaining: {remaining:.1f}s")
        
        elif old_density != new_density:
            logger.info(f"Density changed from {old_density} to {new_density} (vehicles: {traffic_state.total_vehicles})")

def get_road_mask(frame):
    """Extract road mask using segmentation model"""
    if traffic_state.frame_count % ROAD_MASK_UPDATE_FREQ == 0:
        results = traffic_state.seg_model(frame, verbose=False)
        
        if results[0].masks is not None:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for seg_mask in results[0].masks.data:
                seg_mask = seg_mask.cpu().numpy()
                seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
                mask = cv2.bitwise_or(mask, (seg_mask > 0.5).astype(np.uint8) * 255)
            
            traffic_state.road_mask = mask
        else:
            traffic_state.road_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    return traffic_state.road_mask

def process_frame(frame):
    """
    Main processing pipeline for each frame
    ONLY does detection - does NOT control signal
    """
    if frame is None:
        return None
    
    # Skip frames for performance
    traffic_state.frame_count += 1
    if traffic_state.frame_count % FRAME_SKIP != 0:
        return traffic_state.processed_frame if traffic_state.processed_frame is not None else frame
    
    # Get road mask
    road_mask = get_road_mask(frame)
    
    if road_mask is None:
        return frame
    
    # Run vehicle detection
    results = traffic_state.det_model(frame, classes=list(VEHICLE_CLASSES.keys()), verbose=False)
    
    # Reset counts for this frame
    current_counts = defaultdict(int)
    total_valid = 0
    
    # Create overlay frame
    overlay = frame.copy()
    
    # Process detections
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if 0 <= center_y < road_mask.shape[0] and 0 <= center_x < road_mask.shape[1]:
                if road_mask[center_y, center_x] > 0:
                    total_valid += 1
                    vehicle_type = VEHICLE_CLASSES.get(int(cls), 'unknown')
                    current_counts[vehicle_type] += 1
                    
                    color = (0, 255, 0)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    label = f"{vehicle_type}"
                    cv2.putText(overlay, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(overlay, (center_x, center_y), 3, (0, 0, 255), -1)
    
    # Detect emergency vehicles
    emergency_count, emergency_type = detect_emergency_vehicles(frame, results)
    
    # Update global counts (thread-safe)
    with traffic_state.lock:
        traffic_state.total_vehicles = total_valid
        traffic_state.vehicle_counts = dict(current_counts)
        
        # Update emergency detection
        if emergency_count > 0:
            traffic_state.emergency_detected = True
            traffic_state.emergency_type = emergency_type
            traffic_state.emergency_vehicle_count = emergency_count
            traffic_state.emergency_timeout = 10  # 10 seconds green for emergency
        else:
            traffic_state.emergency_detected = False
            traffic_state.emergency_type = None
    
    # Update density level (this updates signal durations WITHOUT resetting)
    update_density_level()
    
    # Add semi-transparent road mask overlay
    if road_mask is not None:
        road_colored = cv2.applyColorMap(road_mask, cv2.COLORMAP_BONE)
        overlay = cv2.addWeighted(overlay, 0.8, road_colored, 0.2, 0)
    
    # Draw traffic information overlay
    overlay = draw_info_overlay(overlay)
    
    return overlay

def draw_info_overlay(frame):
    """Draw traffic statistics and signal information on frame"""
    height, width = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 280), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    y_offset = 40
    line_height = 25
    
    cv2.putText(frame, f"Total Vehicles: {traffic_state.total_vehicles}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    for vtype, count in traffic_state.vehicle_counts.items():
        cv2.putText(frame, f"{vtype}: {count}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    y_offset += 5
    
    # Emergency vehicle indicator
    if traffic_state.emergency_detected:
        cv2.putText(frame, f"🚨 EMERGENCY: {traffic_state.emergency_type}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += line_height
    
    density_color = {
        'LOW': (0, 255, 0),
        'MEDIUM': (0, 255, 255),
        'HIGH': (0, 0, 255)
    }.get(traffic_state.density_level, (255, 255, 255))
    
    cv2.putText(frame, f"Density: {traffic_state.density_level}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_color, 2)
    y_offset += line_height
    
    signal_color = {
        'RED': (0, 0, 255),
        'YELLOW': (0, 255, 255),
        'GREEN': (0, 255, 0)
    }.get(traffic_state.signal_state, (255, 255, 255))
    
    cv2.putText(frame, f"Signal: {traffic_state.signal_state}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, signal_color, 2)
    y_offset += line_height
    
    remaining = max(0, traffic_state.signal_timer - (time.time() - traffic_state.last_switch_time))
    cv2.putText(frame, f"Timer: {remaining:.1f}s", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Mode: {traffic_state.current_mode}", 
                (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Draw traffic light
    light_x = width - 80
    light_y = 40
    radius = 20
    
    cv2.rectangle(frame, (light_x - 25, light_y - 10), (light_x + 25, light_y + 70), (50, 50, 50), -1)
    
    red_color = (0, 0, 255) if traffic_state.signal_state == 'RED' else (100, 100, 100)
    cv2.circle(frame, (light_x, light_y), radius, red_color, -1)
    
    yellow_color = (0, 255, 255) if traffic_state.signal_state == 'YELLOW' else (100, 100, 100)
    cv2.circle(frame, (light_x, light_y + 30), radius, yellow_color, -1)
    
    green_color = (0, 255, 0) if traffic_state.signal_state == 'GREEN' else (100, 100, 100)
    cv2.circle(frame, (light_x, light_y + 60), radius, green_color, -1)
    
    return frame

def generate_frames():
    """Generator function for MJPEG streaming"""
    logger.info("Frame generator started")
    while traffic_state.processing_active:
        if traffic_state.processed_frame is not None:
            ret, buffer = cv2.imencode('.jpg', traffic_state.processed_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)
    
    logger.info("Frame generator stopped")

def camera_processing_loop():
    """Main processing loop for camera/video input"""
    logger.info(f"Processing loop started for mode: {traffic_state.current_mode}")
    
    while traffic_state.processing_active and traffic_state.camera and traffic_state.camera.isOpened():
        ret, frame = traffic_state.camera.read()
        if not ret:
            logger.error("Failed to read frame from source")
            break
        
        traffic_state.current_frame = frame
        processed = process_frame(frame)
        
        if processed is not None:
            traffic_state.processed_frame = processed
    
    logger.info("Processing loop ended")
    
    if traffic_state.camera and not traffic_state.processing_active:
        traffic_state.camera.release()
        traffic_state.camera = None

@app.after_request
def add_header(response):
    """Add headers to disable caching"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Flask routes
@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/start_camera')
def start_camera():
    """Start webcam processing - signal continues independently"""
    logger.info("Starting camera...")
    
    # Stop any existing processing
    with traffic_state.lock:
        if traffic_state.processing_active:
            traffic_state.processing_active = False
            time.sleep(0.5)
        
        if traffic_state.camera:
            traffic_state.camera.release()
            traffic_state.camera = None
    
    # Initialize camera
    traffic_state.camera = cv2.VideoCapture(0)
    if not traffic_state.camera.isOpened():
        logger.error("Failed to open camera")
        return jsonify({'error': 'Failed to open camera'}), 500
    
    # Load models if not loaded
    if not traffic_state.det_model:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Set new state - signal continues running
    with traffic_state.lock:
        traffic_state.processing_active = True
        traffic_state.frame_count = 0
        traffic_state.current_mode = "CAMERA"
        # Signal state is preserved, not reset
    
    # Start processing thread
    traffic_state.processing_thread = threading.Thread(target=camera_processing_loop)
    traffic_state.processing_thread.daemon = True
    traffic_state.processing_thread.start()
    
    logger.info("Camera started successfully - signal continues")
    return jsonify({'status': 'Camera started', 'mode': 'CAMERA'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera processing - signal continues running"""
    logger.info("Stopping camera...")
    
    with traffic_state.lock:
        traffic_state.processing_active = False
        traffic_state.current_mode = "STANDBY"
        
        if traffic_state.camera:
            traffic_state.camera.release()
            traffic_state.camera = None
    
    logger.info("Camera stopped - signal continues")
    return jsonify({'status': 'Camera stopped', 'mode': 'STANDBY'})

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video file upload - signal continues independently"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    logger.info(f"Processing uploaded video: {video_file.filename}")
    
    # Stop any existing processing
    with traffic_state.lock:
        if traffic_state.processing_active:
            traffic_state.processing_active = False
            time.sleep(0.5)
        
        if traffic_state.camera:
            traffic_state.camera.release()
            traffic_state.camera = None
    
    # Save uploaded video temporarily
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = f'uploads/traffic_{timestamp}.mp4'
    os.makedirs('uploads', exist_ok=True)
    video_file.save(video_path)
    
    # Initialize video capture
    traffic_state.camera = cv2.VideoCapture(video_path)
    if not traffic_state.camera.isOpened():
        logger.error("Failed to open video file")
        return jsonify({'error': 'Failed to open video file'}), 500
    
    # Load models if not loaded
    if not traffic_state.det_model:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Set new state - signal continues running
    with traffic_state.lock:
        traffic_state.processing_active = True
        traffic_state.frame_count = 0
        traffic_state.current_mode = "VIDEO"
        # Signal state is preserved, not reset
    
    # Start processing thread
    traffic_state.processing_thread = threading.Thread(target=camera_processing_loop)
    traffic_state.processing_thread.daemon = True
    traffic_state.processing_thread.start()
    
    logger.info("Video processing started - signal continues")
    return jsonify({'status': 'Video processing started', 'mode': 'VIDEO'})

@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Handle image upload for processing
    IMPORTANT: Process once, don't start thread, don't stop signal
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    logger.info(f"Processing uploaded image: {image_file.filename}")
    
    # Read image directly from file
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Failed to decode image'}), 400
    
    # Load models if not loaded
    if not traffic_state.det_model:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Process the image ONCE
    processed_frame = process_frame(frame)
    
    # Update state - preserve existing processing_active and mode
    with traffic_state.lock:
        # Store the processed frame
        traffic_state.processed_frame = processed_frame
        traffic_state.current_frame = frame
        traffic_state.current_mode = "IMAGE"
        # IMPORTANT: Do NOT change processing_active
        # Signal continues independently
    
    logger.info("Image processed successfully - signal continues")
    return jsonify({'status': 'Image processed', 'mode': 'IMAGE'})

@app.route('/video_feed')
def video_feed():
    """MJPEG video feed endpoint"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """JSON endpoint for traffic statistics"""
    with traffic_state.lock:
        remaining = max(0, traffic_state.signal_timer - (time.time() - traffic_state.last_switch_time))
        stats = {
            'total_vehicles': traffic_state.total_vehicles,
            'vehicle_counts': dict(traffic_state.vehicle_counts),
            'density_level': traffic_state.density_level,
            'signal_state': traffic_state.signal_state,
            'signal_timer': round(remaining, 1),
            'base_duration': traffic_state.base_duration,
            'processing_active': traffic_state.processing_active,
            'current_mode': traffic_state.current_mode,
            'emergency': {
                'detected': traffic_state.emergency_detected,
                'type': traffic_state.emergency_type,
                'timeout_remaining': max(0, traffic_state.emergency_timeout - (time.time() - traffic_state.last_switch_time)) if traffic_state.emergency_detected else 0
            }
        }
    return jsonify(stats)

@app.route('/history')
def get_history():
    """Get encrypted traffic history"""
    history = get_traffic_history()
    return jsonify({'history': history})

@app.route('/analytics')
def get_analytics():
    """Get traffic analytics"""
    with traffic_state.lock:
        if not traffic_state.current_session_data:
            return jsonify({'analytics': {}})
        
        # Calculate analytics
        data = traffic_state.current_session_data
        total_vehicles = [d['total_vehicles'] for d in data]
        densities = [d['density_level'] for d in data]
        
        analytics = {
            'average_vehicles': sum(total_vehicles) / len(total_vehicles) if total_vehicles else 0,
            'max_vehicles': max(total_vehicles) if total_vehicles else 0,
            'min_vehicles': min(total_vehicles) if total_vehicles else 0,
            'density_distribution': {
                'LOW': densities.count('LOW'),
                'MEDIUM': densities.count('MEDIUM'),
                'HIGH': densities.count('HIGH')
            },
            'total_data_points': len(data),
            'session_id': traffic_state.session_id,
            'start_time': data[0]['timestamp'] if data else None,
            'end_time': data[-1]['timestamp'] if data else None
        }
        
        # Add vehicle type distribution
        vehicle_types = defaultdict(int)
        for d in data:
            for vtype, count in d['vehicle_counts'].items():
                vehicle_types[vtype] += count
        
        analytics['vehicle_distribution'] = dict(vehicle_types)
        
    return jsonify({'analytics': analytics})

@app.route('/export_report')
def export_report():
    """Export traffic report as JSON"""
    with traffic_state.lock:
        report = {
            'session_id': traffic_state.session_id,
            'export_time': datetime.now().isoformat(),
            'data': traffic_state.current_session_data,
            'analytics': {
                'average_vehicles': sum([d['total_vehicles'] for d in traffic_state.current_session_data]) / len(traffic_state.current_session_data) if traffic_state.current_session_data else 0,
                'total_data_points': len(traffic_state.current_session_data)
            }
        }
        
        # Encrypt the report
        encrypted_report = encrypt_data(report)
        
    return jsonify({'report': encrypted_report})

@app.route('/status')
def get_status():
    """Simple status endpoint"""
    return jsonify({
        'status': 'running',
        'mode': traffic_state.current_mode,
        'processing': traffic_state.processing_active,
        'signal_running': traffic_state.signal_running
    })

def start_signal_controller():
    """Start the continuous signal controller thread"""
    traffic_state.signal_thread = threading.Thread(target=signal_controller_loop)
    traffic_state.signal_thread.daemon = True
    traffic_state.signal_thread.start()
    logger.info("Signal controller thread launched")

def cleanup():
    """Cleanup function for graceful shutdown"""
    logger.info("Shutting down...")
    traffic_state.signal_running = False
    reset_system()
    cv2.destroyAllWindows()

# Register cleanup function
atexit.register(cleanup)
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)

    load_models()
    start_signal_controller()

    port = int(os.environ.get("PORT", 10000))  # Render provides PORT

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )
