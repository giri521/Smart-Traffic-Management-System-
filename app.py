"""
AI Smart Traffic Management System - Enhanced Version
Flask-based application with YOLOv8 for vehicle detection and local data persistence
Includes predictive density forecasting, dynamic settings, and email alerts
"""

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, session, send_file
from ultralytics import YOLO
import threading
import time
from collections import defaultdict
import logging
import os
from datetime import datetime, timedelta
import atexit
import sqlite3
import json
import gc
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import queue
import warnings
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import csv
import re
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with development configuration
app = Flask(__name__)
app.secret_key = 'traffic-management-secret-key-2024'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Email configuration - NOW USING ENVIRONMENT VARIABLES
EMAIL_CONFIG = {
    'SMTP_SERVER': 'smtp.gmail.com',
    'SMTP_PORT': 587,
    'SMTP_USERNAME': os.getenv('SMTP_USER', 'girivennapusa8@gmail.com'),  # Fallback for development
    'SMTP_PASSWORD': os.getenv('SMTP_PASS', 'exftkirzmhjwplmr'),          # Fallback for development
    'SMTP_FROM': os.getenv('SMTP_USER', 'girivennapusa8@gmail.com'),      # Fallback for development
    'ALERT_ENABLED': False,
    'ALERT_EMAIL': ''  # Will be set from frontend
}

# Global configuration with default values
class Config:
    # Detection settings
    FRAME_SKIP = 2
    ROAD_MASK_UPDATE_FREQ = 10
    VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    # Data storage
    DATA_STORAGE_INTERVAL = 300  # 5 minutes in seconds
    FORCE_STORAGE_EVERY_N_FRAMES = 100
    
    # Training settings
    AUTO_TRAINING_INTERVAL = 3600  # 1 hour in seconds
    MIN_TRAINING_SAMPLES = 10  # Reduced for testing
    TRAINING_TIME_LIMIT = 300  # 5 minutes max training time
    PREDICTION_MODEL = 'random_forest'
    
    # Traffic light durations (seconds) - per density level
    SIGNAL_DURATIONS = {
        'RED': {'LOW': 30, 'MEDIUM': 15, 'HIGH': 10},
        'GREEN': {'LOW': 10, 'MEDIUM': 15, 'HIGH': 30},
        'YELLOW': 5
    }
    
    # Alert settings
    ALERT_TYPES = {
        'mode_change_auto_to_manual': True,
        'mode_change_manual_to_auto': True,
        'camera_inactive': True,
        'training_complete': True,
        'model_update': True
    }
    
    # Camera settings
    CAMERA_INACTIVITY_TIMEOUT = 300  # 5 minutes

# Database path
LOCAL_DB_PATH = 'traffic_data.db'

# Model path
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'traffic_prediction_model.pkl')

def validate_email(email):
    """
    Validate email format
    Returns (is_valid, error_message)
    """
    if not email or not isinstance(email, str):
        return False, "Email is required"
    
    email = email.strip()
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return False, "Invalid email format"
    
    # Additional checks for common email providers
    domain = email.split('@')[1].lower()
    common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
    
    # Check if it's a common domain or has valid structure
    if domain not in common_domains and len(domain.split('.')) < 2:
        return False, "Invalid email domain"
    
    return True, "Valid email"

def init_database():
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect(LOCAL_DB_PATH)
    cursor = conn.cursor()
    
    # Create traffic history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_count INTEGER DEFAULT 0,
            density TEXT DEFAULT 'LOW',
            signal_state TEXT DEFAULT 'RED',
            mode TEXT DEFAULT 'STANDBY',
            control_mode TEXT DEFAULT 'AUTO',
            fps REAL DEFAULT 0,
            session_id TEXT,
            vehicle_details TEXT DEFAULT '{}',
            hour_of_day INTEGER,
            day_of_week INTEGER
        )
    ''')
    
    # Create index on timestamp for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON traffic_history(timestamp DESC)
    ''')
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            start_time TEXT,
            end_time TEXT,
            status TEXT DEFAULT 'active',
            control_mode TEXT DEFAULT 'AUTO',
            total_records INTEGER DEFAULT 0
        )
    ''')
    
    # Create mode changes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mode_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            old_mode TEXT,
            new_mode TEXT,
            session_id TEXT
        )
    ''')
    
    # Create manual overrides table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS manual_overrides (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            old_state TEXT,
            new_state TEXT,
            session_id TEXT
        )
    ''')
    
    # Create system_events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event_type TEXT,
            description TEXT,
            session_id TEXT
        )
    ''')
    
    # Create settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            updated_at TEXT
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            hour INTEGER,
            day_of_week INTEGER,
            predicted_density TEXT,
            predicted_count REAL,
            confidence REAL,
            model_used TEXT,
            training_accuracy REAL
        )
    ''')
    
    # Create model_metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_time TEXT,
            samples_used INTEGER,
            accuracy REAL,
            model_type TEXT,
            features TEXT
        )
    ''')
    
    # Create email_settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_email TEXT,
            alert_enabled INTEGER DEFAULT 0,
            updated_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Initialize database
init_database()

def load_settings():
    """Load settings from database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        rows = cursor.fetchall()
        
        # Load email settings
        cursor.execute("SELECT alert_email, alert_enabled FROM email_settings ORDER BY updated_at DESC LIMIT 1")
        email_row = cursor.fetchone()
        if email_row and email_row[0]:
            # Validate email before loading
            is_valid, _ = validate_email(email_row[0])
            if is_valid:
                EMAIL_CONFIG['ALERT_EMAIL'] = email_row[0]
                EMAIL_CONFIG['ALERT_ENABLED'] = bool(email_row[1]) if email_row[1] is not None else False
            else:
                logger.warning(f"Invalid email in database: {email_row[0]}")
                EMAIL_CONFIG['ALERT_EMAIL'] = ''
                EMAIL_CONFIG['ALERT_ENABLED'] = False
        
        conn.close()
        
        for key, value in rows:
            if key == 'SIGNAL_DURATIONS':
                Config.SIGNAL_DURATIONS = json.loads(value)
            elif key == 'ALERT_TYPES':
                Config.ALERT_TYPES = json.loads(value)
            elif hasattr(Config, key):
                if key in ['DATA_STORAGE_INTERVAL', 'CAMERA_INACTIVITY_TIMEOUT', 
                          'AUTO_TRAINING_INTERVAL', 'TRAINING_TIME_LIMIT', 'MIN_TRAINING_SAMPLES']:
                    setattr(Config, key, int(value))
                elif key == 'PREDICTION_MODEL':
                    setattr(Config, key, value)
                else:
                    setattr(Config, key, value)
        
        logger.info(f"Settings loaded from database. Alert email: {EMAIL_CONFIG['ALERT_EMAIL']}, Enabled: {EMAIL_CONFIG['ALERT_ENABLED']}")
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")

def save_settings():
    """Save current settings to database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        settings = [
            ('DATA_STORAGE_INTERVAL', str(Config.DATA_STORAGE_INTERVAL)),
            ('AUTO_TRAINING_INTERVAL', str(Config.AUTO_TRAINING_INTERVAL)),
            ('TRAINING_TIME_LIMIT', str(Config.TRAINING_TIME_LIMIT)),
            ('MIN_TRAINING_SAMPLES', str(Config.MIN_TRAINING_SAMPLES)),
            ('PREDICTION_MODEL', Config.PREDICTION_MODEL),
            ('CAMERA_INACTIVITY_TIMEOUT', str(Config.CAMERA_INACTIVITY_TIMEOUT)),
            ('SIGNAL_DURATIONS', json.dumps(Config.SIGNAL_DURATIONS)),
            ('ALERT_TYPES', json.dumps(Config.ALERT_TYPES))
        ]
        
        for key, value in settings:
            cursor.execute('''
                INSERT INTO settings (key, value, updated_at) 
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
            ''', (key, value, datetime.now().isoformat(), value, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        logger.info("Settings saved to database")
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

def save_email_settings(alert_email, alert_enabled):
    """
    Save email settings to database with validation
    Returns (success, message)
    """
    try:
        # Validate email if provided
        if alert_email:
            is_valid, validation_message = validate_email(alert_email)
            if not is_valid:
                logger.error(f"Email validation failed: {validation_message}")
                return False, validation_message
        
        # Clean and strip email
        alert_email = alert_email.strip() if alert_email else ''
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        # Ensure alert_enabled is stored as integer 1/0
        enabled_int = 1 if alert_enabled else 0
        
        cursor.execute('''
            INSERT INTO email_settings (alert_email, alert_enabled, updated_at)
            VALUES (?, ?, ?)
        ''', (alert_email, enabled_int, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        # Update runtime configuration
        EMAIL_CONFIG['ALERT_EMAIL'] = alert_email
        EMAIL_CONFIG['ALERT_ENABLED'] = alert_enabled
        
        logger.info(f"Email settings saved: '{alert_email}', Enabled: {alert_enabled} (stored as {enabled_int})")
        return True, "Email settings saved successfully"
        
    except Exception as e:
        error_msg = f"Failed to save email settings: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def update_config_from_settings(settings_data):
    """Update Config class with new settings"""
    try:
        # Update storage settings
        if 'storageInterval' in settings_data:
            Config.DATA_STORAGE_INTERVAL = int(settings_data['storageInterval'])
        
        if 'trainingInterval' in settings_data:
            Config.AUTO_TRAINING_INTERVAL = int(settings_data['trainingInterval'])
        
        if 'trainingTimeLimit' in settings_data:
            Config.TRAINING_TIME_LIMIT = int(settings_data['trainingTimeLimit'])
        
        if 'predictionModel' in settings_data:
            Config.PREDICTION_MODEL = settings_data['predictionModel']
        
        if 'cameraTimeout' in settings_data:
            Config.CAMERA_INACTIVITY_TIMEOUT = int(settings_data['cameraTimeout'])
        
        # Update signal durations
        if 'signalDurations' in settings_data:
            signal_data = settings_data['signalDurations']
            Config.SIGNAL_DURATIONS = {
                'RED': {
                    'LOW': int(signal_data['red']['low']),
                    'MEDIUM': int(signal_data['red']['medium']),
                    'HIGH': int(signal_data['red']['high'])
                },
                'GREEN': {
                    'LOW': int(signal_data['green']['low']),
                    'MEDIUM': int(signal_data['green']['medium']),
                    'HIGH': int(signal_data['green']['high'])
                },
                'YELLOW': int(signal_data['yellow'])
            }
        
        # Update alert types
        if 'alertTypes' in settings_data:
            alert_data = settings_data['alertTypes']
            Config.ALERT_TYPES = {
                'mode_change_auto_to_manual': alert_data.get('modeAutoManual', True),
                'mode_change_manual_to_auto': alert_data.get('modeManualAuto', True),
                'camera_inactive': alert_data.get('cameraInactive', True),
                'training_complete': alert_data.get('training', True),
                'model_update': alert_data.get('modelUpdate', True)
            }
        
        # Save to database
        save_settings()
        
        logger.info("Configuration updated successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return False

# Load settings on startup
load_settings()

# Global state management with thread safety
class TrafficState:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera = None
        self.processing_active = False
        self.current_frame = None
        self.processed_frame = None
        self.frame_count = 0
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_counter = 0
        self.current_mode = "STANDBY"  # STANDBY, VIDEO, CAMERA, IMAGE
        
        # Traffic Control Mode
        self.control_mode = "AUTO"  # AUTO or MANUAL
        self.manual_signal_state = "RED"  # Used only in MANUAL mode
        
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
        
        # Training thread
        self.training_thread = None
        self.training_active = False
        self.last_training_time = 0
        self.training_queue = queue.Queue()
        self.prediction_model = None
        self.model_accuracy = 0
        self.model_samples = 0
        self.current_predictions = []  # Store current predictions
        self.hourly_analysis = {}  # Store hourly analysis
        self.peak_hours = []  # Store peak hours
        
        # Camera inactivity tracking
        self.last_frame_time = time.time()
        self.camera_inactive_alert_sent = False
        
        # Data storage
        self.last_storage_time = time.time()
        self.last_frame_storage = 0
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize session
        self.init_session()

    def init_session(self):
        """Initialize session in database"""
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (session_id, start_time, status, control_mode) VALUES (?, ?, ?, ?)",
                (self.session_id, datetime.now().isoformat(), 'active', self.control_mode)
            )
            conn.commit()
            conn.close()
            logger.info(f"Session initialized: {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")

    def log_event(self, event_type, description):
        """Log system event to database"""
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_events (timestamp, event_type, description, session_id) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), event_type, description, self.session_id)
            )
            conn.commit()
            conn.close()
        except:
            pass

    def check_camera_inactivity(self):
        """Check if camera has been inactive and send alert if needed"""
        if not EMAIL_CONFIG['ALERT_ENABLED'] or not EMAIL_CONFIG['ALERT_EMAIL'] or not Config.ALERT_TYPES.get('camera_inactive', False):
            return
        
        if self.current_mode in ['CAMERA', 'VIDEO'] and self.processing_active:
            current_time = time.time()
            if current_time - self.last_frame_time > Config.CAMERA_INACTIVITY_TIMEOUT:
                if not self.camera_inactive_alert_sent:
                    # Use the fixed send_alert function
                    send_alert(
                        "Camera Inactivity Alert",
                        f"Camera has been inactive for {Config.CAMERA_INACTIVITY_TIMEOUT/60:.0f} minutes. "
                        f"Last frame received: {datetime.fromtimestamp(self.last_frame_time).strftime('%H:%M:%S')}",
                        'camera_inactive'
                    )
                    self.camera_inactive_alert_sent = True
            else:
                self.camera_inactive_alert_sent = False

traffic_state = TrafficState()

# FIXED: Email alert function with improved error handling and logging
def send_alert(subject, message, alert_type=None):
    """
    Send email alert if enabled
    Returns (success, error_message)
    """
    # Log current configuration for debugging
    logger.info(f"=== SEND ALERT ATTEMPT ===")
    logger.info(f"Subject: {subject}")
    logger.info(f"Alert Type: {alert_type}")
    logger.info(f"ALERT_ENABLED: {EMAIL_CONFIG['ALERT_ENABLED']}")
    logger.info(f"ALERT_EMAIL: '{EMAIL_CONFIG['ALERT_EMAIL']}'")
    
    # Check if email is configured
    if not EMAIL_CONFIG['ALERT_EMAIL']:
        error_msg = "Alert not sent: No alert email configured"
        logger.warning(error_msg)
        return False, error_msg
    
    # Check if alerts are enabled globally
    if not EMAIL_CONFIG['ALERT_ENABLED']:
        error_msg = "Alert not sent: Alerts are disabled globally"
        logger.warning(error_msg)
        return False, error_msg
    
    # Check if this specific alert type is enabled
    if alert_type and not Config.ALERT_TYPES.get(alert_type, False):
        error_msg = f"Alert not sent: Alert type '{alert_type}' is disabled"
        logger.info(error_msg)
        return False, error_msg
    
    # Validate email format
    is_valid, validation_error = validate_email(EMAIL_CONFIG['ALERT_EMAIL'])
    if not is_valid:
        error_msg = f"Alert not sent: Invalid email format - {validation_error}"
        logger.error(error_msg)
        return False, error_msg
    
    # Check SMTP credentials
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    logger.info(f"SMTP Configured: {smtp_configured}")
    
    if not smtp_configured:
        error_msg = "Alert not sent: SMTP credentials not configured"
        logger.error(error_msg)
        return False, error_msg
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['SMTP_FROM']
        msg['To'] = EMAIL_CONFIG['ALERT_EMAIL']
        msg['Subject'] = f"[Traffic Management] {subject}"
        
        body = f"""
        <html>
        <body>
            <h2>Traffic Management System Alert</h2>
            <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Session:</strong> {traffic_state.session_id}</p>
            <hr>
            <p>{message}</p>
            <hr>
            <p><small>This is an automated message from your AI Traffic Management System</small></p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server with timeout
        logger.info(f"Connecting to SMTP server {EMAIL_CONFIG['SMTP_SERVER']}:{EMAIL_CONFIG['SMTP_PORT']}")
        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'], timeout=30)
        
        # Start TLS
        logger.info("Starting TLS...")
        server.starttls()
        
        # Login
        logger.info(f"Attempting login with username: {EMAIL_CONFIG['SMTP_USERNAME']}")
        server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
        
        # Send email
        logger.info(f"Sending email to {EMAIL_CONFIG['ALERT_EMAIL']}...")
        server.send_message(msg)
        
        # Close connection
        server.quit()
        
        success_msg = f"Alert sent successfully: {subject} to {EMAIL_CONFIG['ALERT_EMAIL']}"
        logger.info(success_msg)
        traffic_state.log_event("ALERT", f"Sent alert: {subject}")
        return True, success_msg
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication Failed: Check username/password - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except smtplib.SMTPException as e:
        error_msg = f"SMTP Error: {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except ConnectionRefusedError as e:
        error_msg = f"Connection Refused: Cannot connect to SMTP server {EMAIL_CONFIG['SMTP_SERVER']}:{EMAIL_CONFIG['SMTP_PORT']} - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except TimeoutError as e:
        error_msg = f"Connection Timeout: SMTP server not responding - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error sending alert: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        traffic_state.log_event("ERROR", error_msg)
        return False, error_msg

def reset_for_new_input():
    """
    Completely stop previous processing and clear all data for new input
    Signal controller continues running independently
    """
    with traffic_state.lock:
        logger.info("=== RESETTING FOR NEW INPUT ===")
        
        # Step 1: Stop processing thread
        if traffic_state.processing_active:
            logger.info("Stopping active processing thread...")
            traffic_state.processing_active = False
            
            # Wait for thread to finish (with timeout)
            if traffic_state.processing_thread and traffic_state.processing_thread.is_alive():
                traffic_state.processing_thread.join(timeout=2.0)
                if traffic_state.processing_thread.is_alive():
                    logger.warning("Processing thread did not stop gracefully")
        
        # Step 2: Release camera/video capture
        if traffic_state.camera:
            logger.info("Releasing camera/video capture...")
            traffic_state.camera.release()
            traffic_state.camera = None
        
        # Step 3: Clear all frames from memory
        logger.info("Clearing frame buffers...")
        traffic_state.current_frame = None
        traffic_state.processed_frame = None
        
        # Step 4: Reset all vehicle data
        logger.info("Resetting vehicle data...")
        traffic_state.total_vehicles = 0
        traffic_state.vehicle_counts = defaultdict(int)
        traffic_state.density_level = "LOW"
        
        # Step 5: Reset frame counters
        traffic_state.frame_count = 0
        traffic_state.frame_counter = 0
        traffic_state.last_fps_update = time.time()
        traffic_state.last_frame_time = time.time()
        
        # Step 6: Reset road mask
        traffic_state.road_mask = None
        traffic_state.last_road_update = 0
        
        # Step 7: Force garbage collection of frames
        gc.collect()
        
        logger.info("=== RESET COMPLETE - SIGNAL CONTINUES ===")
        traffic_state.log_event("INFO", "System reset for new input completed")

def load_models():
    """Load YOLO models with error handling and far vehicle detection optimizations"""
    try:
        # Load detection model with optimized settings for far vehicles
        traffic_state.det_model = YOLO('yolov8n.pt')
        
        # Adjust detection parameters for better far vehicle detection
        traffic_state.det_model.conf = 0.25  # Lower confidence threshold
        traffic_state.det_model.iou = 0.45   # Adjust IoU threshold
        traffic_state.det_model.max_det = 300 # Increase max detections
        
        # Load segmentation model
        traffic_state.seg_model = YOLO('yolov8n-seg.pt')
        
        logger.info("Models loaded successfully with far-vehicle optimizations")
        traffic_state.log_event("INFO", "Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        traffic_state.log_event("ERROR", f"Model loading failed: {e}")
        return False

def calculate_fps():
    """Calculate frames per second"""
    traffic_state.frame_counter += 1
    current_time = time.time()
    
    if current_time - traffic_state.last_fps_update >= 1.0:
        traffic_state.fps = traffic_state.frame_counter
        traffic_state.frame_counter = 0
        traffic_state.last_fps_update = current_time
    
    return traffic_state.fps

def store_traffic_data(force=False):
    """Store current traffic data in SQLite database"""
    try:
        current_time = datetime.now()
        hour = current_time.hour
        day = current_time.weekday()
        
        # Convert vehicle counts to JSON for storage
        vehicle_details = json.dumps(dict(traffic_state.vehicle_counts))
        
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO traffic_history 
            (timestamp, vehicle_count, density, signal_state, mode, control_mode, fps, 
             session_id, vehicle_details, hour_of_day, day_of_week)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_time.isoformat(),
            traffic_state.total_vehicles,
            traffic_state.density_level,
            traffic_state.signal_state,
            traffic_state.current_mode,
            traffic_state.control_mode,
            traffic_state.fps,
            traffic_state.session_id,
            vehicle_details,
            hour,
            day
        ))
        
        # Update session record count
        cursor.execute('''
            UPDATE sessions 
            SET total_records = total_records + 1 
            WHERE session_id = ?
        ''', (traffic_state.session_id,))
        
        conn.commit()
        conn.close()
        
        if force:
            logger.info(f"FORCED: Traffic data stored - {traffic_state.total_vehicles} vehicles at {current_time.strftime('%H:%M:%S')}")
        
        return True
    except Exception as e:
        logger.error(f"Error storing data: {e}")
        traffic_state.log_event("ERROR", f"Data storage failed: {e}")
        return False

def load_saved_model():
    """Load saved prediction model from disk if it exists"""
    try:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Found saved model at {MODEL_PATH}, loading...")
            with open(MODEL_PATH, 'rb') as f:
                traffic_state.prediction_model = pickle.load(f)
            
            # Load model metadata from database
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT accuracy, samples_used, training_time 
                FROM model_metadata 
                ORDER BY training_time DESC 
                LIMIT 1
            ''')
            result = cursor.fetchone()
            conn.close()
            
            if result:
                traffic_state.model_accuracy = result[0]
                traffic_state.model_samples = result[1]
                traffic_state.last_training_time = datetime.fromisoformat(result[2]).timestamp()
                logger.info(f"Loaded model metadata - Accuracy: {result[0]:.1f}%, Samples: {result[1]}")
            else:
                logger.info("No metadata found for loaded model")
            
            logger.info("Saved prediction model loaded successfully")
            traffic_state.log_event("INFO", "Saved prediction model loaded from disk")
            return True
        else:
            logger.info(f"No saved model found at {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Failed to load saved model: {e}")
        traffic_state.log_event("ERROR", f"Failed to load saved model: {e}")
        return False

def save_model_to_disk():
    """Save the trained prediction model to disk"""
    if traffic_state.prediction_model is None:
        logger.warning("No model to save")
        return False
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Save the model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(traffic_state.prediction_model, f)
        
        logger.info(f"Prediction model saved to {MODEL_PATH}")
        traffic_state.log_event("INFO", f"Model saved to disk with accuracy {traffic_state.model_accuracy:.1f}%")
        return True
    except Exception as e:
        logger.error(f"Failed to save model to disk: {e}")
        traffic_state.log_event("ERROR", f"Failed to save model: {e}")
        return False

def analyze_hourly_patterns():
    """Analyze hourly traffic patterns from historical data"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        query = """
            SELECT hour_of_day, AVG(vehicle_count) as avg_count, 
                   MAX(vehicle_count) as max_count,
                   COUNT(*) as sample_count
            FROM traffic_history 
            WHERE vehicle_count > 0
            GROUP BY hour_of_day
            ORDER BY hour_of_day
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return {}
        
        # Calculate peak hours (top 3 hours with highest average)
        peak_hours = df.nlargest(3, 'avg_count')[['hour_of_day', 'avg_count']].to_dict('records')
        
        # Create hourly analysis
        hourly_analysis = {}
        for _, row in df.iterrows():
            hour = int(row['hour_of_day'])
            avg = row['avg_count']
            
            if avg <= 7:
                density = "LOW"
            elif avg <= 15:
                density = "MEDIUM"
            else:
                density = "HIGH"
            
            hourly_analysis[hour] = {
                'hour': hour,
                'avg_count': round(avg, 1),
                'max_count': int(row['max_count']),
                'sample_count': int(row['sample_count']),
                'density': density
            }
        
        traffic_state.hourly_analysis = hourly_analysis
        traffic_state.peak_hours = peak_hours
        
        logger.info(f"Hourly analysis complete. Peak hours: {peak_hours}")
        return hourly_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing hourly patterns: {e}")
        return {}

def generate_predictions():
    """Generate density predictions for next 24 hours"""
    if traffic_state.prediction_model is None:
        logger.info("No trained model available for predictions")
        return generate_demo_predictions()
    
    try:
        predictions = []
        current_time = datetime.now()
        
        # Get average vehicle count for reference
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(vehicle_count) FROM traffic_history WHERE vehicle_count > 0 LIMIT 100")
        result = cursor.fetchone()
        avg_count = result[0] if result and result[0] else 10
        conn.close()
        
        for hour_offset in range(1, 25):
            pred_time = current_time + timedelta(hours=hour_offset)
            hour = pred_time.hour
            day = pred_time.weekday()
            
            # Create features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            
            # Get historical average for this hour as prev_hour
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT AVG(vehicle_count) FROM traffic_history WHERE hour_of_day = ? LIMIT 50",
                (max(0, hour-1),)
            )
            result = cursor.fetchone()
            prev_hour_avg = result[0] if result and result[0] else avg_count
            conn.close()
            
            features = np.array([[hour_sin, hour_cos, day_sin, day_cos, prev_hour_avg]])
            
            # Predict
            try:
                pred_count = traffic_state.prediction_model.predict(features)[0]
            except:
                pred_count = avg_count
            
            # Ensure positive count
            pred_count = max(1, pred_count)
            
            # Determine density based on predicted count
            if pred_count <= 7:
                density = "LOW"
            elif pred_count <= 15:
                density = "MEDIUM"
            else:
                density = "HIGH"
            
            prediction = {
                'hour': hour_offset,
                'time': pred_time.strftime('%H:00'),
                'hour_of_day': hour,
                'predicted_count': round(pred_count, 1),
                'density': density,
                'confidence': round(traffic_state.model_accuracy, 1)
            }
            predictions.append(prediction)
            
            # Store prediction in database
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, hour, day_of_week, predicted_density, predicted_count, confidence, model_used, training_accuracy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                hour_offset,
                day,
                density,
                round(pred_count, 1),
                traffic_state.model_accuracy / 100,
                Config.PREDICTION_MODEL,
                traffic_state.model_accuracy
            ))
            conn.commit()
            conn.close()
        
        logger.info(f"Generated {len(predictions)} hourly predictions")
        
        # Store predictions in traffic_state for quick access
        traffic_state.current_predictions = predictions
        
        # Analyze hourly patterns
        analyze_hourly_patterns()
        
        # Send alert for new predictions if enabled
        if EMAIL_CONFIG['ALERT_ENABLED'] and EMAIL_CONFIG['ALERT_EMAIL'] and Config.ALERT_TYPES.get('model_update', False):
            peak_hours = [p for p in predictions if p['density'] == 'HIGH'][:3]
            peak_info = "<br>".join([f"• {p['time']}: {p['density']} ({p['predicted_count']} vehicles)" 
                                     for p in peak_hours]) if peak_hours else "No peak hours predicted"
            
            message = f"""
            <h3>New Traffic Predictions Available</h3>
            <p><strong>Model accuracy:</strong> {traffic_state.model_accuracy:.1f}%</p>
            <p><strong>Samples used:</strong> {traffic_state.model_samples}</p>
            <h4>Peak Hours (next 24h):</h4>
            {peak_info}
            <p><small>View full predictions in the dashboard</small></p>
            """
            send_alert("New Traffic Predictions", message, 'model_update')
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction generation error: {e}")
        return generate_demo_predictions()

def generate_demo_predictions():
    """Generate demo predictions when no model is available"""
    predictions = []
    current_time = datetime.now()
    
    for hour_offset in range(1, 25):
        pred_time = current_time + timedelta(hours=hour_offset)
        hour = pred_time.hour
        
        # Generate realistic demo predictions based on time of day
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            count = 18 + (hour_offset % 5)
            density = "HIGH"
        elif 10 <= hour <= 16:  # Mid-day
            count = 12 + (hour_offset % 4)
            density = "MEDIUM"
        else:  # Night/early morning
            count = 5 + (hour_offset % 3)
            density = "LOW"
        
        predictions.append({
            'hour': hour_offset,
            'time': pred_time.strftime('%H:00'),
            'hour_of_day': hour,
            'predicted_count': count,
            'density': density,
            'confidence': 85.0
        })
    
    return predictions

def train_prediction_model():
    """
    Background thread for training prediction model
    Runs without blocking real-time detection
    """
    if traffic_state.training_active:
        logger.info("Training already in progress, skipping...")
        return
    
    traffic_state.training_active = True
    start_time = time.time()
    
    try:
        logger.info("Starting background model training...")
        
        # Fetch training data
        conn = sqlite3.connect(LOCAL_DB_PATH)
        query = """
            SELECT hour_of_day, day_of_week, vehicle_count, density 
            FROM traffic_history 
            WHERE vehicle_count > 0
            ORDER BY timestamp DESC 
            LIMIT 10000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < Config.MIN_TRAINING_SAMPLES:
            logger.info(f"Insufficient training samples: {len(df)} < {Config.MIN_TRAINING_SAMPLES}")
            # Generate demo predictions anyway
            traffic_state.current_predictions = generate_demo_predictions()
            traffic_state.training_active = False
            return
        
        # Prepare features
        df['density_encoded'] = df['density'].map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2})
        
        # Feature engineering
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Create lag features (previous hour's traffic)
        df = df.sort_values('hour_of_day')
        df['prev_hour_count'] = df['vehicle_count'].shift(1)
        df['prev_hour_count'] = df['prev_hour_count'].fillna(df['vehicle_count'].mean())
        
        features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'prev_hour_count']
        X = df[features]
        y_count = df['vehicle_count']
        
        # Split data
        X_train, X_test, y_train_count, y_test_count = train_test_split(
            X, y_count, test_size=0.2, random_state=42
        )
        
        # Train model for vehicle count prediction
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Check time limit
        if time.time() - start_time > Config.TRAINING_TIME_LIMIT:
            logger.info("Training time limit reached, stopping early")
            traffic_state.training_active = False
            return
        
        model.fit(X_train, y_train_count)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test_count, y_pred)
        r2 = r2_score(y_test_count, y_pred)
        
        # Calculate accuracy as percentage (0-100)
        y_test_mean = y_test_count.mean()
        if y_test_mean > 0:
            accuracy = max(0, min(100, (1 - mae / y_test_mean) * 100))
        else:
            accuracy = 80  # Default accuracy
        
        # Store model
        traffic_state.prediction_model = model
        traffic_state.model_accuracy = accuracy
        traffic_state.model_samples = len(df)
        traffic_state.last_training_time = time.time()
        
        # Save model to disk
        save_model_to_disk()
        
        # Store metadata
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO model_metadata 
            (training_time, samples_used, accuracy, model_type, features)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            len(df),
            accuracy,
            Config.PREDICTION_MODEL,
            json.dumps(features)
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Model training complete - MAE: {mae:.2f}, R2: {r2:.3f}, Accuracy: {accuracy:.1f}%")
        
        # Analyze hourly patterns
        analyze_hourly_patterns()
        
        # Generate predictions for next 24 hours
        predictions = generate_predictions()
        
        # Store predictions in traffic_state
        traffic_state.current_predictions = predictions
        
        # Send alert if enabled
        if EMAIL_CONFIG['ALERT_ENABLED'] and EMAIL_CONFIG['ALERT_EMAIL'] and Config.ALERT_TYPES.get('training_complete', False):
            message = f"""
            <h3>Model Training Complete</h3>
            <ul>
                <li><strong>Samples used:</strong> {len(df)}</li>
                <li><strong>Mean Absolute Error:</strong> {mae:.2f} vehicles</li>
                <li><strong>R² Score:</strong> {r2:.3f}</li>
                <li><strong>Accuracy:</strong> {accuracy:.1f}%</li>
                <li><strong>Model type:</strong> {Config.PREDICTION_MODEL}</li>
                <li><strong>Training time:</strong> {time.time() - start_time:.1f} seconds</li>
                <li><strong>Predictions generated:</strong> {len(predictions)}</li>
                <li><strong>Model saved to:</strong> {MODEL_PATH}</li>
            </ul>
            """
            send_alert("Model Training Completed", message, 'training_complete')
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        traffic_state.log_event("ERROR", f"Training failed: {e}")
        # Generate demo predictions on error
        traffic_state.current_predictions = generate_demo_predictions()
    
    finally:
        traffic_state.training_active = False

def get_signal_duration(density, signal_type):
    """Get signal duration based on traffic density from config"""
    if signal_type == "YELLOW":
        return Config.SIGNAL_DURATIONS.get('YELLOW', 5)
    else:
        return Config.SIGNAL_DURATIONS.get(signal_type, {}).get(density, 20)

def update_signal_state():
    """
    State machine for traffic signal control
    Runs differently based on AUTO/MANUAL mode
    """
    with traffic_state.lock:
        current_time = time.time()
        
        # If in MANUAL mode, don't auto-switch signals
        if traffic_state.control_mode == "MANUAL":
            # In manual mode, signal state is controlled by user
            # Just update the timer display
            if traffic_state.signal_state != traffic_state.manual_signal_state:
                traffic_state.signal_state = traffic_state.manual_signal_state
                traffic_state.last_switch_time = current_time
                # Set a fixed timer for manual mode (just for display)
                if traffic_state.signal_state == "RED":
                    traffic_state.signal_timer = 30
                elif traffic_state.signal_state == "YELLOW":
                    traffic_state.signal_timer = 5
                elif traffic_state.signal_state == "GREEN":
                    traffic_state.signal_timer = 30
            return
        
        # AUTO mode logic continues below
        elapsed = current_time - traffic_state.last_switch_time
        
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
                logger.info(f"[AUTO] Signal switched to GREEN, duration: {traffic_state.signal_timer}s (density: {traffic_state.density_level})")
                
            elif traffic_state.signal_state == "GREEN":
                traffic_state.signal_state = "YELLOW"
                traffic_state.base_duration = get_signal_duration(None, 'YELLOW')
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"[AUTO] Signal switched to YELLOW, duration: {traffic_state.signal_timer}s")
                
            elif traffic_state.signal_state == "YELLOW":
                traffic_state.signal_state = "RED"
                traffic_state.base_duration = get_signal_duration(
                    traffic_state.density_level, 'RED'
                )
                traffic_state.signal_timer = traffic_state.base_duration
                logger.info(f"[AUTO] Signal switched to RED, duration: {traffic_state.signal_timer}s (density: {traffic_state.density_level})")
                
            traffic_state.last_switch_time = current_time

def signal_controller_loop():
    """
    Dedicated thread for signal control
    Runs forever independent of detection
    """
    logger.info("Signal controller thread started - continuous signal simulation")
    traffic_state.log_event("INFO", "Signal controller thread started")
    
    storage_counter = 0
    training_check_counter = 0
    
    while traffic_state.signal_running:
        update_signal_state()
        
        # Check camera inactivity
        traffic_state.check_camera_inactivity()
        
        # Store data periodically (configurable interval)
        with traffic_state.lock:
            current_time = time.time()
            if current_time - traffic_state.last_storage_time >= Config.DATA_STORAGE_INTERVAL:
                store_traffic_data()
                traffic_state.last_storage_time = current_time
                storage_counter += 1
                logger.info(f"Periodic storage #{storage_counter} completed")
        
        # Check if it's time to train model
        training_check_counter += 1
        if (training_check_counter % 10 == 0 and  # Check every ~1 second
            not traffic_state.training_active and
            time.time() - traffic_state.last_training_time > Config.AUTO_TRAINING_INTERVAL):
            
            logger.info("Auto-training timer triggered")
            traffic_state.training_thread = threading.Thread(target=train_prediction_model)
            traffic_state.training_thread.daemon = True
            traffic_state.training_thread.start()
        
        time.sleep(0.1)  # Update every 100ms for smooth countdown
    
    logger.info("Signal controller thread stopped")
    traffic_state.log_event("INFO", "Signal controller thread stopped")

def update_density_level():
    """
    Update traffic density based on vehicle count
    Dynamically updates signal duration WITHOUT resetting the cycle
    Only affects AUTO mode
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
        
        # Only auto-adjust timings in AUTO mode
        if traffic_state.control_mode == "AUTO":
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

def enhance_image_for_far_vehicles(frame):
    """Apply image preprocessing to improve far vehicle detection"""
    try:
        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Convert back to BGR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_bgr, -1, kernel)
        
        # Blend with original for natural look
        result = cv2.addWeighted(frame, 0.5, sharpened, 0.5, 0)
        
        return result
    except Exception as e:
        logger.error(f"Image enhancement error: {e}")
        return frame

def get_road_mask(frame):
    """Extract road mask using segmentation model"""
    if traffic_state.frame_count % Config.ROAD_MASK_UPDATE_FREQ == 0:
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
    
    # Update last frame time for inactivity detection
    traffic_state.last_frame_time = time.time()
    
    # Calculate FPS
    calculate_fps()
    
    # Skip frames for performance
    traffic_state.frame_count += 1
    if traffic_state.frame_count % Config.FRAME_SKIP != 0:
        return traffic_state.processed_frame if traffic_state.processed_frame is not None else frame
    
    # Enhance image for far vehicle detection
    enhanced_frame = enhance_image_for_far_vehicles(frame)
    
    # Get road mask
    road_mask = get_road_mask(enhanced_frame)
    
    if road_mask is None:
        return frame
    
    # Run vehicle detection with optimized settings for far vehicles
    results = traffic_state.det_model(
        enhanced_frame, 
        classes=list(Config.VEHICLE_CLASSES.keys()), 
        verbose=False,
        conf=0.25,  # Lower confidence for far vehicles
        iou=0.45
    )
    
    # Reset counts for this frame
    current_counts = defaultdict(int)
    total_valid = 0
    
    # Create overlay frame
    overlay = frame.copy()
    
    # Process detections
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if 0 <= center_y < road_mask.shape[0] and 0 <= center_x < road_mask.shape[1]:
                if road_mask[center_y, center_x] > 0:
                    total_valid += 1
                    vehicle_type = Config.VEHICLE_CLASSES.get(int(cls), 'unknown')
                    current_counts[vehicle_type] += 1
                    
                    # Color based on confidence (brighter for higher confidence)
                    color_intensity = int(conf * 255)
                    color = (0, color_intensity, 255 - color_intensity)
                    
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                    
                    # Add confidence to label for far vehicles
                    box_size = (x2 - x1) * (y2 - y1)
                    if box_size < 5000:  # Small box = far vehicle
                        label = f"{vehicle_type} ({conf:.2f})"
                    else:
                        label = vehicle_type
                    
                    cv2.putText(overlay, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    cv2.circle(overlay, (center_x, center_y), 2, (0, 0, 255), -1)
    
    # Update global counts (thread-safe)
    with traffic_state.lock:
        traffic_state.total_vehicles = total_valid
        traffic_state.vehicle_counts = dict(current_counts)
    
    # Update density level
    update_density_level()
    
    # Force store data periodically (configurable)
    if traffic_state.frame_count % Config.FORCE_STORAGE_EVERY_N_FRAMES == 0 and traffic_state.frame_count > 0:
        store_traffic_data(force=True)
    
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
    cv2.rectangle(overlay, (10, 10), (480, 400), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    y_offset = 40
    line_height = 25
    
    # Mode indicator
    mode_color = (0, 255, 0) if traffic_state.control_mode == "AUTO" else (255, 255, 0)
    cv2.putText(frame, f"Mode: {traffic_state.control_mode}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    y_offset += line_height
    
    cv2.putText(frame, f"Total Vehicles: {traffic_state.total_vehicles}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(frame, f"FPS: {traffic_state.fps}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height
    
    for vtype, count in traffic_state.vehicle_counts.items():
        cv2.putText(frame, f"{vtype}: {count}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 20
    
    y_offset += 5
    
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
    
    # Add storage info
    time_since_last = time.time() - traffic_state.last_storage_time
    next_storage = max(0, Config.DATA_STORAGE_INTERVAL - time_since_last)
    cv2.putText(frame, f"Next storage: {next_storage/60:.1f}min", 
                (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Add model info
    if traffic_state.model_accuracy > 0:
        cv2.putText(frame, f"Model acc: {traffic_state.model_accuracy:.1f}%", 
                   (20, y_offset + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
    
    # Add prediction info
    if traffic_state.current_predictions:
        next_hour = traffic_state.current_predictions[0] if traffic_state.current_predictions else None
        if next_hour:
            cv2.putText(frame, f"Next hour: {next_hour['density']} ({next_hour['predicted_count']} veh)", 
                       (20, y_offset + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 100), 1)
    
    cv2.putText(frame, f"Session: {traffic_state.session_id[-8:]}", 
                (20, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    cv2.putText(frame, f"Input: {traffic_state.current_mode}", 
                (20, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Alert indicator
    if EMAIL_CONFIG['ALERT_ENABLED'] and EMAIL_CONFIG['ALERT_EMAIL']:
        cv2.putText(frame, "Alerts ON", (width - 120, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Manual mode hint
    if traffic_state.control_mode == "MANUAL":
        cv2.putText(frame, "Manual Control - Use buttons", 
                   (width - 300, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
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
    
    frame_counter = 0
    while traffic_state.processing_active and traffic_state.camera and traffic_state.camera.isOpened():
        ret, frame = traffic_state.camera.read()
        if not ret:
            logger.error("Failed to read frame from source")
            traffic_state.log_event("ERROR", "Failed to read frame from source")
            break
        
        traffic_state.current_frame = frame
        processed = process_frame(frame)
        
        if processed is not None:
            traffic_state.processed_frame = processed
            frame_counter += 1
            
            # Log every 100 frames
            if frame_counter % 100 == 0:
                logger.info(f"Processed {frame_counter} frames, current vehicles: {traffic_state.total_vehicles}")
    
    logger.info(f"Processing loop ended after {frame_counter} frames")
    
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

@app.route('/set_mode/<mode>')
def set_mode(mode):
    """Set control mode to AUTO or MANUAL"""
    if mode.upper() not in ['AUTO', 'MANUAL']:
        return jsonify({'error': 'Invalid mode'}), 400
    
    with traffic_state.lock:
        old_mode = traffic_state.control_mode
        traffic_state.control_mode = mode.upper()
        
        # If switching to MANUAL, set manual signal to current signal
        if mode.upper() == "MANUAL":
            traffic_state.manual_signal_state = traffic_state.signal_state
        
        logger.info(f"Control mode changed: {old_mode} -> {mode.upper()}")
        traffic_state.log_event("MODE_CHANGE", f"Mode changed from {old_mode} to {mode.upper()}")
        
        # Send alert if enabled
        alert_type = None
        if old_mode == "AUTO" and mode.upper() == "MANUAL":
            alert_type = 'mode_change_auto_to_manual'
            message = f"System switched from AUTO to MANUAL mode at {datetime.now().strftime('%H:%M:%S')}"
        elif old_mode == "MANUAL" and mode.upper() == "AUTO":
            alert_type = 'mode_change_manual_to_auto'
            message = f"System switched from MANUAL to AUTO mode at {datetime.now().strftime('%H:%M:%S')}"
        else:
            message = f"Mode changed from {old_mode} to {mode.upper()}"
        
        # Use the fixed send_alert function
        send_alert("Mode Change Alert", message, alert_type)
        
        # Store mode change
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO mode_changes (timestamp, old_mode, new_mode, session_id) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), old_mode, mode.upper(), traffic_state.session_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store mode change: {e}")
    
    return jsonify({
        'status': 'success',
        'mode': mode.upper(),
        'message': f'Switched to {mode.upper()} mode'
    })

@app.route('/set_signal/<state>')
def set_signal(state):
    """Manually set signal state (only works in MANUAL mode)"""
    if state.upper() not in ['RED', 'YELLOW', 'GREEN']:
        return jsonify({'error': 'Invalid signal state'}), 400
    
    with traffic_state.lock:
        if traffic_state.control_mode != "MANUAL":
            return jsonify({
                'error': 'Cannot manually set signal in AUTO mode',
                'current_mode': traffic_state.control_mode
            }), 400
        
        old_state = traffic_state.manual_signal_state
        traffic_state.manual_signal_state = state.upper()
        
        # Force immediate update
        traffic_state.signal_state = state.upper()
        traffic_state.last_switch_time = time.time()
        
        # Set appropriate timer for display
        if state.upper() == "RED":
            traffic_state.signal_timer = 30
        elif state.upper() == "YELLOW":
            traffic_state.signal_timer = 5
        elif state.upper() == "GREEN":
            traffic_state.signal_timer = 30
        
        logger.info(f"Manual signal changed: {old_state} -> {state.upper()}")
        traffic_state.log_event("MANUAL_OVERRIDE", f"Signal changed from {old_state} to {state.upper()}")
        
        # Store manual override
        try:
            conn = sqlite3.connect(LOCAL_DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO manual_overrides (timestamp, old_state, new_state, session_id) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), old_state, state.upper(), traffic_state.session_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store manual override: {e}")
    
    return jsonify({
        'status': 'success',
        'signal': state.upper(),
        'message': f'Signal set to {state.upper()}'
    })

@app.route('/get_mode')
def get_mode():
    """Get current control mode"""
    with traffic_state.lock:
        return jsonify({
            'control_mode': traffic_state.control_mode,
            'signal_state': traffic_state.signal_state,
            'manual_signal': traffic_state.manual_signal_state if traffic_state.control_mode == "MANUAL" else None,
            'session_id': traffic_state.session_id
        })

@app.route('/start_camera')
def start_camera():
    """Start webcam processing - signal continues independently"""
    logger.info("Starting camera...")
    
    # COMPLETE RESET before starting new input
    reset_for_new_input()
    
    # Initialize camera
    traffic_state.camera = cv2.VideoCapture(0)
    if not traffic_state.camera.isOpened():
        logger.error("Failed to open camera")
        traffic_state.log_event("ERROR", "Failed to open camera")
        return jsonify({'error': 'Failed to open camera'}), 500
    
    # Load models if not loaded
    if not traffic_state.det_model:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Set new state - signal continues running
    with traffic_state.lock:
        traffic_state.processing_active = True
        traffic_state.current_mode = "CAMERA"
        logger.info(f"Processing active set to: {traffic_state.processing_active}")
    
    # Start processing thread
    traffic_state.processing_thread = threading.Thread(target=camera_processing_loop)
    traffic_state.processing_thread.daemon = True
    traffic_state.processing_thread.start()
    
    logger.info("Camera started successfully - signal continues")
    traffic_state.log_event("INFO", "Camera started")
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
    traffic_state.log_event("INFO", "Camera stopped")
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
    
    # COMPLETE RESET before starting new input
    reset_for_new_input()
    
    # Save uploaded video temporarily
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = f'uploads/traffic_{timestamp}.mp4'
    os.makedirs('uploads', exist_ok=True)
    video_file.save(video_path)
    
    # Initialize video capture
    traffic_state.camera = cv2.VideoCapture(video_path)
    if not traffic_state.camera.isOpened():
        logger.error("Failed to open video file")
        traffic_state.log_event("ERROR", f"Failed to open video file: {video_file.filename}")
        return jsonify({'error': 'Failed to open video file'}), 500
    
    # Load models if not loaded
    if not traffic_state.det_model:
        if not load_models():
            return jsonify({'error': 'Failed to load models'}), 500
    
    # Set new state - signal continues running
    with traffic_state.lock:
        traffic_state.processing_active = True
        traffic_state.current_mode = "VIDEO"
        logger.info(f"Processing active set to: {traffic_state.processing_active}")
    
    # Start processing thread
    traffic_state.processing_thread = threading.Thread(target=camera_processing_loop)
    traffic_state.processing_thread.daemon = True
    traffic_state.processing_thread.start()
    
    logger.info("Video processing started - signal continues")
    traffic_state.log_event("INFO", f"Video processing started: {video_file.filename}")
    return jsonify({'status': 'Video processing started', 'mode': 'VIDEO'})

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
            'fps': traffic_state.fps,
            'control_mode': traffic_state.control_mode,
            'manual_signal': traffic_state.manual_signal_state if traffic_state.control_mode == "MANUAL" else None,
            'session_id': traffic_state.session_id,
            'model_accuracy': round(traffic_state.model_accuracy, 1) if traffic_state.model_accuracy > 0 else None,
            'training_active': traffic_state.training_active,
            'storage_interval_min': Config.DATA_STORAGE_INTERVAL / 60,
            'predictions_available': len(traffic_state.current_predictions) > 0,
            'alert_email': EMAIL_CONFIG['ALERT_EMAIL'] if EMAIL_CONFIG['ALERT_EMAIL'] else None,
            'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED']
        }
    return jsonify(stats)

@app.route('/history')
def get_history():
    """Get traffic history from database"""
    limit = request.args.get('limit', 100, type=int)
    session = request.args.get('session', None)
    history = get_traffic_history(limit, session)
    
    # Get total records count
    total_records = 0
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM traffic_history")
        total_records = cursor.fetchone()[0]
        conn.close()
    except:
        pass
    
    return jsonify({
        'history': history, 
        'count': len(history),
        'total_records': total_records
    })

def get_traffic_history(limit=100, session_id=None):
    """Get traffic history from database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute(
                "SELECT * FROM traffic_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
                (session_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM traffic_history ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts and parse vehicle_details
        history = []
        for row in rows:
            item = dict(row)
            try:
                item['vehicle_details'] = json.loads(item['vehicle_details'])
            except:
                item['vehicle_details'] = {}
            history.append(item)
            
        logger.info(f"Retrieved {len(history)} history records")
        return history
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return []

@app.route('/analytics')
def get_analytics():
    """Get traffic analytics from database"""
    analytics = get_traffic_analytics()
    return jsonify({'analytics': analytics})

def get_traffic_analytics():
    """Get traffic analytics from database"""
    try:
        history = get_traffic_history(1000)  # Get last 1000 records
        
        if not history:
            logger.info("No history data for analytics")
            return {}
        
        # Calculate analytics
        vehicle_counts = [item.get('vehicle_count', 0) for item in history]
        densities = [item.get('density', 'LOW') for item in history]
        control_modes = [item.get('control_mode', 'AUTO') for item in history]
        
        # Aggregate vehicle types
        vehicle_types = defaultdict(int)
        for item in history:
            details = item.get('vehicle_details', {})
            for vtype, count in details.items():
                vehicle_types[vtype] += count
        
        # Hourly averages
        hourly_avg = defaultdict(list)
        for item in history:
            try:
                hour = datetime.fromisoformat(item['timestamp']).hour
                hourly_avg[hour].append(item['vehicle_count'])
            except:
                pass
        
        hourly_stats = {
            hour: {
                'avg': sum(counts)/len(counts),
                'max': max(counts),
                'min': min(counts)
            }
            for hour, counts in hourly_avg.items()
        }
        
        # Get peak hours
        peak_hours_data = []
        for hour, stats in hourly_stats.items():
            peak_hours_data.append({
                'hour': hour,
                'avg_count': round(stats['avg'], 1)
            })
        peak_hours_data.sort(key=lambda x: x['avg_count'], reverse=True)
        peak_hours = peak_hours_data[:5] if peak_hours_data else []
        
        analytics = {
            'average_vehicles': round(sum(vehicle_counts) / len(vehicle_counts), 2) if vehicle_counts else 0,
            'max_vehicles': max(vehicle_counts) if vehicle_counts else 0,
            'min_vehicles': min(vehicle_counts) if vehicle_counts else 0,
            'density_distribution': {
                'LOW': densities.count('LOW'),
                'MEDIUM': densities.count('MEDIUM'),
                'HIGH': densities.count('HIGH')
            },
            'control_mode_distribution': {
                'AUTO': control_modes.count('AUTO'),
                'MANUAL': control_modes.count('MANUAL')
            },
            'vehicle_type_distribution': dict(vehicle_types),
            'hourly_statistics': hourly_stats,
            'peak_hours': peak_hours,
            'total_data_points': len(history),
            'start_time': history[-1].get('timestamp') if history else None,
            'end_time': history[0].get('timestamp') if history else None
        }
        
        logger.info(f"Analytics calculated from {len(history)} records")
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return {}

@app.route('/predictions')
def get_predictions():
    """Get traffic predictions from database"""
    try:
        # First check if we have current predictions in memory
        if hasattr(traffic_state, 'current_predictions') and traffic_state.current_predictions:
            predictions = traffic_state.current_predictions
        else:
            # Try to get from database
            conn = sqlite3.connect(LOCAL_DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get most recent predictions (one per hour)
            cursor.execute('''
                SELECT p1.* 
                FROM predictions p1
                INNER JOIN (
                    SELECT hour, MAX(timestamp) as max_timestamp
                    FROM predictions
                    GROUP BY hour
                ) p2 ON p1.hour = p2.hour AND p1.timestamp = p2.max_timestamp
                ORDER BY p1.hour ASC
                LIMIT 24
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            predictions = []
            for row in rows:
                pred = dict(row)
                # Format for frontend
                pred['time'] = f"{pred['hour']:02d}:00"
                if 'predicted_count' not in pred:
                    pred['predicted_count'] = 10
                predictions.append(pred)
        
        # If still no predictions, generate demo predictions
        if not predictions:
            predictions = generate_demo_predictions()
            traffic_state.current_predictions = predictions
        
        # Get hourly analysis
        hourly_analysis = traffic_state.hourly_analysis if hasattr(traffic_state, 'hourly_analysis') else {}
        
        # Get peak hours
        peak_hours = traffic_state.peak_hours if hasattr(traffic_state, 'peak_hours') else []
        
        return jsonify({
            'predictions': predictions,
            'hourly_analysis': hourly_analysis,
            'peak_hours': peak_hours,
            'model_accuracy': traffic_state.model_accuracy if traffic_state.model_accuracy > 0 else 85.0,
            'samples_used': traffic_state.model_samples if traffic_state.model_samples > 0 else 100,
            'last_training': datetime.fromtimestamp(traffic_state.last_training_time).isoformat() 
                            if traffic_state.last_training_time > 0 else None
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        # Return demo predictions even on error
        predictions = generate_demo_predictions()
        
        return jsonify({
            'predictions': predictions,
            'hourly_analysis': {},
            'peak_hours': [],
            'model_accuracy': 85.0,
            'samples_used': 100,
            'last_training': datetime.now().isoformat()
        })

@app.route('/force_train')
def force_train():
    """Force model training immediately"""
    if traffic_state.training_active:
        return jsonify({'status': 'Training already in progress'})
    
    threading.Thread(target=train_prediction_model).start()
    return jsonify({'status': 'Training started'})

@app.route('/force_store')
def force_store():
    """Force store current data immediately"""
    success = store_traffic_data(force=True)
    return jsonify({
        'status': 'success' if success else 'error',
        'message': 'Data stored successfully' if success else 'Failed to store data',
        'vehicles': traffic_state.total_vehicles,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/db_stats')
def db_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        
        # Get table sizes
        cursor.execute("SELECT COUNT(*) FROM traffic_history")
        history_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        sessions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mode_changes")
        mode_changes_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM manual_overrides")
        overrides_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM system_events")
        events_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions")
        predictions_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_metadata")
        models_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT alert_email, alert_enabled FROM email_settings ORDER BY updated_at DESC LIMIT 1")
        email_row = cursor.fetchone()
        alert_email = email_row[0] if email_row else None
        alert_enabled = bool(email_row[1]) if email_row and len(email_row) > 1 else False
        
        # Get latest record
        cursor.execute("SELECT timestamp, vehicle_count FROM traffic_history ORDER BY timestamp DESC LIMIT 1")
        latest = cursor.fetchone()
        
        # Get database file size
        db_size = os.path.getsize(LOCAL_DB_PATH) if os.path.exists(LOCAL_DB_PATH) else 0
        
        conn.close()
        
        return jsonify({
            'traffic_history': history_count,
            'sessions': sessions_count,
            'mode_changes': mode_changes_count,
            'manual_overrides': overrides_count,
            'system_events': events_count,
            'predictions': predictions_count,
            'trained_models': models_count,
            'database_size_kb': round(db_size / 1024, 2),
            'database_path': os.path.abspath(LOCAL_DB_PATH),
            'alert_email': alert_email,
            'alert_enabled': alert_enabled,
            'latest_record': {
                'timestamp': latest[0] if latest else None,
                'vehicles': latest[1] if latest else None
            } if latest else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_email', methods=['POST'])
def save_email():
    """Save email settings with validation"""
    try:
        data = request.get_json()
        alert_email = data.get('alert_email', '').strip()
        alert_enabled = data.get('alert_enabled', False)
        
        logger.info(f"Saving email settings: email='{alert_email}', enabled={alert_enabled}")
        
        # Validate email if provided
        if alert_email:
            is_valid, validation_message = validate_email(alert_email)
            if not is_valid:
                logger.warning(f"Email validation failed: {validation_message}")
                return jsonify({
                    'status': 'error',
                    'message': validation_message
                }), 400
        
        # Save to database
        success, message = save_email_settings(alert_email, alert_enabled)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED']
            })
        else:
            return jsonify({
                'status': 'error',
                'message': message
            }), 500
            
    except Exception as e:
        error_msg = f"Error saving email: {str(e)}"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/save_settings', methods=['POST'])
def save_all_settings():
    """Save all settings from frontend"""
    try:
        data = request.get_json()
        
        # Update Config class with new settings
        success = update_config_from_settings(data)
        
        # Save email settings if provided
        if 'alertEmail' in data:
            alert_email = data.get('alertEmail', '').strip()
            alert_enabled = data.get('alertEnabled', False)
            
            # Validate email if provided
            if alert_email:
                is_valid, validation_message = validate_email(alert_email)
                if not is_valid:
                    logger.warning(f"Email validation failed: {validation_message}")
                    # Continue with other settings but return warning
                    return jsonify({
                        'status': 'warning',
                        'message': f'Settings saved but email validation failed: {validation_message}'
                    })
            
            save_email_settings(alert_email, alert_enabled)
        
        if success:
            return jsonify({'status': 'success', 'message': 'Settings saved successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save settings'}), 500
            
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return jsonify({'error': str(e)}), 500

# FIXED: Test alert endpoint with detailed diagnostics
@app.route('/test_alert')
def test_alert():
    """
    Send a test alert with detailed diagnostics
    Returns JSON with clear explanation of what happened
    """
    logger.info("=== TEST ALERT REQUESTED ===")
    
    # Check if email is configured
    if not EMAIL_CONFIG['ALERT_EMAIL']:
        logger.warning("Test alert failed: No alert email configured")
        return jsonify({
            'success': False,
            'message': 'No alert email configured',
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': None,
                'smtp_configured': bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD']),
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        }), 200  # Return 200 even on failure, with clear message
    
    # Check if alerts are enabled globally
    if not EMAIL_CONFIG['ALERT_ENABLED']:
        logger.warning("Test alert failed: Alerts are disabled globally")
        return jsonify({
            'success': False,
            'message': 'Alerts are disabled globally',
            'details': {
                'alert_enabled': False,
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
            }
        }), 200
    
    # Check SMTP credentials
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    if not smtp_configured:
        logger.warning("Test alert failed: SMTP credentials not configured")
        return jsonify({
            'success': False,
            'message': 'SMTP credentials not configured',
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': False,
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        }), 200
    
    # Send test alert
    success, message = send_alert(
        "Test Alert",
        f"This is a test alert from your Traffic Management System.\n\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Session: {traffic_state.session_id}\n"
        f"Mode: {traffic_state.control_mode}\n"
        f"Vehicles: {traffic_state.total_vehicles}",
        None  # No alert type for test
    )
    
    if success:
        logger.info(f"Test alert sent successfully to {EMAIL_CONFIG['ALERT_EMAIL']}")
        return jsonify({
            'success': True,
            'message': f'Test alert sent successfully to {EMAIL_CONFIG["ALERT_EMAIL"]}',
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': smtp_configured,
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        })
    else:
        logger.error(f"Test alert failed: {message}")
        return jsonify({
            'success': False,
            'message': message,
            'details': {
                'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
                'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
                'smtp_configured': smtp_configured,
                'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
                'smtp_port': EMAIL_CONFIG['SMTP_PORT']
            }
        }), 200  # Return 200 even on failure, with clear error message

# NEW: Email status debug endpoint
@app.route('/email_status')
def email_status():
    """
    Debug endpoint to check email configuration status
    Returns JSON with current email settings and diagnostics
    """
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    
    # Test SMTP connection if credentials are configured
    smtp_test_result = None
    if smtp_configured and EMAIL_CONFIG['ALERT_EMAIL']:
        try:
            # Quick SMTP connection test without sending email
            logger.info("Testing SMTP connection...")
            server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'], timeout=10)
            server.starttls()
            server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
            server.quit()
            smtp_test_result = "Connection successful"
            logger.info("SMTP connection test successful")
        except Exception as e:
            smtp_test_result = f"Connection failed: {str(e)}"
            logger.error(f"SMTP connection test failed: {e}")
    
    return jsonify({
        'alert_enabled': EMAIL_CONFIG['ALERT_ENABLED'],
        'alert_email': EMAIL_CONFIG['ALERT_EMAIL'],
        'smtp_configured': smtp_configured,
        'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
        'smtp_port': EMAIL_CONFIG['SMTP_PORT'],
        'smtp_username': EMAIL_CONFIG['SMTP_USERNAME'][:3] + '...' if EMAIL_CONFIG['SMTP_USERNAME'] else None,  # Partial for security
        'smtp_test': smtp_test_result,
        'alert_types': Config.ALERT_TYPES,
        'database_settings': {
            'has_email_in_db': False  # Will be updated if we check
        }
    })

# NEW: SMTP test endpoint (optional, for debugging)
@app.route('/test_smtp')
def test_smtp():
    """
    Test SMTP connection without sending email
    Useful for debugging connection issues
    """
    smtp_configured = bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])
    
    if not smtp_configured:
        return jsonify({
            'success': False,
            'message': 'SMTP credentials not configured',
            'smtp_configured': False
        })
    
    try:
        logger.info(f"Testing SMTP connection to {EMAIL_CONFIG['SMTP_SERVER']}:{EMAIL_CONFIG['SMTP_PORT']}")
        
        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'], timeout=15)
        server.starttls()
        server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
        server.quit()
        
        logger.info("SMTP connection test successful")
        return jsonify({
            'success': True,
            'message': 'SMTP connection successful',
            'smtp_configured': True,
            'smtp_server': EMAIL_CONFIG['SMTP_SERVER'],
            'smtp_port': EMAIL_CONFIG['SMTP_PORT']
        })
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"SMTP Authentication Failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'message': error_msg,
            'smtp_configured': True
        })
        
    except Exception as e:
        error_msg = f"SMTP Connection Failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'success': False,
            'message': error_msg,
            'smtp_configured': True
        })

@app.route('/export/pdf')
def export_pdf():
    """Export traffic data as PDF"""
    try:
        # Get data
        history = get_traffic_history(100)
        
        if not history:
            return jsonify({'error': 'No data to export'}), 404
        
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        elements = []
        
        # Add title
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        elements.append(Paragraph("Traffic Management System Report", title_style))
        elements.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        timestamp = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        elements.append(timestamp)
        elements.append(Spacer(1, 0.25*inch))
        
        # Create table data
        table_data = [['Time', 'Vehicles', 'Density', 'Signal', 'Cars', 'Mcycles', 'Buses', 'Trucks']]
        
        for item in history[:50]:
            details = item.get('vehicle_details', {})
            table_data.append([
                datetime.fromisoformat(item['timestamp']).strftime('%H:%M:%S'),
                str(item.get('vehicle_count', 0)),
                item.get('density', 'LOW'),
                item.get('signal_state', 'RED'),
                str(details.get('car', 0)),
                str(details.get('motorcycle', 0)),
                str(details.get('bus', 0)),
                str(details.get('truck', 0))
            ])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'traffic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/csv')
def export_csv():
    """Export traffic data as CSV"""
    try:
        history = get_traffic_history(1000)
        
        if not history:
            return jsonify({'error': 'No data to export'}), 404
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Timestamp', 'Total Vehicles', 'Density', 'Signal State', 
                        'Cars', 'Motorcycles', 'Buses', 'Trucks', 'Control Mode'])
        
        # Write data
        for item in history:
            details = item.get('vehicle_details', {})
            writer.writerow([
                item['timestamp'],
                item.get('vehicle_count', 0),
                item.get('density', 'LOW'),
                item.get('signal_state', 'RED'),
                details.get('car', 0),
                details.get('motorcycle', 0),
                details.get('bus', 0),
                details.get('truck', 0),
                item.get('control_mode', 'AUTO')
            ])
        
        # Prepare response
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=traffic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
        )
        
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/export/json')
def export_json():
    """Export traffic data as JSON"""
    try:
        history = get_traffic_history(1000)
        
        if not history:
            return jsonify({'error': 'No data to export'}), 404
        
        return Response(
            json.dumps(history, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename=traffic_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'}
        )
        
    except Exception as e:
        logger.error(f"JSON export error: {e}")
        return jsonify({'error': str(e)}), 500

def start_signal_controller():
    """Start the continuous signal controller thread"""
    traffic_state.signal_thread = threading.Thread(target=signal_controller_loop)
    traffic_state.signal_thread.daemon = True
    traffic_state.signal_thread.start()
    logger.info("Signal controller thread launched")

def cleanup():
    """Cleanup function for graceful shutdown"""
    logger.info("Shutting down...")
    traffic_state.log_event("INFO", "System shutdown initiated")
    traffic_state.signal_running = False
    time.sleep(0.5)
    
    # Store final data point
    logger.info("Storing final data point...")
    store_traffic_data(force=True)
    
    # Update session status
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET end_time = ?, status = ? WHERE session_id = ?",
            (datetime.now().isoformat(), 'completed', traffic_state.session_id)
        )
        conn.commit()
        conn.close()
        logger.info(f"Session {traffic_state.session_id} completed")
    except Exception as e:
        logger.error(f"Failed to update session: {e}")
    
    reset_for_new_input()
    cv2.destroyAllWindows()
    logger.info("Cleanup complete")

def reset_system():
    """Reset system but preserve signal state"""
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
        
        # Reset mode
        traffic_state.current_mode = "STANDBY"
        
        logger.info("System reset complete - signal continues running")

# Register cleanup function
atexit.register(cleanup)


if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load models
    load_models()
    
    # Start the continuous signal controller thread
    start_signal_controller()
    
    # Try to load saved prediction model
    model_loaded = load_saved_model()
    
    if model_loaded:
        logger.info("Using saved prediction model from disk")
        traffic_state.current_predictions = generate_predictions()
    else:
        logger.info("No saved model found, using demo predictions")
        traffic_state.current_predictions = generate_demo_predictions()
    
    # Analyze hourly patterns
    analyze_hourly_patterns()
    
    # Log startup
    traffic_state.log_event("INFO", "System started")
    
    logger.info(
        f"Email configuration: Enabled={EMAIL_CONFIG['ALERT_ENABLED']}, "
        f"Email='{EMAIL_CONFIG['ALERT_EMAIL']}'"
    )
    
    logger.info(
        f"SMTP configured: {bool(EMAIL_CONFIG['SMTP_USERNAME'] and EMAIL_CONFIG['SMTP_PASSWORD'])}"
    )

    # Use environment PORT for Render
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
