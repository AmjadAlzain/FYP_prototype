"""
Enhanced ESP32-S3-EYE Container Detection GUI
Multi-mode interface: ESP32, Laptop Camera, and Video Upload
Professional interface for container damage detection
"""

import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
import json
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QTextEdit, QGroupBox, QGridLayout,
    QComboBox, QSlider, QProgressBar, QFileDialog, QSplitter,
    QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QMessageBox, QSpinBox, QCheckBox, QSizePolicy
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, QTimer, Qt, QSize, QPropertyAnimation, 
    QEasingCurve, QRect
)
from PyQt6.QtGui import (
    QPixmap, QImage, QFont, QPalette, QColor, QPainter, QPen,
    QBrush, QIcon, QAction
)

# Import our custom modules
from local_inference_enhanced import EnhancedInferenceEngine, create_inference_engine, DetectionResult
from camera_manager import CameraManager

# Modern color scheme
COLORS = {
    'primary': '#0078D4',
    'secondary': '#6B73FF',
    'success': '#107C10',
    'warning': '#FFB900',
    'danger': '#D13438',
    'background': '#F5F5F5',
    'surface': '#FFFFFF',
    'text': '#323130',
    'accent': '#00BCF2'
}

# Simple detection colors - Red for any damage, Green for no damage
BOX_COLOR_DAMAGED = (0, 0, 255)   # Red for any damage  
BOX_COLOR_HEALTHY = (0, 255, 0)   # Green for no damage

class InferenceWorker(QThread):
    """Background thread for running inference"""
    
    frame_processed = pyqtSignal(dict)  # Results
    error_occurred = pyqtSignal(str)
    
    def __init__(self, inference_engine):
        super().__init__()
        self.inference_engine = inference_engine
        self.frame_queue = []
        self.queue_lock = threading.Lock()
        self.running = False
        
    def add_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        with self.queue_lock:
            # Keep only latest frame to avoid lag
            self.frame_queue = [frame]
    
    def run(self):
        """Process frames in background"""
        self.running = True
        
        while self.running:
            frame = None
            
            with self.queue_lock:
                if self.frame_queue:
                    frame = self.frame_queue.pop(0)
            
            if frame is not None:
                try:
                    # Run inference using enhanced engine
                    start_time = time.time()
                    result = self.inference_engine.process_full_image(frame)
                    processing_time = time.time() - start_time
                    
                    # Add timing info
                    result['processing_time'] = processing_time
                    result['frame'] = frame
                    
                    self.frame_processed.emit(result)
                    
                except Exception as e:
                    self.error_occurred.emit(f"Inference error: {str(e)}")
            
            self.msleep(33)  # ~30 FPS max
    
    def stop(self):
        """Stop the worker thread"""
        self.running = False

class BatchProcessingWorker(QThread):
    """Background thread for batch video processing"""
    
    progress_updated = pyqtSignal(int, int)  # current_frame, total_frames
    processing_completed = pyqtSignal(dict)  # final results
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path: str, inference_engine):
        super().__init__()
        self.video_path = video_path
        self.inference_engine = inference_engine
        self.running = False
        
    def run(self):
        """Process entire video frame by frame"""
        self.running = True
        results = []
        
        try:
            print(f"🎬 Starting batch processing: {self.video_path}")
            
            # Open video file
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Failed to open video: {self.video_path}")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📊 Video info: {total_frames} frames at {fps:.1f} FPS")
            
            frame_count = 0
            processed_count = 0
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break  # End of video
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process every 5th frame to speed up batch processing
                if frame_count % 5 == 0:
                    try:
                        # Run inference on frame
                        result = self.inference_engine.process_full_image(frame_rgb)
                        
                        # Add frame info
                        result['frame_number'] = frame_count
                        result['timestamp'] = frame_count / fps if fps > 0 else 0
                        
                        results.append(result)
                        processed_count += 1
                        
                        # Emit progress
                        self.progress_updated.emit(frame_count, total_frames)
                        
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                
                frame_count += 1
                
                # Update progress every 10 frames
                if frame_count % 10 == 0:
                    self.progress_updated.emit(frame_count, total_frames)
            
            cap.release()
            
            # Create summary
            summary = {
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'results': results,
                'video_path': self.video_path,
                'fps': fps
            }
            
            print(f"✅ Batch processing completed: {processed_count} frames processed")
            self.processing_completed.emit(summary)
            
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            print(f"❌ {error_msg}")
            self.error_occurred.emit(error_msg)
    
    def stop(self):
        """Stop batch processing"""
        self.running = False

class CameraDisplay(QLabel):
    """Enhanced camera display with detection overlays"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setStyleSheet(f"""
            QLabel {{
                border: 3px solid {COLORS['primary']};
                background-color: #000000;
                border-radius: 15px;
            }}
        """)
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("📷 Select camera source to begin")
        self.setFont(QFont("Segoe UI", 16))
        
        # Display state
        self.current_frame = None
        self.detection_results = None
        self.show_overlays = True
        self.recording = False
        
    def update_frame(self, frame: np.ndarray, detection_results: Dict = None):
        """Update display with new frame and detections"""
        self.current_frame = frame.copy()
        self.detection_results = detection_results
        self.render_display()
    
    def set_overlay_visibility(self, visible: bool):
        """Toggle detection overlay visibility"""
        self.show_overlays = visible
        self.render_display()
    
    def set_recording_status(self, recording: bool):
        """Set recording status indicator"""
        self.recording = recording
        self.render_display()
    
    def render_display(self):
        """Render the display with overlays"""
        if self.current_frame is None:
            return
        
        display_frame = self.current_frame.copy()
        
        # Draw detection overlays
        if self.show_overlays and self.detection_results:
            display_frame = self.draw_detections(display_frame)
        
        # Add recording indicator
        if self.recording:
            self.add_recording_indicator(display_frame)
        
        # Convert to QPixmap and display
        self.display_image(display_frame)
    
    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection overlays on frame"""
        result_frame = frame.copy()
        detections = self.detection_results.get('detections', [])
        
        for detection in detections:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            
            # NEW APPROACH: Use damage_type and is_damaged instead of class_name
            damage_type = detection.get('damage_type', detection.get('class_name', 'unknown'))
            is_damaged = detection.get('is_damaged', damage_type != 'no_damage')
            confidence = detection['confidence']
            
            # Choose color based on damage status - GREEN for healthy, RED for damaged
            if is_damaged:
                color = (0, 0, 255)  # Red for damaged containers
                text_color = (255, 255, 255)  # White text
            else:
                color = (0, 255, 0)  # Green for undamaged containers  
                text_color = (0, 0, 0)    # Black text
            
            # Draw thick container bounding box
            thickness = 4
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Prepare label text based on damage status
            if is_damaged:
                if damage_type == 'axis':
                    label = "AXIS DAMAGE"
                elif damage_type == 'concave':
                    label = "CONCAVE DAMAGE"
                elif damage_type == 'dentado':
                    label = "DENTADO DAMAGE"
                elif damage_type == 'perforation':
                    label = "PERFORATION DAMAGE"
                else:
                    label = f"{damage_type.upper()} DAMAGE"
            else:
                label = "NO DAMAGE"
            
            # Add confidence to label
            label_with_conf = f"{label} ({confidence:.0%})"
            
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label_with_conf, font, font_scale, font_thickness)
            
            # Position text above the container box
            text_x = x
            text_y = max(label_h + 15, y - 15)  # Ensure text doesn't go off screen
            
            # Draw text background rectangle
            padding = 8
            bg_x1 = text_x - padding
            bg_y1 = text_y - label_h - padding
            bg_x2 = text_x + label_w + padding
            bg_y2 = text_y + baseline + padding
            
            # Semi-transparent background
            overlay = result_frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            alpha = 0.85
            cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
            
            # Draw text
            cv2.putText(result_frame, label_with_conf, (text_x, text_y), 
                       font, font_scale, text_color, font_thickness)
        
        # Add summary info
        self.add_summary_overlay(result_frame)
        
        return result_frame
    
    def add_summary_overlay(self, frame: np.ndarray):
        """Add summary information overlay"""
        if not self.detection_results:
            return
        
        # Summary text
        num_detections = self.detection_results.get('num_detections', 0)
        damage_detected = self.detection_results.get('damage_detected', False)
        processing_time = self.detection_results.get('processing_time', 0)
        
        # Background panel
        panel_height = 100
        panel_width = 300
        panel_x = frame.shape[1] - panel_width - 20
        panel_y = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add border
        border_color = (0, 120, 212) if not damage_detected else (69, 53, 220)  # BGR format
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     border_color, 2)
        
        # Summary text
        text_x = panel_x + 15
        text_y = panel_y + 25
        line_height = 20
        
        # Status
        status = "🔍 DAMAGE DETECTED" if damage_detected else "✅ HEALTHY"
        color = (220, 53, 69) if damage_detected else (40, 167, 69)
        cv2.putText(frame, status, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Detection count
        cv2.putText(frame, f"Detections: {num_detections}", 
                   (text_x, text_y + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Processing time
        cv2.putText(frame, f"Process: {processing_time*1000:.1f}ms", 
                   (text_x, text_y + 2*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, 
                   (text_x, text_y + 3*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def add_recording_indicator(self, frame: np.ndarray):
        """Add recording indicator"""
        # Red recording dot
        center = (30, 30)
        radius = 12
        cv2.circle(frame, center, radius, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (50, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def display_image(self, frame: np.ndarray):
        """Convert numpy array to QPixmap and display"""
        height, width, channels = frame.shape
        bytes_per_line = channels * width
        
        q_image = QImage(frame.data, width, height, 
                        bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        
        self.setPixmap(scaled_pixmap)

class ESP32ModeWidget(QWidget):
    """ESP32-S3-EYE mode interface"""
    
    command_sent = pyqtSignal(str)
    
    def __init__(self, camera_manager, inference_engine):
        super().__init__()
        self.camera_manager = camera_manager
        self.inference_engine = inference_engine
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Connection controls
        conn_group = QGroupBox("🔌 ESP32-S3-EYE Connection")
        conn_layout = QHBoxLayout()
        
        self.port_combo = QComboBox()
        self.refresh_ports_btn = QPushButton("🔄")
        self.connect_btn = QPushButton("Connect")
        
        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_combo, 1)
        conn_layout.addWidget(self.refresh_ports_btn)
        conn_layout.addWidget(self.connect_btn)
        
        conn_group.setLayout(conn_layout)
        
        # Device controls
        controls_group = QGroupBox("🎮 Device Controls")
        controls_layout = QGridLayout()
        
        self.freeze_btn = QPushButton("📸 Freeze Frame")
        self.detect_btn = QPushButton("🔍 Toggle Detection")
        self.train_damage_btn = QPushButton("❌ Train: Damage")
        self.train_healthy_btn = QPushButton("✅ Train: Healthy")
        
        controls_layout.addWidget(self.freeze_btn, 0, 0)
        controls_layout.addWidget(self.detect_btn, 0, 1)
        controls_layout.addWidget(self.train_damage_btn, 1, 0)
        controls_layout.addWidget(self.train_healthy_btn, 1, 1)
        
        controls_group.setLayout(controls_layout)
        
        # Status panel
        status_group = QGroupBox("📊 ESP32 Status")
        status_layout = QGridLayout()
        
        self.connection_label = QLabel("❌ Disconnected")
        self.fps_label = QLabel("FPS: --")
        self.memory_label = QLabel("Memory: --")
        self.detections_label = QLabel("Detections: --")
        
        status_layout.addWidget(self.connection_label, 0, 0, 1, 2)
        status_layout.addWidget(self.fps_label, 1, 0)
        status_layout.addWidget(self.memory_label, 1, 1)
        status_layout.addWidget(self.detections_label, 2, 0, 1, 2)
        
        status_group.setLayout(status_layout)
        
        layout.addWidget(conn_group)
        layout.addWidget(controls_group)
        layout.addWidget(status_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connect signals
        self.refresh_ports_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self.connect_esp32)
        self.freeze_btn.clicked.connect(lambda: self.command_sent.emit("BTN:1"))
        self.detect_btn.clicked.connect(lambda: self.command_sent.emit("BTN:2"))
        self.train_damage_btn.clicked.connect(lambda: self.command_sent.emit("BTN:3"))
        self.train_healthy_btn.clicked.connect(lambda: self.command_sent.emit("BTN:4"))
        
        self.refresh_ports()
    
    def refresh_ports(self):
        """Refresh available serial ports"""
        self.port_combo.clear()
        sources = self.camera_manager.get_available_sources()
        
        esp32_sources = [name for name, info in sources.items() if info['type'] == 'esp32']
        
        if esp32_sources:
            for source in esp32_sources:
                self.port_combo.addItem(source)
        else:
            self.port_combo.addItem("No ESP32 devices found")
    
    def connect_esp32(self):
        """Connect to selected ESP32 device"""
        selected = self.port_combo.currentText()
        if "not found" in selected:
            return
        
        if self.camera_manager.switch_to_source(selected):
            self.connection_label.setText("✅ Connected")
            self.connect_btn.setText("Disconnect")
        else:
            self.connection_label.setText("❌ Connection failed")

class LaptopCameraModeWidget(QWidget):
    """Laptop camera mode interface"""
    
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    
    def __init__(self, camera_manager, inference_engine):
        super().__init__()
        self.camera_manager = camera_manager
        self.inference_engine = inference_engine
        self.recording = False
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Camera selection
        camera_group = QGroupBox("📷 Laptop Camera")
        camera_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.start_camera_btn = QPushButton("Start Camera")
        
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo, 1)
        camera_layout.addWidget(self.start_camera_btn)
        
        camera_group.setLayout(camera_layout)
        
        # Optional recording controls
        recording_group = QGroupBox("🎥 Optional Demo Recording")
        recording_layout = QVBoxLayout()
        
        # Add explanation label
        info_label = QLabel("📋 Primary function: Live real-time inference\n💡 Optional: Record demos for presentations")
        info_label.setStyleSheet("color: #666666; font-style: italic; margin: 5px;")
        recording_layout.addWidget(info_label)
        
        self.record_btn = QPushButton("🔴 Start Optional Recording")
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
            }}
        """)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        self.timer_label = QLabel("Ready to record")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        recording_layout.addWidget(self.record_btn)
        recording_layout.addWidget(self.progress_bar)
        recording_layout.addWidget(self.timer_label)
        
        recording_group.setLayout(recording_layout)
        
        # Camera settings
        settings_group = QGroupBox("⚙️ Camera Settings")
        settings_layout = QGridLayout()
        
        self.fps_label = QLabel("Target FPS:")
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(5, 30)
        self.fps_slider.setValue(15)
        self.fps_value_label = QLabel("15")
        
        self.overlay_checkbox = QCheckBox("Show Detection Overlays")
        self.overlay_checkbox.setChecked(True)
        
        settings_layout.addWidget(self.fps_label, 0, 0)
        settings_layout.addWidget(self.fps_slider, 0, 1)
        settings_layout.addWidget(self.fps_value_label, 0, 2)
        settings_layout.addWidget(self.overlay_checkbox, 1, 0, 1, 3)
        
        settings_group.setLayout(settings_layout)
        
        layout.addWidget(camera_group)
        layout.addWidget(recording_group)
        layout.addWidget(settings_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connect signals
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.fps_slider.valueChanged.connect(self.update_fps_label)
        
        self.refresh_cameras()
        
        # Recording timer
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_progress)
        self.recording_start_time = None
        self.recording_duration = 120  # 2 minutes in seconds
    
    def refresh_cameras(self):
        """Refresh available cameras"""
        self.camera_combo.clear()
        sources = self.camera_manager.get_available_sources()
        
        laptop_sources = [name for name, info in sources.items() if info['type'] == 'laptop']
        
        if laptop_sources:
            for source in laptop_sources:
                self.camera_combo.addItem(source)
        else:
            self.camera_combo.addItem("No cameras found")
    
    def start_camera(self):
        """Start laptop camera"""
        selected = self.camera_combo.currentText()
        if "not found" in selected:
            return
        
        if self.start_camera_btn.text() == "Start Camera":
            # Start camera
            if self.camera_manager.switch_to_source(selected):
                self.start_camera_btn.setText("Stop Camera")
                print(f"✅ Laptop camera started: {selected}")
            else:
                QMessageBox.warning(self, "Error", "Failed to start camera")
        else:
            # Stop camera
            self.camera_manager.stop_current_source()
            self.start_camera_btn.setText("Start Camera")
            print("⏹️ Laptop camera stopped")
    
    def toggle_recording(self):
        """Start/stop 2-minute recording"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start 2-minute recording"""
        self.recording = True
        self.recording_start_time = time.time()
        
        self.record_btn.setText("⏹️ Stop Recording")
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
            }}
        """)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.recording_duration)
        self.progress_bar.setValue(0)
        
        self.recording_timer.start(1000)  # Update every second
        self.recording_started.emit()
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        self.recording_timer.stop()
        
        self.record_btn.setText("🔴 Start 2-Minute Recording")
        self.record_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
            }}
        """)
        
        self.progress_bar.setVisible(False)
        self.timer_label.setText("Recording saved")
        
        self.recording_stopped.emit()
    
    def update_recording_progress(self):
        """Update recording progress"""
        if not self.recording or not self.recording_start_time:
            return
        
        elapsed = time.time() - self.recording_start_time
        remaining = max(0, self.recording_duration - elapsed)
        
        self.progress_bar.setValue(int(elapsed))
        
        minutes = int(remaining) // 60
        seconds = int(remaining) % 60
        self.timer_label.setText(f"Recording: {minutes:02d}:{seconds:02d} remaining")
        
        # Auto-stop at 2 minutes
        if elapsed >= self.recording_duration:
            self.stop_recording()
    
    def update_fps_label(self, value):
        """Update FPS label"""
        self.fps_value_label.setText(str(value))

class VideoUploadModeWidget(QWidget):
    """Video upload and processing mode"""
    
    video_loaded = pyqtSignal(str)
    processing_started = pyqtSignal()
    processing_completed = pyqtSignal(dict)
    
    def __init__(self, camera_manager, inference_engine):
        super().__init__()
        self.camera_manager = camera_manager
        self.inference_engine = inference_engine
        self.current_video_path = None
        self.processing_results = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Video upload
        upload_group = QGroupBox("📹 Video Upload")
        upload_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("📁 Select Video File")
        self.video_path_label = QLabel("No video selected")
        
        upload_layout.addWidget(self.upload_btn)
        upload_layout.addWidget(self.video_path_label, 1)
        
        upload_group.setLayout(upload_layout)
        
        # Video controls
        controls_group = QGroupBox("🎮 Video Playback")
        controls_layout = QVBoxLayout()
        
        # Playback buttons
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Play")
        self.pause_btn = QPushButton("⏸️ Pause")
        self.stop_btn = QPushButton("⏹️ Stop")
        
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.pause_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_layout.addStretch()
        
        # Progress slider
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setEnabled(False)
        
        # Time labels
        time_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.total_time_label = QLabel("00:00")
        time_layout.addWidget(self.current_time_label)
        time_layout.addStretch()
        time_layout.addWidget(self.total_time_label)
        
        controls_layout.addLayout(playback_layout)
        controls_layout.addWidget(self.video_slider)
        controls_layout.addLayout(time_layout)
        
        controls_group.setLayout(controls_layout)
        
        # Processing controls
        processing_group = QGroupBox("🔬 Analysis & Processing")
        processing_layout = QVBoxLayout()
        
        self.process_btn = QPushButton("🚀 Start Batch Analysis")
        self.process_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
            }}
        """)
        
        self.processing_progress = QProgressBar()
        self.processing_progress.setVisible(False)
        
        self.processing_label = QLabel("Ready for analysis")
        self.processing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        processing_layout.addWidget(self.process_btn)
        processing_layout.addWidget(self.processing_progress)
        processing_layout.addWidget(self.processing_label)
        
        processing_group.setLayout(processing_layout)
        
        # Export controls
        export_group = QGroupBox("📤 Export Results")
        export_layout = QHBoxLayout()
        
        self.export_video_btn = QPushButton("🎬 Export Annotated Video")
        self.export_csv_btn = QPushButton("📊 Export CSV Report")
        self.export_summary_btn = QPushButton("📋 Export Summary")
        
        export_layout.addWidget(self.export_video_btn)
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_summary_btn)
        
        export_group.setLayout(export_layout)
        
        layout.addWidget(upload_group)
        layout.addWidget(controls_group)
        layout.addWidget(processing_group)
        layout.addWidget(export_group)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connect signals
        self.upload_btn.clicked.connect(self.select_video_file)
        self.play_btn.clicked.connect(self.play_video)
        self.pause_btn.clicked.connect(self.pause_video)
        self.stop_btn.clicked.connect(self.stop_video)
        self.process_btn.clicked.connect(self.start_batch_processing)
        
        # Export buttons
        self.export_video_btn.clicked.connect(self.export_annotated_video)
        self.export_csv_btn.clicked.connect(self.export_csv_report)
        self.export_summary_btn.clicked.connect(self.export_summary_report)
        
        # Initially disable controls
        self.set_controls_enabled(False)
    
    def select_video_file(self):
        """Select video file for upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            if self.camera_manager.load_video_file(file_path):
                self.current_video_path = file_path
                self.video_path_label.setText(Path(file_path).name)
                self.set_controls_enabled(True)
                self.video_loaded.emit(file_path)
            else:
                QMessageBox.warning(self, "Error", "Failed to load video file")
    
    def set_controls_enabled(self, enabled: bool):
        """Enable/disable video controls"""
        self.play_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.video_slider.setEnabled(enabled)
        self.process_btn.setEnabled(enabled)
    
    def play_video(self):
        """Play video with live inference"""
        video_controls = self.camera_manager.get_video_controls()
        if video_controls:
            video_controls.play()
            print("▶️ Video playback started with live inference")
            self.processing_label.setText("🔍 Live inference active during playback")
    
    def pause_video(self):
        """Pause video"""
        video_controls = self.camera_manager.get_video_controls()
        if video_controls:
            video_controls.pause()
    
    def stop_video(self):
        """Stop video"""
        video_controls = self.camera_manager.get_video_controls()
        if video_controls:
            video_controls.pause()
            video_controls.seek_frame(0)
    
    def start_batch_processing(self):
        """Start batch processing of video"""
        if not self.current_video_path:
            QMessageBox.warning(self, "Error", "No video loaded")
            return
        
        self.process_btn.setEnabled(False)
        self.processing_progress.setVisible(True)
        self.processing_label.setText("Processing video frames...")
        
        # Get video info for progress tracking
        video_controls = self.camera_manager.get_video_controls()
        if video_controls:
            current_frame, total_frames = video_controls.get_progress()
            self.processing_progress.setMaximum(total_frames)
            self.processing_progress.setValue(0)
        
        # Start processing in background thread
        self.batch_worker = BatchProcessingWorker(self.current_video_path, self.inference_engine)
        self.batch_worker.progress_updated.connect(self.update_batch_progress)
        self.batch_worker.processing_completed.connect(self.on_batch_completed)
        self.batch_worker.error_occurred.connect(self.on_batch_error)
        self.batch_worker.start()
        
        self.processing_started.emit()
    
    def update_batch_progress(self, current_frame: int, total_frames: int):
        """Update batch processing progress"""
        self.processing_progress.setValue(current_frame)
        progress_percent = (current_frame / total_frames) * 100 if total_frames > 0 else 0
        self.processing_label.setText(f"Processing: {current_frame}/{total_frames} frames ({progress_percent:.1f}%)")
    
    def on_batch_completed(self, results: Dict):
        """Handle batch processing completion"""
        self.processing_results = results
        self.process_btn.setEnabled(True)
        self.processing_progress.setVisible(False)
        
        # Show completion message
        total_frames = results.get('total_frames', 0)
        processed_frames = results.get('processed_frames', 0)
        detections = len(results.get('results', []))
        
        self.processing_label.setText(
            f"✅ Complete: {processed_frames}/{total_frames} frames processed, {detections} detections found"
        )
        
        print(f"✅ Batch processing completed: {processed_frames} frames, {detections} detections")
        self.processing_completed.emit(results)
    
    def on_batch_error(self, error: str):
        """Handle batch processing error"""
        self.process_btn.setEnabled(True)
        self.processing_progress.setVisible(False)
        self.processing_label.setText(f"❌ Error: {error}")
        
        print(f"❌ Batch processing error: {error}")
        QMessageBox.critical(self, "Batch Processing Error", 
                           f"Failed to process video:\n{error}")
    
    def export_annotated_video(self):
        """Export video with detection annotations"""
        if not self.processing_results:
            QMessageBox.warning(self, "Error", "No processing results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Video", "",
            "Video Files (*.mp4);;All Files (*)"
        )
        
        if file_path:
            # TODO: Implement video export
            QMessageBox.information(self, "Success", f"Video exported to {file_path}")
    
    def export_csv_report(self):
        """Export CSV report of detections"""
        if not self.processing_results:
            QMessageBox.warning(self, "Error", "No processing results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV Report", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            # TODO: Implement CSV export
            QMessageBox.information(self, "Success", f"Report exported to {file_path}")
    
    def export_summary_report(self):
        """Export summary report"""
        if not self.processing_results:
            QMessageBox.warning(self, "Error", "No processing results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Summary Report", "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            # TODO: Implement PDF export
            QMessageBox.information(self, "Success", f"Summary exported to {file_path}")

class AnalyticsDashboard(QWidget):
    """Analytics and statistics dashboard"""
    
    def __init__(self):
        super().__init__()
        self.detection_history = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Statistics summary
        stats_group = QGroupBox("📊 Detection Statistics")
        stats_layout = QGridLayout()
        
        self.total_detections_label = QLabel("Total Detections: 0")
        self.damage_count_label = QLabel("Damage Detected: 0")
        self.healthy_count_label = QLabel("Healthy Containers: 0")
        self.avg_confidence_label = QLabel("Avg Confidence: 0%")
        
        stats_layout.addWidget(self.total_detections_label, 0, 0)
        stats_layout.addWidget(self.damage_count_label, 0, 1)
        stats_layout.addWidget(self.healthy_count_label, 1, 0)
        stats_layout.addWidget(self.avg_confidence_label, 1, 1)
        
        stats_group.setLayout(stats_layout)
        
        # Detection history table
        history_group = QGroupBox("📋 Detection History")
        history_layout = QVBoxLayout()
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp", "Class", "Confidence", "Position", "Source"
        ])
        
        # Clear history button
        clear_btn = QPushButton("🗑️ Clear History")
        clear_btn.clicked.connect(self.clear_history)
        
        history_layout.addWidget(self.history_table)
        history_layout.addWidget(clear_btn)
        
        history_group.setLayout(history_layout)
        
        layout.addWidget(stats_group)
        layout.addWidget(history_group)
        
        self.setLayout(layout)
    
    def add_detection(self, detection_result: Dict, source: str = "Unknown"):
        """Add detection to history - Updated for new container detection approach"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        for detection in detection_result.get('detections', []):
            # NEW APPROACH: Use damage_type and is_damaged
            damage_type = detection.get('damage_type', detection.get('class_name', 'unknown'))
            is_damaged = detection.get('is_damaged', damage_type != 'no_damage')
            
            self.detection_history.append({
                'timestamp': timestamp,
                'class': damage_type,
                'is_damaged': is_damaged,
                'confidence': detection['confidence'],
                'position': f"({detection['x']}, {detection['y']})",
                'source': source
            })
        
        self.update_display()
    
    def update_display(self):
        """Update analytics display"""
        if not self.detection_history:
            return
        
        # Update statistics
        total = len(self.detection_history)
        damage_count = sum(1 for d in self.detection_history if d['class'] != 'no_damage')
        healthy_count = total - damage_count
        avg_confidence = np.mean([d['confidence'] for d in self.detection_history]) * 100
        
        self.total_detections_label.setText(f"Total Detections: {total}")
        self.damage_count_label.setText(f"Damage Detected: {damage_count}")
        self.healthy_count_label.setText(f"Healthy Containers: {healthy_count}")
        self.avg_confidence_label.setText(f"Avg Confidence: {avg_confidence:.1f}%")
        
        # Update history table
        self.history_table.setRowCount(len(self.detection_history))
        
        for row, detection in enumerate(self.detection_history[-50:]):  # Show last 50
            self.history_table.setItem(row, 0, QTableWidgetItem(detection['timestamp']))
            self.history_table.setItem(row, 1, QTableWidgetItem(detection['class']))
            self.history_table.setItem(row, 2, QTableWidgetItem(f"{detection['confidence']:.2f}"))
            self.history_table.setItem(row, 3, QTableWidgetItem(detection['position']))
            self.history_table.setItem(row, 4, QTableWidgetItem(detection['source']))
    
    def clear_history(self):
        """Clear detection history"""
        self.detection_history.clear()
        self.history_table.setRowCount(0)
        self.update_display()

class ContainerDetectionAppEnhanced(QMainWindow):
    """Enhanced main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.inference_engine = None  # Will be initialized in load_models()
        self.camera_manager = CameraManager()
        
        # Workers
        self.inference_worker = None
        
        # Application state
        self.current_mode = "ESP32"
        
        self.init_ui()
        self.load_models()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Enhanced ESP32-S3-EYE Container Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Modern styling
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {COLORS['primary']};
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 10px;
                background-color: {COLORS['surface']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {COLORS['primary']};
            }}
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['secondary']};
            }}
            QPushButton:disabled {{
                background-color: #CCCCCC;
                color: #666666;
            }}
        """)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Camera display
        self.camera_display = CameraDisplay()
        splitter.addWidget(self.camera_display)
        
        # Right side: Control panels
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Mode selection tabs
        self.mode_tabs = QTabWidget()
        self.mode_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {COLORS['primary']};
                border-radius: 10px;
                background-color: {COLORS['surface']};
            }}
            QTabBar::tab {{
                background-color: {COLORS['background']};
                color: {COLORS['text']};
                padding: 12px 20px;
                margin: 2px;
                border-radius: 8px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['primary']};
                color: white;
            }}
        """)
        
        # Create mode widgets
        self.esp32_widget = ESP32ModeWidget(self.camera_manager, self.inference_engine)
        self.laptop_widget = LaptopCameraModeWidget(self.camera_manager, self.inference_engine)
        self.video_widget = VideoUploadModeWidget(self.camera_manager, self.inference_engine)
        self.analytics_widget = AnalyticsDashboard()
        
        # Add tabs
        self.mode_tabs.addTab(self.esp32_widget, "🔌 ESP32-S3-EYE")
        self.mode_tabs.addTab(self.laptop_widget, "📷 Laptop Camera")
        self.mode_tabs.addTab(self.video_widget, "📹 Video Upload")
        self.mode_tabs.addTab(self.analytics_widget, "📊 Analytics")
        
        right_layout.addWidget(self.mode_tabs)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([800, 600])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 8px;
            }}
        """)
        self.statusBar().showMessage("Enhanced Container Detection System Ready")
        
        # Connect signals
        self.setup_connections()
        
    def setup_connections(self):
        """Setup signal connections"""
        # Camera manager callbacks
        self.camera_manager.set_frame_callback(self.on_frame_received)
        self.camera_manager.set_error_callback(self.on_camera_error)
        
        # Mode-specific connections
        self.esp32_widget.command_sent.connect(self.send_esp32_command)
        self.laptop_widget.recording_started.connect(self.start_recording)
        self.laptop_widget.recording_stopped.connect(self.stop_recording)
        self.video_widget.video_loaded.connect(self.on_video_loaded)
        
        # Tab change
        self.mode_tabs.currentChanged.connect(self.on_mode_changed)
        
    def load_models(self):
        """Load inference models"""
        try:
            print("🔄 Loading inference models...")
            
            # Set correct models directory path
            models_dir = Path(__file__).parent.parent.parent.parent / "models"
            print(f"📁 Models directory: {models_dir}")
            
            self.inference_engine = create_inference_engine(str(models_dir))
            
            if self.inference_engine is not None:
                self.statusBar().showMessage("✅ Models loaded successfully")
                print("✅ TinyNAS + HDC models loaded successfully")
                
                # Initialize inference worker
                self.inference_worker = InferenceWorker(self.inference_engine)
                self.inference_worker.frame_processed.connect(self.on_inference_result)
                self.inference_worker.error_occurred.connect(self.on_inference_error)
                self.inference_worker.start()
                
                print("🔄 Inference worker started")
                
            else:
                self.statusBar().showMessage("❌ Failed to load models")
                QMessageBox.critical(self, "Error", 
                    f"Failed to load inference models from {models_dir}\n\n"
                    "Please ensure model files exist:\n"
                    "- feature_extractor_fp32_best.pth\n"
                    "- module3_hdc_model_embhd.npz")
                
        except Exception as e:
            error_msg = f"Model loading error: {str(e)}"
            print(f"❌ {error_msg}")
            self.statusBar().showMessage("❌ Model loading failed")
            QMessageBox.critical(self, "Error", error_msg)
    
    def on_frame_received(self, frame: np.ndarray):
        """Handle new frame from camera"""
        # Add frame to inference queue
        if self.inference_worker:
            self.inference_worker.add_frame(frame)
        
        # Update display immediately (will be overlaid with detections when ready)
        self.camera_display.update_frame(frame)
    
    def on_inference_result(self, result: Dict):
        """Handle inference results"""
        frame = result.get('frame')
        
        # Update display with detections
        if frame is not None:
            self.camera_display.update_frame(frame, result)
        
        # Add to analytics
        current_tab = self.mode_tabs.currentIndex()
        source_names = ["ESP32", "Laptop Camera", "Video Upload", "Analytics"]
        source = source_names[current_tab] if current_tab < len(source_names) else "Unknown"
        
        self.analytics_widget.add_detection(result, source)
        
        # Update status
        fps = 1.0 / result.get('processing_time', 1.0)
        num_detections = result.get('num_detections', 0)
        self.statusBar().showMessage(
            f"FPS: {fps:.1f} | Detections: {num_detections} | "
            f"Processing: {result.get('processing_time', 0)*1000:.1f}ms"
        )
    
    def on_camera_error(self, error: str):
        """Handle camera errors"""
        self.statusBar().showMessage(f"Camera Error: {error}")
        QMessageBox.warning(self, "Camera Error", error)
    
    def on_inference_error(self, error: str):
        """Handle inference errors"""
        self.statusBar().showMessage(f"Inference Error: {error}")
        print(f"Inference Error: {error}")
    
    def send_esp32_command(self, command: str):
        """Send command to ESP32"""
        self.camera_manager.send_esp32_command(command)
        self.statusBar().showMessage(f"ESP32 Command: {command}")
    
    def start_recording(self):
        """Start recording mode"""
        self.camera_display.set_recording_status(True)
        self.statusBar().showMessage("🔴 Recording 2-minute demo...")
    
    def stop_recording(self):
        """Stop recording mode"""
        self.camera_display.set_recording_status(False)
        self.statusBar().showMessage("Recording saved")
    
    def on_video_loaded(self, video_path: str):
        """Handle video file loaded"""
        self.statusBar().showMessage(f"Video loaded: {Path(video_path).name}")
    
    def on_mode_changed(self, index: int):
        """Handle mode tab change"""
        try:
            modes = ["ESP32-S3-EYE", "Laptop Camera", "Video Upload", "Analytics"]
            if index < len(modes):
                old_mode = getattr(self, 'current_mode', 'Unknown')
                self.current_mode = modes[index]
                
                # Stop current camera source when switching modes
                if hasattr(self, 'camera_manager') and self.camera_manager:
                    try:
                        self.camera_manager.stop_current_source()
                    except Exception as e:
                        print(f"Warning: Error stopping camera source: {e}")
                
                self.statusBar().showMessage(f"Mode: {self.current_mode}")
                print(f"Mode changed: {old_mode} -> {self.current_mode}")
                
        except Exception as e:
            print(f"Error in mode change: {e}")
            self.statusBar().showMessage(f"Error switching mode: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop workers
        if self.inference_worker:
            self.inference_worker.stop()
            self.inference_worker.wait()
        
        # Stop camera
        self.camera_manager.stop_current_source()
        
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Container Detection")
    app.setApplicationVersion("2.0")
    app.setStyle('Fusion')
    
    # Set modern font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = ContainerDetectionAppEnhanced()
    window.show()
    
    print("Enhanced ESP32-S3-EYE Container Detection GUI Started")
    print("Multi-mode interface: ESP32, Laptop Camera, Video Upload")
    print("Professional container damage detection with analytics")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
