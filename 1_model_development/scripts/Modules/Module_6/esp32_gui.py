"""
ESP32-S3-EYE Container Anomaly Detection GUI
Enhanced PyQt6 interface for real-time monitoring and training
Optimized for maritime container damage detection
"""

import sys
import json
import base64
import time
from pathlib import Path
from typing import Optional, Dict, List
import cv2
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QPushButton, QTextEdit, QGroupBox,
    QGridLayout, QProgressBar, QComboBox, QSpinBox,
    QSplitter, QStatusBar, QFrame, QScrollArea, QDialog,
    QDialogButtonBox, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QPainter, QPen, QLinearGradient

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print("Warning: pyserial not installed. Install with: pip install pyserial")
    SERIAL_AVAILABLE = False

# Configuration
DEFAULT_SERIAL_PORT = "COM3"  # Windows default
DEFAULT_BAUD_RATE = 115200
UPDATE_INTERVAL_MS = 100

# Enhanced color mapping for damage types
DAMAGE_COLORS = {
    'axis': (255, 100, 255),      # Bright Pink
    'concave': (128, 0, 255),     # Purple  
    'dentado': (255, 255, 0),     # Yellow
    'perforation': (255, 50, 50), # Red
    'no_damage': (50, 255, 50),   # Green
    'container': (50, 255, 50),   # Green for healthy
    'container_damaged': (255, 50, 50)  # Red for damaged
}

# Damage type descriptions
DAMAGE_DESCRIPTIONS = {
    'axis': 'Structural axis damage',
    'concave': 'Concave deformation', 
    'dentado': 'Dented surface',
    'perforation': 'Hole or puncture',
    'no_damage': 'No visible damage'
}

class SerialWorker(QThread):
    """Enhanced worker thread for ESP32 communication"""
    frame_received = pyqtSignal(np.ndarray)
    detection_received = pyqtSignal(dict)
    status_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)
    training_complete = pyqtSignal(str, bool)
    
    def __init__(self, port: str, baudrate: int = DEFAULT_BAUD_RATE):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False
        
    def run(self):
        """Main communication loop with enhanced error handling"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow ESP32 to initialize
            self.connection_status.emit(True, f"Connected to {self.port}")
            self.running = True
            
            while self.running:
                try:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8').strip()
                        if line:
                            self.parse_message(line)
                    time.sleep(0.01)  # Small delay to prevent CPU overload
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Serial read error: {e}")
                    
        except Exception as e:
            self.connection_status.emit(False, f"Connection failed: {str(e)}")
            
    def parse_message(self, message: str):
        """Enhanced message parser with more message types"""
        try:
            if message.startswith("IMG:"):
                # Format: IMG:width,height,format,base64_data
                parts = message[4:].split(',', 3)
                if len(parts) >= 3:
                    width, height = int(parts[0]), int(parts[1])
                    if len(parts) == 4:
                        img_data = base64.b64decode(parts[3])
                        # Handle different image formats
                        if parts[2] == "RGB565":
                            img_array = np.frombuffer(img_data, dtype=np.uint16)
                            img_array = img_array.reshape((height, width))
                            img_rgb = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR5652RGB)
                        else:  # Assume RGB888
                            img_array = np.frombuffer(img_data, dtype=np.uint8)
                            img_rgb = img_array.reshape((height, width, 3))
                        self.frame_received.emit(img_rgb)
                    
            elif message.startswith("DETECT:"):
                # Format: DETECT:x,y,w,h,class,confidence,timestamp
                parts = message[7:].split(',')
                if len(parts) >= 6:
                    detection = {
                        'x': int(parts[0]),
                        'y': int(parts[1]),
                        'w': int(parts[2]),
                        'h': int(parts[3]),
                        'class': parts[4],
                        'confidence': float(parts[5]),
                        'timestamp': parts[6] if len(parts) > 6 else str(int(time.time()))
                    }
                    self.detection_received.emit(detection)
                    
            elif message.startswith("STATUS:"):
                # Format: STATUS:fps,memory_used,detections_count,temperature
                parts = message[7:].split(',')
                if len(parts) >= 3:
                    status = {
                        'fps': float(parts[0]),
                        'memory_used': int(parts[1]),
                        'detections': int(parts[2]),
                        'temperature': float(parts[3]) if len(parts) > 3 else 0.0
                    }
                    self.status_received.emit(status)
                    
            elif message.startswith("TRAIN_OK:"):
                # Training successful
                class_name = message[10:]
                self.training_complete.emit(class_name, True)
                
            elif message.startswith("TRAIN_ERR:"):
                # Training failed
                error_msg = message[11:]
                self.training_complete.emit(error_msg, False)
                    
        except Exception as e:
            print(f"Message parse error: {e}")
            
    def send_command(self, command: str):
        """Enhanced command sending with confirmation"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{command}\n".encode('utf-8'))
                self.serial_conn.flush()
            except Exception as e:
                print(f"Send command error: {e}")
                
    def stop(self):
        """Graceful shutdown"""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

class CameraWidget(QLabel):
    """Enhanced camera widget with improved visualization"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("""
            border: 3px solid #0078d4; 
            background-color: #000; 
            border-radius: 10px;
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("🎥 Waiting for camera feed...")
        self.setFont(QFont("Arial", 14))
        self.current_frame: Optional[np.ndarray] = None
        self.detections: List[Dict] = []
        self.show_fps = True
        self.current_fps = 0.0
        
    def update_frame(self, frame: np.ndarray):
        """Update with enhanced frame processing"""
        self.current_frame = frame.copy()
        self.update_display()
        
    def add_detection(self, detection: Dict):
        """Add detection with timestamp management"""
        # Remove old detections (older than 3 seconds)
        current_time = time.time()
        self.detections = [d for d in self.detections 
                          if current_time - float(d.get('timestamp', 0)) < 3.0]
        
        detection['timestamp'] = str(current_time)
        self.detections.append(detection)
        self.update_display()
        
    def set_fps(self, fps: float):
        """Update FPS display"""
        self.current_fps = fps
        self.update_display()
        
    def update_display(self):
        """Enhanced display with better graphics"""
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # Draw detections with enhanced styling
        for detection in self.detections:
            x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
            class_name = detection['class']
            confidence = detection['confidence']
            
            color = DAMAGE_COLORS.get(class_name, (255, 255, 255))
            
            # Draw thick bounding box with rounded corners effect
            cv2.rectangle(display_frame, (x-2, y-2), (x + w+2, y + h+2), (0, 0, 0), 4)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
            
            # Enhanced label with background
            description = DAMAGE_DESCRIPTIONS.get(class_name, class_name)
            label = f"{description}"
            conf_label = f"Confidence: {confidence:.1%}"
            
            # Calculate text dimensions
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (cw, ch), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background with padding
            padding = 8
            bg_height = th + ch + padding * 3
            bg_width = max(tw, cw) + padding * 2
            
            cv2.rectangle(display_frame, 
                         (x, y - bg_height - 5), 
                         (x + bg_width, y), 
                         color, -1)
            cv2.rectangle(display_frame, 
                         (x, y - bg_height - 5), 
                         (x + bg_width, y), 
                         (255, 255, 255), 2)
            
            # Draw text
            cv2.putText(display_frame, label, 
                       (x + padding, y - bg_height + th), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, conf_label, 
                       (x + padding, y - padding), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add FPS counter
        if self.show_fps and self.current_fps > 0:
            fps_text = f"FPS: {self.current_fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Convert and display
        height, width, channels = display_frame.shape
        bytes_per_line = channels * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(self.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

class DamageSelectionDialog(QDialog):
    """Dialog for selecting damage type during training"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Damage Type")
        self.setModal(True)
        self.selected_class = None
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select the type of damage detected:"))
        
        self.damage_list = QListWidget()
        
        for i, (damage_type, description) in enumerate(DAMAGE_DESCRIPTIONS.items()):
            if damage_type != 'no_damage':  # Exclude no_damage from selection
                item = QListWidgetItem(f"{damage_type.upper()}: {description}")
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.damage_list.addItem(item)
        
        self.damage_list.itemDoubleClicked.connect(self.accept)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                 QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        layout.addWidget(self.damage_list)
        layout.addWidget(buttons)
        self.setLayout(layout)
        
    def get_selected_class(self):
        """Get selected damage class ID"""
        current_item = self.damage_list.currentItem()
        if current_item:
            return current_item.data(Qt.ItemDataRole.UserRole)
        return None

class ControlPanel(QGroupBox):
    """Enhanced control panel with better organization"""
    
    command_requested = pyqtSignal(str)
    
    def __init__(self):
        super().__init__("🎛️ System Controls")
        self.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #0078d4;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Connection section
        conn_group = QGroupBox("📡 Connection")
        conn_layout = QHBoxLayout()
        
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.port_combo.setMinimumWidth(120)
        
        self.connect_btn = QPushButton("🔌 Connect")
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                border: none;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #45a049, stop:1 #3d8b40); }
        """)
        self.connect_btn.clicked.connect(self.request_connection)
        
        refresh_btn = QPushButton("🔄")
        refresh_btn.clicked.connect(self.refresh_ports)
        refresh_btn.setMaximumWidth(40)
        
        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_combo)
        conn_layout.addWidget(refresh_btn)
        conn_layout.addWidget(self.connect_btn)
        
        conn_group.setLayout(conn_layout)
        
        # Camera controls
        camera_group = QGroupBox("📷 Camera Controls")
        camera_layout = QGridLayout()
        
        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.clicked.connect(lambda: self.command_requested.emit("BTN:1"))
        
        self.toggle_detection_btn = QPushButton("🔍 Toggle Detection")
        self.toggle_detection_btn.clicked.connect(lambda: self.command_requested.emit("BTN:2"))
        
        camera_layout.addWidget(self.capture_btn, 0, 0)
        camera_layout.addWidget(self.toggle_detection_btn, 0, 1)
        
        camera_group.setLayout(camera_layout)
        
        # Training controls
        train_group = QGroupBox("🎓 Training Controls")
        train_layout = QGridLayout()
        
        self.train_healthy_btn = QPushButton("✅ Train: Healthy")
        self.train_healthy_btn.clicked.connect(lambda: self.command_requested.emit("BTN:4"))
        self.train_healthy_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white; padding: 12px; border-radius: 6px; font-weight: bold;
            }
        """)
        
        self.train_damage_btn = QPushButton("❌ Train: Damaged")
        self.train_damage_btn.clicked.connect(self.show_damage_selection)
        self.train_damage_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f44336, stop:1 #da190b);
                color: white; padding: 12px; border-radius: 6px; font-weight: bold;
            }
        """)
        
        self.training_menu_btn = QPushButton("📋 Training Menu")
        self.training_menu_btn.clicked.connect(lambda: self.command_requested.emit("BTN:3"))
        
        train_layout.addWidget(self.train_healthy_btn, 0, 0)
        train_layout.addWidget(self.train_damage_btn, 0, 1)
        train_layout.addWidget(self.training_menu_btn, 1, 0, 1, 2)
        
        train_group.setLayout(train_layout)
        
        layout.addWidget(conn_group)
        layout.addWidget(camera_group)
        layout.addWidget(train_group)
        self.setLayout(layout)
        
    def refresh_ports(self):
        """Enhanced port detection"""
        self.port_combo.clear()
        if SERIAL_AVAILABLE:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                description = port.description
                if "ESP32" in description or "USB" in description:
                    self.port_combo.addItem(f"🔌 {port.device} - {description}")
                else:
                    self.port_combo.addItem(f"📱 {port.device} - {description}")
            if not ports:
                self.port_combo.addItem("❌ No ports found")
        else:
            self.port_combo.addItem("❌ Serial library not available")
            
    def request_connection(self):
        """Request connection with validation"""
        current_text = self.port_combo.currentText()
        if "No ports" in current_text or "not available" in current_text:
            return
            
        port = current_text.split(' - ')[0].split(' ', 1)[1]  # Remove emoji
        self.command_requested.emit(f"CONNECT:{port}")
        self.connect_btn.setText("🔄 Connecting...")
        self.connect_btn.setEnabled(False)
        
    def connection_established(self, success: bool):
        """Update UI based on connection status"""
        if success:
            self.connect_btn.setText("✅ Connected")
            self.connect_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4CAF50, stop:1 #45a049);
                    color: white; padding: 10px; border-radius: 5px; font-weight: bold;
                }
            """)
        else:
            self.connect_btn.setText("🔌 Connect")
            self.connect_btn.setEnabled(True)
            self.connect_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #f44336, stop:1 #da190b);
                    color: white; padding: 10px; border-radius: 5px; font-weight: bold;
                }
            """)
        
    def show_damage_selection(self):
        """Show enhanced damage selection dialog"""
        dialog = DamageSelectionDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            class_id = dialog.get_selected_class()
            if class_id is not None:
                self.command_requested.emit(f"TRAIN:{class_id}")

class StatusPanel(QGroupBox):
    """Enhanced status panel with more metrics"""
    
    def __init__(self):
        super().__init__("📊 System Status")
        self.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #0078d4;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 10px;
            }
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout()
        
        # Connection status with icon
        self.conn_status = QLabel("❌ Disconnected")
        self.conn_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 14px;")
        
        # Performance metrics with icons
        self.fps_label = QLabel("📈 FPS: --")
        self.memory_label = QLabel("💾 Memory: --")
        self.detections_label = QLabel("🔍 Detections: --")
        self.temp_label = QLabel("🌡️ Temp: --")
        
        # Enhanced progress bars
        self.memory_bar = QProgressBar()
        self.memory_bar.setMaximum(100)
        self.memory_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 3px;
            }
        """)
        
        # Layout with better spacing
        layout.addWidget(QLabel("Status:"), 0, 0)
        layout.addWidget(self.conn_status, 0, 1)
        layout.addWidget(self.fps_label, 1, 0)
        layout.addWidget(self.detections_label, 1, 1)
        layout.addWidget(QLabel("Memory Usage:"), 2, 0)
        layout.addWidget(self.memory_bar, 2, 1)
        layout.addWidget(self.memory_label, 3, 0)
        layout.addWidget(self.temp_label, 3, 1)
        
        self.setLayout(layout)
        
    def update_connection_status(self, connected: bool, message: str):
        """Enhanced connection status display"""
        if connected:
            self.conn_status.setText(f"✅ Connected")
            self.conn_status.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 14px;")
        else:
            self.conn_status.setText(f"❌ {message}")
            self.conn_status.setStyleSheet("color: #f44336; font-weight: bold; font-size: 14px;")
            
    def update_metrics(self, status: Dict):
        """Update with enhanced formatting"""
        fps = status.get('fps', 0)
        memory_used = status.get('memory_used', 0)
        detections = status.get('detections', 0)
        temperature = status.get('temperature', 0)
        
        self.fps_label.setText(f"📈 FPS: {fps:.1f}")
        self.memory_label.setText(f"💾 Memory: {memory_used}KB")
        self.detections_label.setText(f"🔍 Detections: {detections}")
        self.temp_label.setText(f"🌡️ Temp: {temperature:.1f}°C")
        
        # Update memory bar with color coding
        memory_percent = min(100, (memory_used / 512) * 100)
        self.memory_bar.setValue(int(memory_percent))
        
        # Color code based on usage
        if memory_percent > 80:
            color = "#f44336"  # Red
        elif memory_percent > 60:
            color = "#ff9800"  # Orange
        else:
            color = "#4CAF50"  # Green
            
        self.memory_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)

class LogPanel(QGroupBox):
    """Enhanced log panel with filtering"""
    
    def __init__(self):
        super().__init__("📝 Event Log")
        self.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #0078d4;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 10px;
            }
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Log controls
        controls_layout = QHBoxLayout()
        
        self.auto_scroll_cb = QPushButton("📜 Auto-scroll: ON")
        self.auto_scroll_cb.setCheckable(True)
        self.auto_scroll_cb.setChecked(True)
        self.auto_scroll_cb.clicked.connect(self.toggle_auto_scroll)
        
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.clear_log)
        
        controls_layout.addWidget(self.auto_scroll_cb)
        controls_layout.addStretch()
        controls_layout.addWidget(clear_btn)
        
        # Enhanced log display
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        layout.addLayout(controls_layout)
        layout.addWidget(self.log_text)
        self.setLayout(layout)
        
        # Start with welcome message
        self.add_log("ESP32-S3-EYE Container Detection System Ready", "INFO")
        
    def toggle_auto_scroll(self):
        """Toggle auto-scroll functionality"""
        if self.auto_scroll_cb.isChecked():
            self.auto_scroll_cb.setText("📜 Auto-scroll: ON")
        else:
            self.auto_scroll_cb.setText("📜 Auto-scroll: OFF")
        
    def clear_log(self):
        """Clear log with confirmation"""
        self.log_text.clear()
        self.add_log("Log cleared", "INFO")
        
    def add_log(self, message: str, level: str = "INFO"):
        """Enhanced log formatting with timestamps and colors"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            "INFO": "#00ff00",
            "WARNING": "#ffff00", 
            "ERROR": "#ff0000",
            "SUCCESS": "#00ff7f",
            "DEBUG": "#87ceeb"
        }
        
        icon_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅",
            "DEBUG": "🔧"
        }
        
        color = color_map.get(level, "#ffffff")
        icon = icon_map.get(level, "•")
        
        formatted_msg = f'<span style="color: {color}; font-weight: bold;">[{timestamp}] {icon} {level}:</span> <span style="color: #ffffff;">{message}</span><br>'
        
        self.log_text.insertHtml(formatted_msg)
        
        if self.auto_scroll_cb.isChecked():
            self.log_text.ensureCursorVisible()

class MainWindow(QMainWindow):
    """Enhanced main window with better layout and functionality"""
    
    def __init__(self):
        super().__init__()
        self.serial_worker: Optional[SerialWorker] = None
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.clear_detections)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the enhanced user interface"""
        self.setWindowTitle("ESP32-S3-EYE Container Anomaly Detection System v2.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply enhanced dark theme
        self.setStyleSheet("""
            QMainWindow { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2b2b2b, stop:1 #1e1e1e);
                color: #ffffff; 
            }
            QGroupBox { 
                font-weight: bold; 
                border: 2px solid #0078d4; 
                margin: 5px; 
                padding-top: 15px;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.05);
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 15px; 
                padding: 0 8px 0 8px;
                color: #0078d4;
            }
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #106ebe);
                color: white; 
                border: none; 
                padding: 10px; 
                margin: 3px; 
                border-radius: 6px; 
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #106ebe, stop:1 #005a9e);
            }
            QPushButton:pressed { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #005a9e, stop:1 #004578);
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 2px solid #555;
                border-radius: 4px;
                padding: 5px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #aaa;
            }
        """)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side: Camera feed (larger)
        self.camera_widget = CameraWidget()
        
        # Right side: Control panels
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        
        self.control_panel = ControlPanel()
        self.control_panel.command_requested.connect(self.handle_command)
        
        self.status_panel = StatusPanel()
        self.log_panel = LogPanel()
        
        right_panel.addWidget(self.control_panel)
        right_panel.addWidget(self.status_panel)
        right_panel.addWidget(self.log_panel)
        right_panel.addStretch()
        
        # Create right widget container
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setMaximumWidth(400)
        right_widget.setMinimumWidth(350)
        
        # Add to main layout with proportions
        main_layout.addWidget(self.camera_widget, 3)  # 3/4 of width
        main_layout.addWidget(right_widget, 1)        # 1/4 of width
        
        # Enhanced status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 5px;
            }
        """)
        self.status_bar.showMessage("🚀 Ready - Connect to ESP32-S3-EYE to begin")
        
        # Initialize with welcome message
        self.log_panel.add_log("ESP32-S3-EYE Container Detection System v2.0 Started", "SUCCESS")
        self.log_panel.add_log("Maritime Container Damage Detection Ready", "INFO")
        
    def handle_command(self, command: str):
        """Handle enhanced commands from control panel"""
        if command.startswith("CONNECT:"):
            port = command.split(":", 1)[1]
            self.connect_to_esp32(port)
            self.log_panel.add_log(f"Attempting connection to {port}", "INFO")
        elif command.startswith("TRAIN:"):
            class_id = command.split(":", 1)[1]
            if self.serial_worker:
                self.serial_worker.send_command(f"TRAIN:{class_id}")
                damage_type = list(DAMAGE_DESCRIPTIONS.keys())[int(class_id)]
                self.log_panel.add_log(f"Training initiated for: {damage_type}", "WARNING")
        elif command.startswith("BTN:"):
            button_id = command.split(":", 1)[1]
            if self.serial_worker:
                self.serial_worker.send_command(f"BTN:{button_id}")
                button_actions = {
                    "1": "Capture/Freeze Frame",
                    "2": "Toggle Detection", 
                    "3": "Enter Training Menu",
                    "4": "Train as Healthy"
                }
                action = button_actions.get(button_id, f"Button {button_id}")
                self.log_panel.add_log(f"Command sent: {action}", "DEBUG")
            
    def connect_to_esp32(self, port: str):
        """Enhanced ESP32 connection with better error handling"""
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker.wait()
            
        self.serial_worker = SerialWorker(port)
        
        # Connect all signals
        self.serial_worker.frame_received.connect(self.camera_widget.update_frame)
        self.serial_worker.detection_received.connect(self.handle_detection)
        self.serial_worker.status_received.connect(self.handle_status_update)
        self.serial_worker.connection_status.connect(self.handle_connection_status)
        self.serial_worker.training_complete.connect(self.handle_training_complete)
        
        self.serial_worker.start()
        
    def handle_detection(self, detection: Dict):
        """Handle incoming detection with enhanced logging"""
        self.camera_widget.add_detection(detection)
        
        # Clear detections after 3 seconds
        self.detection_timer.start(3000)
        
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Enhanced logging based on damage type
        if class_name == 'no_damage':
            self.log_panel.add_log(f"✅ Container healthy (confidence: {confidence:.1%})", "SUCCESS")
        else:
            description = DAMAGE_DESCRIPTIONS.get(class_name, class_name)
            self.log_panel.add_log(f"⚠️ Damage detected: {description} (confidence: {confidence:.1%})", "WARNING")
        
    def handle_status_update(self, status: Dict):
        """Handle status updates with FPS forwarding"""
        self.status_panel.update_metrics(status)
        
        # Update camera widget FPS
        fps = status.get('fps', 0)
        self.camera_widget.set_fps(fps)
        
        # Update status bar with key metrics
        memory = status.get('memory_used', 0)
        detections = status.get('detections', 0)
        self.status_bar.showMessage(
            f"🔍 FPS: {fps:.1f} | 💾 Memory: {memory}KB | 📊 Detections: {detections}")
        
    def handle_connection_status(self, connected: bool, message: str):
        """Handle connection status with UI updates"""
        self.status_panel.update_connection_status(connected, message)
        self.control_panel.connection_established(connected)
        
        if connected:
            self.status_bar.showMessage(f"✅ Connected: {message}")
            self.log_panel.add_log(f"Connection established: {message}", "SUCCESS")
        else:
            self.status_bar.showMessage(f"❌ Connection failed: {message}")
            self.log_panel.add_log(f"Connection failed: {message}", "ERROR")
            
    def handle_training_complete(self, result: str, success: bool):
        """Handle training completion notifications"""
        if success:
            self.log_panel.add_log(f"✅ Training completed successfully: {result}", "SUCCESS")
        else:
            self.log_panel.add_log(f"❌ Training failed: {result}", "ERROR")
        
    def clear_detections(self):
        """Clear detection overlays"""
        self.camera_widget.detections.clear()
        self.camera_widget.update_display()
        self.detection_timer.stop()
        
    def closeEvent(self, event):
        """Enhanced application shutdown"""
        self.log_panel.add_log("Shutting down application...", "INFO")
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker.wait()
        event.accept()

def main():
    """Enhanced main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("ESP32-S3-EYE Container Detection")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Maritime AI Solutions")
    
    # Set application style and palette
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    print("ESP32-S3-EYE Container Anomaly Detection GUI v2.0")
    print("Enhanced interface for maritime damage detection")
    print("Ready for ESP32 connection...")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
