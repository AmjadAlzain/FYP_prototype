"""
ESP32-S3-EYE Container Anomaly Detection - Simplified GUI
Clean, modern interface for maritime container damage detection
"""

import sys
import json
import base64
import time
from pathlib import Path
from typing import Optional, Dict
import cv2
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QPushButton, QTextEdit, QGroupBox,
    QGridLayout, QComboBox, QFrame, QSplitter
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QPainter, QPen

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print("Warning: Install pyserial with: pip install pyserial")
    SERIAL_AVAILABLE = False

# Modern color scheme
COLORS = {
    'primary': '#0078D4',
    'success': '#107C10', 
    'warning': '#FFB900',
    'danger': '#D13438',
    'background': '#F5F5F5',
    'surface': '#FFFFFF',
    'text': '#323130'
}

# Damage detection colors
DAMAGE_COLORS = {
    'axis': (220, 53, 69),        # Bootstrap red
    'concave': (255, 193, 7),     # Bootstrap yellow  
    'dentado': (0, 123, 255),     # Bootstrap blue
    'perforation': (220, 53, 69), # Bootstrap red
    'no_damage': (40, 167, 69),   # Bootstrap green
    'container': (40, 167, 69)    # Green for healthy
}

class ESP32SerialWorker(QThread):
    """Simplified serial communication worker"""
    
    frame_received = pyqtSignal(np.ndarray)
    detection_received = pyqtSignal(dict)
    status_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)
    
    def __init__(self, port: str):
        super().__init__()
        self.port = port
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False
        
    def run(self):
        try:
            self.serial_conn = serial.Serial(self.port, 115200, timeout=1)
            time.sleep(2)  # ESP32 init time
            self.connection_status.emit(True, f"Connected to {self.port}")
            self.running = True
            
            while self.running:
                try:
                    if self.serial_conn.in_waiting > 0:
                        line = self.serial_conn.readline().decode('utf-8').strip()
                        if line:
                            self.parse_message(line)
                    time.sleep(0.01)
                except (UnicodeDecodeError, Exception):
                    continue
                    
        except Exception as e:
            self.connection_status.emit(False, f"Connection failed: {str(e)}")
            
    def parse_message(self, message: str):
        try:
            if message.startswith("IMG:"):
                # Simple image format: IMG:width,height,base64_data
                parts = message[4:].split(',', 2)
                if len(parts) == 3:
                    width, height = int(parts[0]), int(parts[1])
                    img_data = base64.b64decode(parts[2])
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img_rgb = img_array.reshape((height, width, 3))
                    self.frame_received.emit(img_rgb)
                    
            elif message.startswith("DETECT:"):
                # DETECT:x,y,w,h,class,confidence
                parts = message[7:].split(',')
                if len(parts) >= 6:
                    detection = {
                        'x': int(parts[0]),
                        'y': int(parts[1]), 
                        'w': int(parts[2]),
                        'h': int(parts[3]),
                        'class': parts[4],
                        'confidence': float(parts[5])
                    }
                    self.detection_received.emit(detection)
                    
            elif message.startswith("STATUS:"):
                # STATUS:fps,memory,detections
                parts = message[7:].split(',')
                if len(parts) >= 3:
                    status = {
                        'fps': float(parts[0]),
                        'memory': int(parts[1]),
                        'detections': int(parts[2])
                    }
                    self.status_received.emit(status)
                    
        except Exception as e:
            print(f"Parse error: {e}")
            
    def send_command(self, command: str):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{command}\n".encode())
                self.serial_conn.flush()
            except Exception as e:
                print(f"Send error: {e}")
                
    def stop(self):
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

class CameraDisplay(QLabel):
    """Clean camera display widget"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet(f"""
            QLabel {{
                border: 3px solid {COLORS['primary']};
                background-color: #000000;
                border-radius: 12px;
                color: white;
                font-size: 16px;
            }}
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("📷 Waiting for camera...")
        self.setFont(QFont("Segoe UI", 14))
        
        self.current_frame = None
        self.detection = None
        self.fps = 0
        
    def update_frame(self, frame: np.ndarray):
        self.current_frame = frame.copy()
        self.update_display()
        
    def update_detection(self, detection: Dict):
        self.detection = detection
        self.update_display()
        
    def set_fps(self, fps: float):
        self.fps = fps
        self.update_display()
        
    def update_display(self):
        if self.current_frame is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # Draw detection if available
        if self.detection:
            x, y, w, h = self.detection['x'], self.detection['y'], self.detection['w'], self.detection['h']
            class_name = self.detection['class']
            confidence = self.detection['confidence']
            
            color = DAMAGE_COLORS.get(class_name, (255, 255, 255))
            
            # Modern bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 3)
            
            # Clean label
            label = f"{class_name.upper()}: {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Label background
            cv2.rectangle(display_frame, (x, y-40), (x + tw + 20, y), color, -1)
            cv2.putText(display_frame, label, (x + 10, y - 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # FPS display
        if self.fps > 0:
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Convert to QPixmap
        height, width, channels = display_frame.shape
        bytes_per_line = channels * width
        q_image = QImage(display_frame.data, width, height, 
                        bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(self.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

class ControlPanel(QGroupBox):
    """Simplified control panel"""
    
    command_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__("🎮 Controls")
        self.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: bold;
                color: {COLORS['text']};
                border: 2px solid {COLORS['primary']};
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: {COLORS['surface']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {COLORS['primary']};
            }}
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Connection
        conn_layout = QHBoxLayout()
        
        self.port_combo = QComboBox()
        self.port_combo.setStyleSheet(f"""
            QComboBox {{
                padding: 10px;
                border: 2px solid {COLORS['primary']};
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
                min-width: 120px;
            }}
        """)
        
        self.connect_btn = QPushButton("🔌 Connect")
        self.connect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #0F5132;
            }}
            QPushButton:pressed {{
                background-color: #0A3622;
            }}
        """)
        self.connect_btn.clicked.connect(self.connect_device)
        
        refresh_btn = QPushButton("🔄")
        refresh_btn.setMaximumWidth(50)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: #106EBE;
            }}
        """)
        refresh_btn.clicked.connect(self.refresh_ports)
        
        conn_layout.addWidget(QLabel("Port:"))
        conn_layout.addWidget(self.port_combo)
        conn_layout.addWidget(refresh_btn)
        conn_layout.addWidget(self.connect_btn)
        
        # Main controls
        controls_layout = QGridLayout()
        
        # Freeze button
        self.freeze_btn = QPushButton("📸 Freeze Frame")
        self.freeze_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #CC9A06;
            }}
        """)
        self.freeze_btn.clicked.connect(lambda: self.command_signal.emit("BTN:1"))
        
        # Detection toggle
        self.detect_btn = QPushButton("🔍 Toggle Detection")
        self.detect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #106EBE;
            }}
        """)
        self.detect_btn.clicked.connect(lambda: self.command_signal.emit("BTN:2"))
        
        # Training buttons
        self.train_healthy_btn = QPushButton("✅ Train: Healthy")
        self.train_healthy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #0F5132;
            }}
        """)
        self.train_healthy_btn.clicked.connect(lambda: self.command_signal.emit("BTN:4"))
        
        self.train_damage_btn = QPushButton("❌ Train: Damage")
        self.train_damage_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['danger']};
                color: white;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #A02622;
            }}
        """)
        self.train_damage_btn.clicked.connect(lambda: self.command_signal.emit("BTN:3"))
        
        controls_layout.addWidget(self.freeze_btn, 0, 0)
        controls_layout.addWidget(self.detect_btn, 0, 1)
        controls_layout.addWidget(self.train_healthy_btn, 1, 0)
        controls_layout.addWidget(self.train_damage_btn, 1, 1)
        
        layout.addLayout(conn_layout)
        layout.addLayout(controls_layout)
        self.setLayout(layout)
        
        # Initialize ports
        self.refresh_ports()
        
    def refresh_ports(self):
        self.port_combo.clear()
        if SERIAL_AVAILABLE:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                self.port_combo.addItem(f"{port.device} - {port.description}")
            if not ports:
                self.port_combo.addItem("No ports found")
        else:
            self.port_combo.addItem("pyserial not installed")
            
    def connect_device(self):
        current_text = self.port_combo.currentText()
        if "not found" in current_text or "not installed" in current_text:
            return
            
        port = current_text.split(' - ')[0]
        self.command_signal.emit(f"CONNECT:{port}")
        self.connect_btn.setText("🔄 Connecting...")
        self.connect_btn.setEnabled(False)
        
    def connection_status(self, connected: bool):
        if connected:
            self.connect_btn.setText("✅ Connected")
            self.connect_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['success']};
                    color: white;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)
        else:
            self.connect_btn.setText("🔌 Connect")
            self.connect_btn.setEnabled(True)
            self.connect_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['danger']};
                    color: white;
                    border: none;
                    padding: 12px 20px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)

class StatusPanel(QGroupBox):
    """Clean status display"""
    
    def __init__(self):
        super().__init__("📊 Status")
        self.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: bold;
                color: {COLORS['text']};
                border: 2px solid {COLORS['primary']};
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: {COLORS['surface']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {COLORS['primary']};
            }}
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QGridLayout()
        
        # Status labels
        self.connection_label = QLabel("❌ Disconnected")
        self.connection_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 14px; font-weight: bold;")
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-size: 14px;")
        
        self.memory_label = QLabel("Memory: --")
        self.memory_label.setStyleSheet("font-size: 14px;")
        
        self.detections_label = QLabel("Detections: --")
        self.detections_label.setStyleSheet("font-size: 14px;")
        
        layout.addWidget(self.connection_label, 0, 0, 1, 2)
        layout.addWidget(self.fps_label, 1, 0)
        layout.addWidget(self.memory_label, 1, 1)
        layout.addWidget(self.detections_label, 2, 0, 1, 2)
        
        self.setLayout(layout)
        
    def update_connection(self, connected: bool, message: str):
        if connected:
            self.connection_label.setText("✅ Connected")
            self.connection_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 14px; font-weight: bold;")
        else:
            self.connection_label.setText(f"❌ {message}")
            self.connection_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 14px; font-weight: bold;")
            
    def update_status(self, status: Dict):
        fps = status.get('fps', 0)
        memory = status.get('memory', 0)
        detections = status.get('detections', 0)
        
        self.fps_label.setText(f"📈 FPS: {fps:.1f}")
        self.memory_label.setText(f"💾 Memory: {memory}KB")
        self.detections_label.setText(f"🔍 Detections: {detections}")

class EventLog(QGroupBox):
    """Simple event logging"""
    
    def __init__(self):
        super().__init__("📝 Events")
        self.setStyleSheet(f"""
            QGroupBox {{
                font-size: 16px;
                font-weight: bold;
                color: {COLORS['text']};
                border: 2px solid {COLORS['primary']};
                border-radius: 12px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: {COLORS['surface']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {COLORS['primary']};
            }}
        """)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setMaximumWidth(80)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['warning']};
                color: white;
                border: none;
                padding: 8px;
                border-radius: 6px;
                font-size: 12px;
            }}
        """)
        clear_btn.clicked.connect(self.clear_log)
        
        # Log display
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
        """)
        
        layout.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.log_text)
        self.setLayout(layout)
        
        self.add_event("System ready", "info")
        
    def add_event(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": "#17a2b8",
            "success": "#28a745", 
            "warning": "#ffc107",
            "error": "#dc3545"
        }
        
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️", 
            "error": "❌"
        }
        
        color = colors.get(level, "#17a2b8")
        icon = icons.get(level, "•")
        
        html = f'<span style="color: {color}; font-weight: bold;">[{timestamp}] {icon}</span> {message}<br>'
        self.log_text.insertHtml(html)
        self.log_text.ensureCursorVisible()
        
    def clear_log(self):
        self.log_text.clear()
        self.add_event("Log cleared", "info")

class ContainerDetectionApp(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.serial_worker = None
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.clear_detection)
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("ESP32-S3-EYE Container Anomaly Detection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Modern styling
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Camera display (left side)
        self.camera_display = CameraDisplay()
        
        # Right panel
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        self.control_panel = ControlPanel()
        self.control_panel.command_signal.connect(self.handle_command)
        
        self.status_panel = StatusPanel()
        self.event_log = EventLog()
        
        right_layout.addWidget(self.control_panel)
        right_layout.addWidget(self.status_panel)
        right_layout.addWidget(self.event_log)
        right_layout.addStretch()
        
        # Right panel container
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setMaximumWidth(400)
        right_widget.setMinimumWidth(350)
        
        # Add to main layout
        main_layout.addWidget(self.camera_display, 3)
        main_layout.addWidget(right_widget, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready - Connect to ESP32-S3-EYE")
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
                padding: 8px;
            }}
        """)
        
        self.event_log.add_event("Container Detection System Started", "success")
        
    def handle_command(self, command: str):
        if command.startswith("CONNECT:"):
            port = command.split(":", 1)[1]
            self.connect_esp32(port)
        elif self.serial_worker:
            self.serial_worker.send_command(command)
            
            # Log actions
            actions = {
                "BTN:1": "Freeze frame",
                "BTN:2": "Toggle detection", 
                "BTN:3": "Enter training",
                "BTN:4": "Train healthy"
            }
            action = actions.get(command, command)
            self.event_log.add_event(f"Action: {action}", "info")
            
    def connect_esp32(self, port: str):
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker.wait()
            
        self.serial_worker = ESP32SerialWorker(port)
        self.serial_worker.frame_received.connect(self.camera_display.update_frame)
        self.serial_worker.detection_received.connect(self.handle_detection)
        self.serial_worker.status_received.connect(self.handle_status)
        self.serial_worker.connection_status.connect(self.handle_connection)
        
        self.serial_worker.start()
        
    def handle_detection(self, detection: Dict):
        self.camera_display.update_detection(detection)
        self.detection_timer.start(3000)  # Clear after 3 seconds
        
        class_name = detection['class']
        confidence = detection['confidence']
        
        if class_name == 'no_damage':
            self.event_log.add_event(f"Container healthy ({confidence:.0%})", "success")
        else:
            self.event_log.add_event(f"Damage: {class_name} ({confidence:.0%})", "warning")
            
    def handle_status(self, status: Dict):
        self.status_panel.update_status(status)
        self.camera_display.set_fps(status.get('fps', 0))
        
    def handle_connection(self, connected: bool, message: str):
        self.status_panel.update_connection(connected, message)
        self.control_panel.connection_status(connected)
        
        if connected:
            self.statusBar().showMessage("Connected - System Ready")
            self.event_log.add_event(f"Connected to {message}", "success")
        else:
            self.statusBar().showMessage("Disconnected")
            self.event_log.add_event(f"Connection failed: {message}", "error")
            
    def clear_detection(self):
        self.camera_display.detection = None
        self.camera_display.update_display()
        self.detection_timer.stop()
        
    def closeEvent(self, event):
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker.wait()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Container Detection")
    app.setStyle('Fusion')
    
    # Set modern font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = ContainerDetectionApp()
    window.show()
    
    print("ESP32-S3-EYE Container Detection GUI Started")
    print("Clean, modern interface for damage detection")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
