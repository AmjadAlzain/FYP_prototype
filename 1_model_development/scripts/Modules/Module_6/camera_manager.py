"""
Camera Manager for Enhanced Container Detection GUI
Handles ESP32, laptop camera, and video file sources
"""

import cv2
import numpy as np
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable, Dict, Any
from pathlib import Path
import logging

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logger = logging.getLogger(__name__)

class CameraSource(ABC):
    """Abstract base class for camera sources"""
    
    def __init__(self):
        self.is_active = False
        self.frame_callback = None
        self.error_callback = None
        
    @abstractmethod
    def start(self) -> bool:
        """Start the camera source"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the camera source"""
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame"""
        pass
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for new frames"""
        self.frame_callback = callback
        
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for errors"""
        self.error_callback = callback

class ESP32Camera(CameraSource):
    """ESP32-S3-EYE camera source via serial communication"""
    
    def __init__(self, port: str, baudrate: int = 115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.read_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
    def start(self) -> bool:
        """Start ESP32 camera connection"""
        try:
            if not SERIAL_AVAILABLE:
                if self.error_callback:
                    self.error_callback("PySerial not available. Install with: pip install pyserial")
                return False
                
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow ESP32 to initialize
            
            self.is_active = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            logger.info(f"ESP32 camera connected on {self.port}")
            return True
            
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"ESP32 connection failed: {str(e)}")
            return False
    
    def stop(self):
        """Stop ESP32 camera connection"""
        self.is_active = False
        
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2)
        
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            
        logger.info("ESP32 camera disconnected")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from ESP32"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def send_command(self, command: str):
        """Send command to ESP32"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(f"{command}\n".encode())
                self.serial_conn.flush()
            except Exception as e:
                logger.error(f"Error sending command: {e}")
    
    def _read_loop(self):
        """Read data from ESP32 in background thread"""
        while self.is_active:
            try:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        self._parse_message(line)
                time.sleep(0.01)  # 100 Hz polling
                
            except Exception as e:
                if self.is_active:  # Only log if we're supposed to be active
                    logger.error(f"ESP32 read error: {e}")
                break
    
    def _parse_message(self, message: str):
        """Parse incoming message from ESP32"""
        try:
            if message.startswith("IMG:"):
                # Parse image data: IMG:width,height,base64_data
                import base64
                parts = message[4:].split(',', 2)
                if len(parts) == 3:
                    width, height = int(parts[0]), int(parts[1])
                    img_data = base64.b64decode(parts[2])
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    
                    if len(img_array) == width * height * 3:
                        frame = img_array.reshape((height, width, 3))
                        
                        with self.frame_lock:
                            self.latest_frame = frame
                        
                        if self.frame_callback:
                            self.frame_callback(frame)
                            
        except Exception as e:
            logger.error(f"Error parsing ESP32 message: {e}")

class LaptopCamera(CameraSource):
    """Laptop camera source using OpenCV"""
    
    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.cap = None
        self.capture_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.fps_target = 15.0  # Target 15 FPS
        
    def start(self) -> bool:
        """Start laptop camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                if self.error_callback:
                    self.error_callback(f"Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)
            
            self.is_active = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info(f"Laptop camera {self.camera_id} started")
            return True
            
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Laptop camera error: {str(e)}")
            return False
    
    def stop(self):
        """Stop laptop camera capture"""
        self.is_active = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            
        logger.info("Laptop camera stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from laptop camera"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def _capture_loop(self):
        """Capture frames in background thread"""
        frame_interval = 1.0 / self.fps_target
        
        while self.is_active:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB for consistency
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with self.frame_lock:
                    self.latest_frame = frame_rgb
                
                if self.frame_callback:
                    self.frame_callback(frame_rgb)
            else:
                if self.error_callback:
                    self.error_callback("Failed to read from camera")
                break
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

class VideoFileCamera(CameraSource):
    """Video file source for batch processing"""
    
    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = Path(video_path)
        self.cap = None
        self.playback_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Playback control
        self.is_playing = False
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 15.0  # Default playback FPS
        self.playback_speed = 1.0  # 1.0 = normal speed
        
    def start(self) -> bool:
        """Start video file playback"""
        try:
            if not self.video_path.exists():
                if self.error_callback:
                    self.error_callback(f"Video file not found: {self.video_path}")
                return False
            
            self.cap = cv2.VideoCapture(str(self.video_path))
            
            if not self.cap.isOpened():
                if self.error_callback:
                    self.error_callback(f"Cannot open video file: {self.video_path}")
                return False
            
            # Get video properties
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
            
            self.is_active = True
            logger.info(f"Video loaded: {self.total_frames} frames at {self.fps:.1f} FPS")
            return True
            
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Video file error: {str(e)}")
            return False
    
    def stop(self):
        """Stop video playback"""
        self.is_active = False
        self.is_playing = False
        
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            
        logger.info("Video playback stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame from video"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def play(self):
        """Start/resume video playback"""
        if not self.is_active:
            return
            
        self.is_playing = True
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()
    
    def pause(self):
        """Pause video playback"""
        self.is_playing = False
    
    def seek_frame(self, frame_number: int):
        """Seek to specific frame"""
        if not self.cap:
            return
            
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame_number = frame_number
        
        # Read and update current frame
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.frame_lock:
                self.latest_frame = frame_rgb
            
            if self.frame_callback:
                self.frame_callback(frame_rgb)
    
    def get_progress(self) -> Tuple[int, int]:
        """Get current playback progress"""
        return self.current_frame_number, self.total_frames
    
    def set_playback_speed(self, speed: float):
        """Set playback speed multiplier"""
        self.playback_speed = max(0.1, min(speed, 10.0))
    
    def _playback_loop(self):
        """Video playback loop"""
        frame_interval = 1.0 / (self.fps * self.playback_speed)
        
        while self.is_active and self.is_playing:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                with self.frame_lock:
                    self.latest_frame = frame_rgb
                
                if self.frame_callback:
                    self.frame_callback(frame_rgb)
            else:
                # End of video
                self.is_playing = False
                break
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

class CameraManager:
    """Main camera manager handling multiple sources"""
    
    def __init__(self):
        self.current_source = None
        self.available_sources = {}
        self.frame_callback = None
        self.error_callback = None
        
        # Initialize available sources
        self._scan_available_sources()
    
    def _scan_available_sources(self):
        """Scan for available camera sources"""
        self.available_sources = {}
        
        # ESP32 ports
        if SERIAL_AVAILABLE:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                # Check for common ESP32 identifiers
                if any(identifier in port.description.lower() 
                       for identifier in ['esp32', 'silicon labs', 'ch340', 'ftdi']):
                    self.available_sources[f"ESP32 ({port.device})"] = {
                        'type': 'esp32',
                        'device': port.device,
                        'description': port.description
                    }
        
        # Laptop cameras
        for i in range(4):  # Check first 4 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.available_sources[f"Laptop Camera {i}"] = {
                    'type': 'laptop',
                    'device': i,
                    'description': f"Camera device {i}"
                }
                cap.release()
    
    def get_available_sources(self) -> Dict[str, Dict]:
        """Get list of available camera sources"""
        self._scan_available_sources()  # Refresh
        return self.available_sources.copy()
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback for new frames"""
        self.frame_callback = callback
        if self.current_source:
            self.current_source.set_frame_callback(callback)
    
    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for errors"""
        self.error_callback = callback
        if self.current_source:
            self.current_source.set_error_callback(callback)
    
    def switch_to_source(self, source_name: str) -> bool:
        """Switch to specified camera source"""
        # Stop current source
        if self.current_source:
            self.current_source.stop()
            self.current_source = None
        
        # Check if source exists
        if source_name not in self.available_sources:
            if self.error_callback:
                self.error_callback(f"Source not found: {source_name}")
            return False
        
        source_info = self.available_sources[source_name]
        
        try:
            # Create new source
            if source_info['type'] == 'esp32':
                self.current_source = ESP32Camera(source_info['device'])
            elif source_info['type'] == 'laptop':
                self.current_source = LaptopCamera(source_info['device'])
            else:
                if self.error_callback:
                    self.error_callback(f"Unsupported source type: {source_info['type']}")
                return False
            
            # Set callbacks
            if self.frame_callback:
                self.current_source.set_frame_callback(self.frame_callback)
            if self.error_callback:
                self.current_source.set_error_callback(self.error_callback)
            
            # Start source
            if self.current_source.start():
                logger.info(f"Switched to camera source: {source_name}")
                return True
            else:
                self.current_source = None
                return False
                
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Error switching to {source_name}: {str(e)}")
            return False
    
    def load_video_file(self, video_path: str) -> bool:
        """Load video file as source"""
        # Stop current source
        if self.current_source:
            self.current_source.stop()
        
        try:
            self.current_source = VideoFileCamera(video_path)
            
            # Set callbacks
            if self.frame_callback:
                self.current_source.set_frame_callback(self.frame_callback)
            if self.error_callback:
                self.current_source.set_error_callback(self.error_callback)
            
            # Start source
            if self.current_source.start():
                logger.info(f"Loaded video file: {video_path}")
                return True
            else:
                self.current_source = None
                return False
                
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Error loading video: {str(e)}")
            return False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame from active source"""
        if self.current_source:
            return self.current_source.get_frame()
        return None
    
    def send_esp32_command(self, command: str):
        """Send command to ESP32 (if current source is ESP32)"""
        if isinstance(self.current_source, ESP32Camera):
            self.current_source.send_command(command)
    
    def is_video_source(self) -> bool:
        """Check if current source is a video file"""
        return isinstance(self.current_source, VideoFileCamera)
    
    def get_video_controls(self) -> Optional[VideoFileCamera]:
        """Get video controls (if current source is video)"""
        if isinstance(self.current_source, VideoFileCamera):
            return self.current_source
        return None
    
    def stop_current_source(self):
        """Stop current camera source"""
        if self.current_source:
            self.current_source.stop()
            self.current_source = None

# Test function
def test_camera_manager():
    """Test camera manager functionality"""
    import time
    
    def frame_callback(frame):
        print(f"Received frame: {frame.shape}")
    
    def error_callback(error):
        print(f"Error: {error}")
    
    manager = CameraManager()
    manager.set_frame_callback(frame_callback)
    manager.set_error_callback(error_callback)
    
    # List available sources
    sources = manager.get_available_sources()
    print("Available sources:")
    for name, info in sources.items():
        print(f"  {name}: {info['description']}")
    
    # Test laptop camera if available
    laptop_sources = [name for name, info in sources.items() if info['type'] == 'laptop']
    if laptop_sources:
        print(f"\nTesting laptop camera: {laptop_sources[0]}")
        if manager.switch_to_source(laptop_sources[0]):
            time.sleep(5)  # Capture for 5 seconds
            manager.stop_current_source()
    
    print("Camera manager test completed")

if __name__ == "__main__":
    test_camera_manager()
