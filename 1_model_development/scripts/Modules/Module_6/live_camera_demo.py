"""
Live Camera Demo for Container Anomaly Detection
Real-time inference using laptop camera or webcam
Perfect for FYP 2 VIVA live demonstration
"""

import os
import sys
import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_inference_enhanced import EnhancedInferenceEngine, DetectionResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveCameraDemo:
    """Live camera demonstration for container anomaly detection"""
    
    def __init__(self, models_dir: str = "../../models", camera_id: int = 0):
        self.models_dir = Path(models_dir)
        self.camera_id = camera_id
        self.inference_engine = None
        
        # Camera settings
        self.camera_width = 1280
        self.camera_height = 720
        self.target_fps = 15  # Lower FPS for real-time processing
        
        # Processing settings
        self.detection_mode = "grid"
        self.confidence_threshold = 0.5  # Lower threshold for demo
        self.process_every_n_frames = 3  # Process every 3rd frame for speed
        
        # Display settings
        self.display_scale = 0.8
        self.show_fps = True
        self.show_stats = True
        
        # State variables
        self.is_running = False
        self.current_frame = None
        self.current_detections = []
        self.frame_count = 0
        self.last_inference_time = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'damage_detections': 0,
            'avg_processing_time': 0.0,
            'current_fps': 0.0
        }
        
        logger.info(f"Live camera demo initialized for camera {camera_id}")
    
    def initialize_models(self) -> bool:
        """Initialize the inference engine and load models"""
        try:
            self.inference_engine = EnhancedInferenceEngine(str(self.models_dir))
            
            if self.inference_engine.load_models():
                logger.info("✅ Models loaded successfully")
                return True
            else:
                logger.error("❌ Failed to load models")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
    
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual camera properties
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized:")
            logger.info(f"  Resolution: {actual_width}x{actual_height}")
            logger.info(f"  FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def process_frame_async(self, frame: np.ndarray):
        """Process frame asynchronously to maintain real-time performance"""
        try:
            start_time = time.time()
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with inference engine
            result = self.inference_engine.process_full_image(frame_rgb, self.detection_mode)
            
            processing_time = time.time() - start_time
            
            # Update detections and stats
            if 'error' not in result:
                self.current_detections = result['detections']
                self.stats['processed_frames'] += 1
                self.stats['total_detections'] += len(self.current_detections)
                
                # Count damage detections
                damage_count = sum(1 for d in self.current_detections if d.is_damaged)
                self.stats['damage_detections'] += damage_count
                
                # Update average processing time
                if self.stats['processed_frames'] == 1:
                    self.stats['avg_processing_time'] = processing_time
                else:
                    alpha = 0.1
                    self.stats['avg_processing_time'] = (
                        alpha * processing_time + 
                        (1 - alpha) * self.stats['avg_processing_time']
                    )
            
            self.last_inference_time = processing_time
            
        except Exception as e:
            logger.error(f"Error in frame processing: {e}")
            self.current_detections = []
    
    def draw_enhanced_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw enhanced overlay with detections and information"""
        overlay_frame = frame.copy()
        
        # Draw detections if available
        if self.current_detections and self.inference_engine:
            # Convert to RGB for drawing, then back to BGR
            frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
            frame_annotated = self.inference_engine.draw_detections(
                frame_rgb, self.current_detections, show_confidence=True
            )
            overlay_frame = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)
        
        # Add information overlay
        overlay_frame = self._draw_info_overlay(overlay_frame)
        
        # Add demo instructions
        overlay_frame = self._draw_instructions(overlay_frame)
        
        return overlay_frame
    
    def _draw_info_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw information overlay"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Info panel background
        panel_height = 120
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        
        # Title
        cv2.putText(overlay, "ESP32 Container Anomaly Detection - Live Demo", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Current status
        num_detections = len(self.current_detections)
        damage_detected = any(d.is_damaged for d in self.current_detections)
        
        status_text = f"Detections: {num_detections}"
        status_color = (0, 255, 0)  # Green
        
        if damage_detected:
            status_text += " | DAMAGE DETECTED!"
            status_color = (0, 0, 255)  # Red
        else:
            status_text += " | System OK"
        
        cv2.putText(overlay, status_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Performance info
        if self.show_fps:
            perf_text = f"FPS: {self.stats['current_fps']:.1f} | "
            perf_text += f"Processing: {self.last_inference_time*1000:.1f}ms | "
            perf_text += f"Avg: {self.stats['avg_processing_time']*1000:.1f}ms"
            
            cv2.putText(overlay, perf_text, (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistics
        if self.show_stats:
            stats_text = f"Total Frames: {self.stats['total_frames']} | "
            stats_text += f"Processed: {self.stats['processed_frames']} | "
            stats_text += f"Damage Found: {self.stats['damage_detections']}"
            
            cv2.putText(overlay, stats_text, (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Blend overlay
        alpha = 0.8
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw demo instructions"""
        h, w = frame.shape[:2]
        
        instructions = [
            "DEMO CONTROLS:",
            "SPACE - Toggle detection mode",
            "S - Save screenshot", 
            "R - Reset statistics",
            "Q - Quit demo"
        ]
        
        # Draw instructions in bottom right
        start_y = h - len(instructions) * 25 - 10
        
        for i, instruction in enumerate(instructions):
            y_pos = start_y + i * 25
            
            # Text background
            (text_w, text_h), baseline = cv2.getTextSize(
                instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(frame, (w - text_w - 20, y_pos - text_h - 5),
                         (w - 5, y_pos + baseline + 5), (0, 0, 0), -1)
            
            # Text
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, instruction, (w - text_w - 15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.stats['current_fps'] = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_screenshot_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        logger.info(f"Screenshot saved: {filename}")
    
    def reset_statistics(self):
        """Reset demo statistics"""
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_detections': 0,
            'damage_detections': 0,
            'avg_processing_time': 0.0,
            'current_fps': 0.0
        }
        
        if self.inference_engine:
            self.inference_engine.reset_stats()
        
        logger.info("Statistics reset")
    
    def run_demo(self):
        """Run the live camera demo"""
        if not self.initialize_models():
            logger.error("Failed to initialize models")
            return False
        
        if not self.initialize_camera():
            logger.error("Failed to initialize camera")
            return False
        
        logger.info("🎬 Starting live camera demo...")
        logger.info("Press 'Q' to quit, 'SPACE' to toggle mode, 'S' to save screenshot")
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                self.frame_count += 1
                self.stats['total_frames'] += 1
                
                # Process frame for inference (every Nth frame)
                if self.frame_count % self.process_every_n_frames == 0:
                    # Run inference in separate thread for better performance
                    threading.Thread(
                        target=self.process_frame_async,
                        args=(frame.copy(),),
                        daemon=True
                    ).start()
                
                # Draw overlay and display
                display_frame = self.draw_enhanced_overlay(frame)
                
                # Scale for display if needed
                if self.display_scale != 1.0:
                    new_width = int(display_frame.shape[1] * self.display_scale)
                    new_height = int(display_frame.shape[0] * self.display_scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                # Show frame
                cv2.imshow("ESP32 Container Anomaly Detection - Live Demo", display_frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    logger.info("Demo stopped by user")
                    break
                elif key == ord(' '):  # Spacebar - toggle detection mode
                    self.detection_mode = "contour" if self.detection_mode == "grid" else "grid"
                    logger.info(f"Detection mode switched to: {self.detection_mode}")
                elif key == ord('s') or key == ord('S'):
                    self.save_screenshot(display_frame)
                elif key == ord('r') or key == ord('R'):
                    self.reset_statistics()
                elif key == 27:  # ESC key
                    break
        
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
        except Exception as e:
            logger.error(f"Error in demo loop: {e}")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if hasattr(self, 'cap'):
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        logger.info("Demo completed!")
        logger.info(f"Final Statistics:")
        logger.info(f"  Total frames: {self.stats['total_frames']}")
        logger.info(f"  Processed frames: {self.stats['processed_frames']}")
        logger.info(f"  Total detections: {self.stats['total_detections']}")
        logger.info(f"  Damage detections: {self.stats['damage_detections']}")
        logger.info(f"  Average processing time: {self.stats['avg_processing_time']*1000:.1f}ms")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Live Camera Container Anomaly Detection Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--models', default='../../models', help='Models directory path')
    parser.add_argument('--scale', type=float, default=0.8, help='Display scale factor')
    parser.add_argument('--fps', type=int, default=15, help='Target FPS')
    parser.add_argument('--mode', choices=['grid', 'contour'], default='grid', 
                       help='Initial detection mode')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = LiveCameraDemo(args.models, args.camera)
    demo.display_scale = args.scale
    demo.target_fps = args.fps
    demo.detection_mode = args.mode
    demo.confidence_threshold = args.confidence
    
    # Run demo
    success = demo.run_demo()
    
    if not success:
        logger.error("Demo failed to start")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
