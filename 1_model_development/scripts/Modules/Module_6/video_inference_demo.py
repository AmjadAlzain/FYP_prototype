"""
Video Inference Demo for Container Anomaly Detection
Processes video files and outputs annotated results with bounding boxes
Supports the demo video for FYP 2 VIVA presentation
"""

import os
import sys
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_inference_enhanced import EnhancedInferenceEngine, DetectionResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoInferenceProcessor:
    """Video processor for container anomaly detection"""
    
    def __init__(self, models_dir: str = "../../models"):
        self.models_dir = Path(models_dir)
        self.inference_engine = None
        
        # Video processing settings
        self.target_fps = 30
        self.detection_mode = "grid"  # "grid" or "contour"
        self.confidence_threshold = 0.6
        
        # Output settings
        self.output_codec = 'mp4v'
        self.output_quality = 95
        
        # Statistics tracking
        self.frame_stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'damaged_frames': 0,
            'avg_processing_time': 0.0,
            'total_detections': 0,
            'damage_summary': {}
        }
        
        logger.info("Video inference processor initialized")
    
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
    
    def process_video(self, input_path: str, output_path: str, 
                     start_time: float = 0.0, end_time: Optional[float] = None,
                     skip_frames: int = 1) -> Dict:
        """
        Process video file and generate annotated output
        
        Args:
            input_path: Path to input video
            output_path: Path for output video  
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (default: None for full video)
            skip_frames: Process every Nth frame (default: 1 for all frames)
        """
        if self.inference_engine is None:
            raise RuntimeError("Models not initialized. Call initialize_models() first.")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        logger.info(f"Input video info:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps:.2f}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Duration: {duration:.2f}s")
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames
        
        # Set video position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Could not create output video: {output_path}")
        
        # Processing loop
        frame_count = 0
        processed_count = 0
        processing_times = []
        all_detections = []
        
        logger.info(f"Processing frames {start_frame} to {end_frame}")
        logger.info(f"Processing every {skip_frames} frame(s)")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame:
                    break
                
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                frame_count += 1
                
                # Skip frames if specified
                if (frame_count - 1) % skip_frames != 0:
                    out.write(frame)
                    continue
                
                processed_count += 1
                
                # Convert BGR to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                start_proc_time = time.time()
                result = self.inference_engine.process_full_image(frame_rgb, self.detection_mode)
                proc_time = time.time() - start_proc_time
                processing_times.append(proc_time)
                
                # Handle processing errors
                if 'error' in result:
                    logger.warning(f"Frame {current_frame}: {result['error']}")
                    out.write(frame)
                    continue
                
                # Draw detections on frame
                detections = result['detections']
                if detections:
                    frame_annotated = self.inference_engine.draw_detections(
                        frame_rgb, detections, show_confidence=True
                    )
                    # Convert back to BGR for video output
                    frame_annotated = cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR)
                else:
                    frame_annotated = frame
                
                # Add frame information overlay
                frame_annotated = self._add_frame_overlay(
                    frame_annotated, current_frame, len(detections), 
                    result.get('damage_detected', False), proc_time
                )
                
                # Write annotated frame
                out.write(frame_annotated)
                
                # Store detection info
                frame_info = {
                    'frame_number': current_frame,
                    'timestamp': current_frame / fps,
                    'num_detections': len(detections),
                    'damage_detected': result.get('damage_detected', False),
                    'processing_time': proc_time,
                    'class_counts': result.get('class_counts', {}),
                    'detections': [
                        {
                            'x': d.x, 'y': d.y, 'w': d.w, 'h': d.h,
                            'damage_type': d.damage_type,
                            'confidence': d.confidence,
                            'is_damaged': d.is_damaged
                        } for d in detections
                    ]
                }
                all_detections.append(frame_info)
                
                # Update statistics
                self.frame_stats['total_detections'] += len(detections)
                if result.get('damage_detected', False):
                    self.frame_stats['damaged_frames'] += 1
                
                # Progress update
                if processed_count % 30 == 0:
                    progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                    avg_time = np.mean(processing_times[-30:])
                    logger.info(f"Progress: {progress:.1f}% | Avg time: {avg_time:.3f}s/frame")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
        finally:
            # Cleanup
            cap.release()
            out.release()
        
        # Calculate final statistics
        self.frame_stats['total_frames'] = frame_count
        self.frame_stats['processed_frames'] = processed_count
        self.frame_stats['avg_processing_time'] = np.mean(processing_times) if processing_times else 0.0
        
        # Summarize damage types
        damage_summary = {}
        for frame_info in all_detections:
            for class_name, count in frame_info['class_counts'].items():
                if class_name != 'no_damage' and count > 0:
                    damage_summary[class_name] = damage_summary.get(class_name, 0) + count
        
        self.frame_stats['damage_summary'] = damage_summary
        
        # Create results summary
        results = {
            'input_file': input_path,
            'output_file': output_path,
            'processing_stats': self.frame_stats.copy(),
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'duration': duration
            },
            'frame_detections': all_detections,
            'inference_engine_stats': self.inference_engine.get_stats()
        }
        
        logger.info(f"✅ Video processing completed!")
        logger.info(f"  Processed {processed_count}/{frame_count} frames")
        logger.info(f"  Average processing time: {self.frame_stats['avg_processing_time']:.3f}s/frame")
        logger.info(f"  Frames with damage: {self.frame_stats['damaged_frames']}")
        logger.info(f"  Total detections: {self.frame_stats['total_detections']}")
        logger.info(f"  Damage summary: {damage_summary}")
        
        return results
    
    def _add_frame_overlay(self, frame: np.ndarray, frame_num: int, 
                          num_detections: int, damage_detected: bool, 
                          processing_time: float) -> np.ndarray:
        """Add information overlay to frame"""
        overlay = frame.copy()
        
        # Overlay background
        overlay_height = 80
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        
        # Frame info
        info_text = [
            f"Frame: {frame_num}",
            f"Detections: {num_detections}",
            f"Status: {'DAMAGE DETECTED' if damage_detected else 'NO DAMAGE'}",
            f"Processing: {processing_time*1000:.1f}ms"
        ]
        
        y_pos = 20
        for i, text in enumerate(info_text):
            color = (0, 0, 255) if damage_detected and i == 2 else (255, 255, 255)
            cv2.putText(overlay, text, (10 + i * 200, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add timestamp
        timestamp = f"Time: {frame_num/30:.1f}s"
        cv2.putText(overlay, timestamp, (10, y_pos + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Blend overlay
        alpha = 0.8
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def save_results_json(self, results: Dict, json_path: str):
        """Save processing results to JSON file"""
        try:
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {json_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def create_summary_video(self, results: Dict, summary_path: str):
        """Create a summary video with key statistics"""
        # This could be extended to create a summary dashboard video
        logger.info("Summary video creation not implemented yet")

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Container Anomaly Detection Video Processing')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Output video path (default: auto-generated)')
    parser.add_argument('--models', default='../../models', help='Models directory path')
    parser.add_argument('--start', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--end', type=float, help='End time in seconds')
    parser.add_argument('--skip', type=int, default=1, help='Process every N frames')
    parser.add_argument('--mode', choices=['grid', 'contour'], default='grid', 
                       help='Detection mode')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON')
    
    args = parser.parse_args()
    
    # Setup paths
    input_path = Path(args.input_video)
    if not input_path.exists():
        logger.error(f"Input video not found: {input_path}")
        return
    
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.parent / f"{input_path.stem}_annotated_{timestamp}.mp4"
    
    # Initialize processor
    processor = VideoInferenceProcessor(args.models)
    processor.detection_mode = args.mode
    
    if not processor.initialize_models():
        logger.error("Failed to initialize models")
        return
    
    # Process video
    try:
        results = processor.process_video(
            str(input_path), str(output_path),
            start_time=args.start,
            end_time=args.end,
            skip_frames=args.skip
        )
        
        logger.info(f"Output video saved to: {output_path}")
        
        # Save JSON results if requested
        if args.save_json:
            json_path = output_path.with_suffix('.json')
            processor.save_results_json(results, str(json_path))
        
        # Print summary
        stats = results['processing_stats']
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Input: {input_path.name}")
        print(f"Output: {output_path.name}")
        print(f"Frames processed: {stats['processed_frames']}/{stats['total_frames']}")
        print(f"Frames with damage: {stats['damaged_frames']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average processing time: {stats['avg_processing_time']:.3f}s/frame")
        print(f"Damage types found: {stats['damage_summary']}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")

# Demo function for testing
def demo_processing():
    """Demo function to process the test video"""
    # Use absolute path to the testing video
    video_path = "f:/FYP/Prototype/testing_video/demo_short.mp4"
    models_path = "../../models"
    
    logger.info("🎬 Starting demo video processing...")
    
    # Check if video exists
    if not os.path.exists(video_path):
        logger.error(f"Demo video not found: {video_path}")
        logger.info("Please ensure the demo video is in the 'testing video' directory")
        return
    
    # Initialize processor
    processor = VideoInferenceProcessor(models_path)
    
    if not processor.initialize_models():
        logger.error("Failed to initialize models for demo")
        return
    
    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"demo_annotated_{timestamp}.mp4"
    
    # Process with demo settings
    try:
        results = processor.process_video(
            video_path, output_path,
            skip_frames=2  # Process every 2nd frame for speed
        )
        
        # Save results
        json_path = f"demo_results_{timestamp}.json"
        processor.save_results_json(results, json_path)
        
        print("\n🎉 Demo processing completed!")
        print(f"📹 Annotated video: {output_path}")
        print(f"📊 Results JSON: {json_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo processing failed: {e}")
        return None

if __name__ == "__main__":
    demo_processing()
