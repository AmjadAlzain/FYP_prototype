"""
Streamlit App for Container Anomaly Detection Video Inference
Independent web-based interface for video upload and real-time camera processing
Enhanced with performance analytics and export capabilities
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, List, Optional

# --- Start of Path Correction ---
import sys
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
# --- End of Path Correction ---


# Set page config first
st.set_page_config(
    page_title="Container Anomaly Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
try:
    from local_inference_enhanced import EnhancedInferenceEngine, create_inference_engine, DetectionResult
except ImportError as e:
    st.error(f"Could not import inference modules: {e}")
    st.stop()

# Initialize session state
if 'inference_engine' not in st.session_state:
    st.session_state.inference_engine = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

@st.cache_resource
def load_inference_models():
    """Load inference models with caching"""
    # The models_dir path needs to be absolute for reliability
    # Assuming this script is in Module_6, and models are in 1_model_development/models
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4] # Adjust this number based on your exact project structure
    models_dir = project_root / "1_model_development" / "models"
    
    st.info(f"Loading models from: {models_dir}")
    
    with st.spinner("🔄 Loading Container Detector, TinyNAS & HDC models..."):
        engine = create_inference_engine(str(models_dir))
        
    if engine:
        st.success("✅ Models loaded successfully!")
        return engine
    else:
        st.error("❌ Failed to load models. Please check the console for errors.")
        return None

def process_video_frame(frame: np.ndarray, inference_engine: EnhancedInferenceEngine, confidence_threshold: float) -> Dict:
    """Process a single video frame and apply confidence filtering."""
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set the confidence threshold in the engine before processing
        inference_engine.confidence_threshold = confidence_threshold
        
        # Run inference
        result = inference_engine.process_full_image(frame_rgb)
        result['frame'] = frame_rgb # Use the RGB frame for display consistency
        
        return result
        
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return {
            'error': str(e),
            'detections': [],
            'processing_time': 0,
            'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        }

def draw_detections_on_frame(frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    """Draw detection overlays on frame"""
    result_frame = frame.copy()
    
    for i, detection in enumerate(detections):
        x, y, w, h = detection.x, detection.y, detection.w, detection.h
        damage_type = detection.damage_type
        confidence = detection.confidence
        is_damaged = detection.is_damaged
        
        # Color coding
        if is_damaged:
            color = (255, 50, 50)  # Red for damage
            status = f"⚠️ {damage_type.upper()}"
        else:
            color = (50, 255, 50)  # Green for healthy
            status = "✅ NO DAMAGE"
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 3)
        
        # Draw label
        label = f"{status} ({confidence:.1%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label above box
        text_x = x
        text_y = y - 15 if y - 15 > 15 else y + 30
        
        # Draw label background
        cv2.rectangle(result_frame, 
                     (text_x - 5, text_y - label_h - 10),
                     (text_x + label_w + 5, text_y + baseline),
                     color, -1)
        
        # Draw label text
        cv2.putText(result_frame, label, (text_x, text_y),
                   font, font_scale, (0, 0, 0), thickness) # Black text for better contrast
    
    return result_frame

def create_performance_chart(results: List[Dict]) -> go.Figure:
    """Create performance analytics chart"""
    if not results:
        return go.Figure()
    
    frame_numbers = [r.get('frame_number', i) for i, r in enumerate(results)]
    processing_times = [r.get('processing_time', 0) * 1000 for r in results]
    detection_counts = [len(r.get('detections', [])) for r in results]
    damage_detected = [r.get('damage_detected', False) for r in results]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame_numbers, y=processing_times, mode='lines', name='Processing Time (ms)', yaxis='y'))
    fig.add_trace(go.Scatter(x=frame_numbers, y=detection_counts, mode='lines', name='Detection Count', yaxis='y2'))
    
    damage_frames = [f for f, d in zip(frame_numbers, damage_detected) if d]
    if damage_frames:
        fig.add_trace(go.Scatter(x=damage_frames, y=[max(processing_times) if processing_times else 0] * len(damage_frames), mode='markers', name='Damage Detected', marker=dict(color='red', size=10, symbol='x'), yaxis='y'))
    
    fig.update_layout(title='Real-Time Performance Analytics', xaxis_title='Frame Number', yaxis=dict(title='Processing Time (ms)'), yaxis2=dict(title='Detection Count', overlaying='y', side='right'), hovermode='x unified', height=400)
    return fig

def create_damage_summary_chart(results: List[Dict]) -> go.Figure:
    """Create damage type summary chart"""
    if not results:
        return go.Figure()
    
    damage_counts = {}
    for result in results:
        for detection in result.get('detections', []):
            damage_type = detection.damage_type
            if detection.is_damaged:
                damage_counts[damage_type] = damage_counts.get(damage_type, 0) + 1
    
    if not damage_counts:
        return go.Figure().add_annotation(text="No damage detected in processed frames", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    
    fig = go.Figure(data=[go.Pie(labels=list(damage_counts.keys()), values=list(damage_counts.values()), hole=0.3)])
    fig.update_layout(title='Damage Type Distribution', height=400)
    return fig

def main():
    """Main Streamlit application"""
    st.title("🔍 Container Anomaly Detection System")
    st.markdown("### Real-time video inference with a Deep Learning-based pipeline")
    
    if st.session_state.inference_engine is None:
        st.session_state.inference_engine = load_inference_models()
        
    if st.session_state.inference_engine is None:
        st.stop()
    
    st.sidebar.header("🎛️ Control Panel")
    mode = st.sidebar.selectbox("Select Processing Mode", ["📁 Video Upload", "📷 Live Camera", "📊 Analytics Dashboard"], index=0)
    
    st.sidebar.subheader("⚙️ Processing Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05) # Default from test script
    frame_skip = st.sidebar.selectbox("Frame Skip (process every Nth frame)", [1, 2, 3, 5, 10], index=0)
    show_overlays = st.sidebar.checkbox("Show Detection Overlays", value=True)
    
    st.sidebar.subheader("🚀 System Info")
    try:
        import torch
        if torch.cuda.is_available():
            st.sidebar.success(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.warning("Using CPU (GPU not available)")
    except ImportError:
        st.sidebar.error("PyTorch not available")
    
    if mode == "📁 Video Upload":
        video_upload_mode(confidence_threshold, frame_skip, show_overlays)
    elif mode == "📷 Live Camera":
        live_camera_mode(confidence_threshold, frame_skip, show_overlays)
    elif mode == "📊 Analytics Dashboard":
        analytics_dashboard()

def video_upload_mode(confidence_threshold, frame_skip, show_overlays):
    """Video upload and processing mode"""
    st.header("📁 Video Upload & Processing")
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        st.metric("Total Frames", total_frames)
        
        if st.button("🚀 Start Batch Processing", type="primary"):
            process_video_batch(video_path, total_frames, fps, confidence_threshold, frame_skip, show_overlays)
        
        os.unlink(video_path)

def process_video_batch(video_path, total_frames, fps, confidence_threshold, frame_skip, show_overlays):
    """Process video in batch mode"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    results = []
    cap = cv2.VideoCapture(video_path)
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            result = process_video_frame(frame, st.session_state.inference_engine, confidence_threshold)
            result['frame_number'] = frame_idx
            result['timestamp'] = frame_idx / fps if fps > 0 else 0
            results.append(result)
            
            if show_overlays and 'frame' in result and result.get('detections'):
                annotated_frame = draw_detections_on_frame(result['frame'], result['detections'])
                results_container.image(annotated_frame, caption=f"Processing Frame {frame_idx}", use_column_width=True)
        
        progress_bar.progress((frame_idx + 1) / total_frames)
    
    cap.release()
    st.session_state.processing_results = results
    progress_bar.progress(1.0)
    status_text.success("✅ Processing complete!")
    display_processing_summary(results)

def live_camera_mode(confidence_threshold, frame_skip, show_overlays):
    """Live camera processing mode"""
    st.header("📷 Live Camera Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Camera Controls")
        if st.button("📷 Start Camera", type="primary"):
            st.session_state.camera_active = True
        if st.button("⏹️ Stop Camera"):
            st.session_state.camera_active = False

    with col1:
        if st.session_state.camera_active:
            process_live_camera(confidence_threshold, frame_skip, show_overlays)
        else:
            st.info("Click 'Start Camera' to begin live processing.")

def process_live_camera(confidence_threshold, frame_skip, show_overlays):
    """Process live camera feed"""
    camera_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open camera. Please check permissions.")
        st.session_state.camera_active = False
        return

    frame_count = 0
    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            break
        
        if frame_count % frame_skip == 0:
            result = process_video_frame(frame, st.session_state.inference_engine, confidence_threshold)
            
            if show_overlays:
                annotated_frame = draw_detections_on_frame(result['frame'], result.get('detections', []))
                camera_placeholder.image(annotated_frame, caption=f"Live Feed (Frame {frame_count})", use_column_width=True)
            else:
                camera_placeholder.image(result['frame'], caption=f"Live Feed (Frame {frame_count})", use_column_width=True)

        frame_count += 1
        time.sleep(0.01) # Small delay to allow UI to update

    cap.release()
    st.info("Camera stopped.")

def analytics_dashboard():
    """Analytics dashboard for processing results"""
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.processing_results:
        st.info("No processing results available. Process a video from the 'Video Upload' tab first.")
        return
    
    results = st.session_state.processing_results
    
    total_frames = len(results)
    total_detections = sum(len(r.get('detections', [])) for r in results)
    damage_frames = sum(1 for r in results if r.get('damage_detected', False))
    avg_processing_time = np.mean([r.get('processing_time', 0) for r in results]) * 1000
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processed Frames", total_frames)
    col2.metric("Total Detections", total_detections)
    col3.metric("Avg. Processing Time", f"{avg_processing_time:.1f} ms")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_performance_chart(results), use_container_width=True)
    with col2:
        st.plotly_chart(create_damage_summary_chart(results), use_container_width=True)
    
    st.subheader("📋 Detailed Results")
    df_data = [{'Frame': r.get('frame_number', i), 'Timestamp': f"{r.get('timestamp', 0):.2f}s", 'Detections': len(r.get('detections', [])), 'Damage Detected': '✅' if r.get('damage_detected', False) else '❌', 'Processing Time (ms)': f"{r.get('processing_time', 0)*1000:.1f}"} for i, r in enumerate(results)]
    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

def display_processing_summary(results: List[Dict]):
    """Display processing summary"""
    st.subheader("📊 Processing Summary")
    
    if not results:
        st.warning("No frames were processed.")
        return

    total_frames = len(results)
    total_detections = sum(len(r.get('detections', [])) for r in results)
    damage_frames = sum(1 for r in results if r.get('damage_detected', False))
    avg_processing_time = np.mean([r.get('processing_time', 0) for r in results]) * 1000 if results else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Processed Frames", total_frames)
    col2.metric("Total Detections", total_detections)
    col3.metric("Avg Processing Time", f"{avg_processing_time:.1f} ms")

    st.subheader("🖼️ Sample Detections")
    frames_with_detections = [r for r in results if r.get('detections')]
    
    if frames_with_detections:
        cols = st.columns(min(3, len(frames_with_detections)))
        for i, result in enumerate(frames_with_detections[:3]):
            with cols[i]:
                annotated_frame = draw_detections_on_frame(result['frame'], result['detections'])
                st.image(annotated_frame, caption=f"Frame {result.get('frame_number', i)}", use_column_width=True)
    else:
        st.info("No detections found in the processed frames.")

if __name__ == "__main__":
    main()