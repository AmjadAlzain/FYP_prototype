"""
System Requirements Visualization for Container Anomaly Detection
Comprehensive evaluation of functional and non-functional requirements
Based on thesis proposal requirements and industry best practices
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import psutil
import cv2
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Import our inference modules
try:
    from local_inference_enhanced import EnhancedInferenceEngine, create_inference_engine
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False
    st.warning("Inference modules not available - running in demo mode")

# Set page config
st.set_page_config(
    page_title="System Requirements Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Updated System Requirements Database based on summarized requirements
FUNCTIONAL_REQUIREMENTS = {
    "FR1": {
        "title": "Real-Time Anomaly Detection",
        "description": "Detect container defects (dents, cracks, holes) with bounding boxes in real-time",
        "metrics": ["Detection Accuracy (mAP)", "Processing Latency", "False Positive Rate", "Detection Rate"],
        "target_values": {"accuracy": ">90%", "latency": "<50ms", "false_positive": "<5%", "detection_rate": ">95%"},
        "measurement_method": "Average inference time in milliseconds, mAP@0.5 score, confusion matrix analysis"
    },
    "FR2": {
        "title": "Dataset Processing & Augmentation",
        "description": "Real-time image preprocessing, augmentation, and data pipeline management",
        "metrics": ["Data Loading Speed", "Augmentation Efficiency", "Pipeline Throughput", "Memory Efficiency"],
        "target_values": {"loading_speed": ">30 FPS", "augmentation": "<10ms", "throughput": ">25 FPS", "memory": "<512MB"},
        "measurement_method": "Frames processed per second, augmentation time measurement, memory allocation tracking"
    },
    "FR3": {
        "title": "On-Device Training Capability",
        "description": "Adaptive learning with quantization-aware training and sparse updates on ESP32",
        "metrics": ["Training Speed", "Model Size", "Convergence Rate", "Adaptation Accuracy"],
        "target_values": {"training_speed": "<5min/epoch", "model_size": "<2MB", "convergence": "<10 epochs", "adaptation": ">85%"},
        "measurement_method": "Training time measurement, model file size, accuracy improvement tracking"
    },
    "FR4": {
        "title": "Feature Integration (CNN + HDC)",
        "description": "Seamless integration of Convolutional Neural Networks with Hyperdimensional Computing",
        "metrics": ["Integration Efficiency", "Feature Quality", "Classification Accuracy", "Processing Speed"],
        "target_values": {"efficiency": ">90%", "feature_quality": ">0.85", "classification": ">88%", "speed": "<30ms"},
        "measurement_method": "Feature extraction timing, classification accuracy measurement, integration overhead analysis"
    }
}

NON_FUNCTIONAL_REQUIREMENTS = {
    "NFR1": {
        "title": "Performance (Low Latency)",
        "description": "Achieve ultra-low latency for real-time container inspection applications",
        "metrics": ["Average Latency", "P95 Latency", "P99 Latency", "Throughput (FPS)", "Response Time"],
        "target_values": {"avg_latency": "<45ms", "p95_latency": "<75ms", "p99_latency": "<100ms", "throughput": ">20 FPS", "response_time": "<50ms"},
        "measurement_method": "Statistical analysis of inference times, percentile calculations, FPS measurement"
    },
    "NFR2": {
        "title": "Efficiency (ESP32 Constraints)",
        "description": "Operate efficiently within ESP32-S3-EYE hardware constraints",
        "metrics": ["Memory Usage", "Flash Storage", "CPU Utilization", "Power Consumption", "Model Size"],
        "target_values": {"memory": "<400KB", "storage": "<6MB", "cpu": "<80%", "power": "<2W", "model_size": "<2MB"},
        "measurement_method": "Real-time system monitoring, memory profiling, power measurement, resource utilization tracking"
    },
    "NFR3": {
        "title": "Scalability (Handle Diverse Data)",
        "description": "Process datasets with various container types, damage patterns, and environmental conditions",
        "metrics": ["Dataset Size Capacity", "Model Generalization", "Batch Processing Speed", "Cross-Domain Performance"],
        "target_values": {"dataset_size": ">10K images", "generalization": ">85%", "batch_speed": ">100 images/min", "cross_domain": ">80%"},
        "measurement_method": "Large dataset processing, cross-validation on diverse datasets, domain adaptation testing"
    },
    "NFR4": {
        "title": "Privacy (On-Device Processing)",
        "description": "Ensure complete data privacy through local processing without external data transmission",
        "metrics": ["Data Locality", "Processing Independence", "Privacy Compliance", "Security Level"],
        "target_values": {"locality": "100%", "independence": "100%", "compliance": "GDPR", "security": "AES-256"},
        "measurement_method": "Network traffic analysis, data flow auditing, privacy compliance verification"
    }
}

class SystemMonitor:
    """Real-time system monitoring for performance evaluation"""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()
        
    def get_current_metrics(self) -> Dict:
        """Get current system performance metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # GPU metrics (if available)
            gpu_available = False
            gpu_memory_used = 0
            gpu_utilization = 0
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_available = True
                    gpu_memory_used = gpu.memoryUsed
                    gpu_utilization = gpu.load * 100
            except ImportError:
                pass
            
            metrics = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_mb': memory.used / (1024**2),
                'memory_available_mb': memory.available / (1024**2),
                'gpu_available': gpu_available,
                'gpu_memory_used_mb': gpu_memory_used,
                'gpu_utilization': gpu_utilization
            }
            
            self.metrics_history.append(metrics)
            
            # Keep only last 100 measurements
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
                
            return metrics
            
        except Exception as e:
            st.error(f"Error getting system metrics: {e}")
            return {}

class PerformanceBenchmark:
    """Performance benchmarking for inference engine"""
    
    def __init__(self, inference_engine=None):
        self.inference_engine = inference_engine
        self.benchmark_results = []
        
    def run_latency_benchmark(self, num_samples: int = 100, image_size: Tuple[int, int] = (640, 480)) -> Dict:
        """Run latency benchmark with statistical analysis"""
        if not self.inference_engine:
            # Generate mock results for demo
            return self._generate_mock_latency_results(num_samples)
        
        latencies = []
        throughputs = []
        
        st.info(f"Running latency benchmark with {num_samples} samples...")
        progress_bar = st.progress(0)
        
        for i in range(num_samples):
            # Generate test image
            test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            
            # Measure inference time
            start_time = time.time()
            result = self.inference_engine.process_full_image(test_image)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Update progress
            progress_bar.progress((i + 1) / num_samples)
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        results = {
            'num_samples': num_samples,
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies),
            'avg_fps': 1000 / np.mean(latencies),
            'latencies': latencies.tolist()
        }
        
        self.benchmark_results.append({
            'timestamp': datetime.now(),
            'type': 'latency',
            'results': results
        })
        
        return results
    
    def _generate_mock_latency_results(self, num_samples: int) -> Dict:
        """Generate mock latency results for demo"""
        # Simulate realistic latency distribution
        base_latency = 45  # ms
        latencies = np.random.gamma(2, base_latency/2, num_samples)
        
        return {
            'num_samples': num_samples,
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies),
            'avg_fps': 1000 / np.mean(latencies),
            'latencies': latencies.tolist()
        }
    
    def run_memory_benchmark(self) -> Dict:
        """Run memory usage benchmark"""
        if not self.inference_engine:
            return self._generate_mock_memory_results()
        
        # Measure memory before and after model loading
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Run several inferences to measure steady-state memory
        test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        memory_samples = []
        for _ in range(10):
            self.inference_engine.process_full_image(test_image)
            current_memory = process.memory_info().rss / (1024**2)
            memory_samples.append(current_memory)
        
        results = {
            'initial_memory_mb': initial_memory,
            'avg_memory_mb': np.mean(memory_samples),
            'peak_memory_mb': np.max(memory_samples),
            'memory_overhead_mb': np.mean(memory_samples) - initial_memory
        }
        
        return results
    
    def _generate_mock_memory_results(self) -> Dict:
        """Generate mock memory results for demo"""
        return {
            'initial_memory_mb': 120.5,
            'avg_memory_mb': 145.2,
            'peak_memory_mb': 156.8,
            'memory_overhead_mb': 24.7
        }

def create_requirements_matrix() -> pd.DataFrame:
    """Create requirements evaluation matrix"""
    all_requirements = []
    
    # Add functional requirements
    for req_id, req_data in FUNCTIONAL_REQUIREMENTS.items():
        for metric in req_data["metrics"]:
            all_requirements.append({
                'Requirement ID': req_id,
                'Type': 'Functional',
                'Title': req_data["title"],
                'Metric': metric,
                'Target': req_data["target_values"].get(metric.lower().replace(" ", "_"), "TBD"),
                'Measurement Method': req_data["measurement_method"],
                'Status': 'Not Tested'
            })
    
    # Add non-functional requirements
    for req_id, req_data in NON_FUNCTIONAL_REQUIREMENTS.items():
        for metric in req_data["metrics"]:
            all_requirements.append({
                'Requirement ID': req_id,
                'Type': 'Non-Functional',
                'Title': req_data["title"],
                'Metric': metric,
                'Target': req_data["target_values"].get(metric.lower().replace(" ", "_").replace("(", "").replace(")", ""), "TBD"),
                'Measurement Method': req_data["measurement_method"],
                'Status': 'Not Tested'
            })
    
    return pd.DataFrame(all_requirements)

def create_performance_dashboard(benchmark_results: Dict, system_metrics: Dict):
    """Create comprehensive performance dashboard"""
    
    # Performance Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_latency = benchmark_results.get('avg_latency_ms', 0)
        st.metric(
            "Average Latency",
            f"{avg_latency:.1f} ms",
            delta=f"Target: <50ms",
            delta_color="normal" if avg_latency < 50 else "inverse"
        )
    
    with col2:
        avg_fps = benchmark_results.get('avg_fps', 0)
        st.metric(
            "Throughput",
            f"{avg_fps:.1f} FPS",
            delta=f"Target: >15 FPS",
            delta_color="normal" if avg_fps > 15 else "inverse"
        )
    
    with col3:
        memory_usage = system_metrics.get('memory_used_mb', 0)
        st.metric(
            "Memory Usage",
            f"{memory_usage:.1f} MB",
            delta=f"Target: <400MB",
            delta_color="normal" if memory_usage < 400 else "inverse"
        )
    
    with col4:
        cpu_usage = system_metrics.get('cpu_percent', 0)
        st.metric(
            "CPU Usage",
            f"{cpu_usage:.1f}%",
            delta=f"Target: <80%",
            delta_color="normal" if cpu_usage < 80 else "inverse"
        )
    
    # Latency Distribution Chart
    if 'latencies' in benchmark_results:
        st.subheader("📊 Latency Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                x=benchmark_results['latencies'],
                nbins=30,
                title="Latency Distribution",
                labels={'x': 'Latency (ms)', 'y': 'Frequency'}
            )
            fig_hist.add_vline(x=50, line_dash="dash", line_color="red", 
                             annotation_text="Target: 50ms")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=benchmark_results['latencies'],
                name="Latency",
                boxpoints="outliers"
            ))
            fig_box.update_layout(
                title="Latency Statistics",
                yaxis_title="Latency (ms)"
            )
            fig_box.add_hline(y=50, line_dash="dash", line_color="red",
                            annotation_text="Target: 50ms")
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Performance vs Requirements Chart
    st.subheader("🎯 Performance vs Requirements")
    
    performance_data = {
        'Metric': ['Avg Latency', 'P99 Latency', 'Throughput', 'Memory Usage', 'CPU Usage'],
        'Current': [
            benchmark_results.get('avg_latency_ms', 0),
            benchmark_results.get('p99_latency_ms', 0),
            benchmark_results.get('avg_fps', 0),
            system_metrics.get('memory_used_mb', 0),
            system_metrics.get('cpu_percent', 0)
        ],
        'Target': [50, 100, 15, 400, 80],
        'Unit': ['ms', 'ms', 'FPS', 'MB', '%']
    }
    
    fig_radar = go.Figure()
    
    # Normalize values to 0-1 scale for radar chart
    normalized_current = []
    normalized_target = []
    
    for i, (current, target) in enumerate(zip(performance_data['Current'], performance_data['Target'])):
        if performance_data['Metric'][i] in ['Avg Latency', 'P99 Latency', 'Memory Usage', 'CPU Usage']:
            # Lower is better
            normalized_current.append(max(0, 1 - current / target))
            normalized_target.append(1.0)
        else:
            # Higher is better (Throughput)
            normalized_current.append(min(1, current / target))
            normalized_target.append(1.0)
    
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_current,
        theta=performance_data['Metric'],
        fill='toself',
        name='Current Performance',
        line_color='blue'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_target,
        theta=performance_data['Metric'],
        fill='toself',
        name='Target Performance',
        line_color='green',
        opacity=0.5
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                ticktext=['0%', '25%', '50%', '75%', '100%'],
                tickvals=[0, 0.25, 0.5, 0.75, 1]
            )
        ),
        showlegend=True,
        title="Performance vs Requirements (Normalized)"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

def create_requirements_compliance_chart(df: pd.DataFrame):
    """Create requirements compliance visualization"""
    st.subheader("📋 Requirements Compliance Overview")
    
    # Summary by type
    col1, col2 = st.columns(2)
    
    with col1:
        # Functional vs Non-Functional breakdown
        type_counts = df['Type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Requirements by Type"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Status breakdown
        status_counts = df['Status'].value_counts()
        fig_status = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Requirements Testing Status",
            color=status_counts.index,
            color_discrete_map={
                'Passed': 'green',
                'Failed': 'red',
                'Not Tested': 'gray'
            }
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    # Detailed requirements table
    st.subheader("📊 Detailed Requirements Matrix")
    st.dataframe(df, use_container_width=True)

def main():
    """Main application"""
    st.title("🔍 Container Anomaly Detection System Requirements Evaluation")
    st.markdown("### Comprehensive analysis of functional and non-functional requirements")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Evaluation Controls")
    
    # Load inference engine
    inference_engine = None
    if INFERENCE_AVAILABLE:
        if st.sidebar.button("🔄 Load Inference Models"):
            with st.spinner("Loading TinyNAS + HDC models..."):
                models_dir = Path(__file__).parent.parent.parent.parent / "models"
                inference_engine = create_inference_engine(str(models_dir))
                if inference_engine:
                    st.sidebar.success("✅ Models loaded successfully")
                    st.session_state.inference_engine = inference_engine
                else:
                    st.sidebar.error("❌ Failed to load models")
    
    if 'inference_engine' in st.session_state:
        inference_engine = st.session_state.inference_engine
    
    # Evaluation options
    st.sidebar.subheader("📊 Evaluation Options")
    
    show_requirements_matrix = st.sidebar.checkbox("Show Requirements Matrix", value=True)
    show_performance_benchmark = st.sidebar.checkbox("Show Performance Benchmark", value=True)
    show_system_monitoring = st.sidebar.checkbox("Show System Monitoring", value=True)
    
    # Initialize components
    system_monitor = SystemMonitor()
    benchmark = PerformanceBenchmark(inference_engine)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Requirements Matrix", "⚡ Performance Benchmark", "📊 System Monitoring", "📈 Analysis & Reports"])
    
    with tab1:
        if show_requirements_matrix:
            st.header("Requirements Traceability Matrix")
            
            # Create and display requirements matrix
            requirements_df = create_requirements_matrix()
            create_requirements_compliance_chart(requirements_df)
            
            # Export options
            st.subheader("📤 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Export CSV"):
                    csv = requirements_df.to_csv(index=False)
                    st.download_button(
                        label="Download Requirements CSV",
                        data=csv,
                        file_name=f"requirements_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📋 Export JSON"):
                    json_data = {
                        'functional_requirements': FUNCTIONAL_REQUIREMENTS,
                        'non_functional_requirements': NON_FUNCTIONAL_REQUIREMENTS,
                        'evaluation_timestamp': datetime.now().isoformat()
                    }
                    st.download_button(
                        label="Download Requirements JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"requirements_spec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("📈 Generate Report"):
                    st.success("Comprehensive report generation will be implemented in the full system")
    
    with tab2:
        if show_performance_benchmark:
            st.header("Performance Benchmarking")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🚀 Latency Benchmark")
                
                # Benchmark parameters
                num_samples = st.slider("Number of samples", 10, 1000, 100)
                image_width = st.selectbox("Image width", [320, 640, 1280], index=1)
                image_height = st.selectbox("Image height", [240, 480, 720], index=1)
                
                if st.button("▶️ Run Latency Benchmark", type="primary"):
                    benchmark_results = benchmark.run_latency_benchmark(
                        num_samples=num_samples,
                        image_size=(image_width, image_height)
                    )
                    st.session_state.benchmark_results = benchmark_results
            
            with col2:
                st.subheader("💾 Memory Benchmark")
                
                if st.button("▶️ Run Memory Benchmark"):
                    memory_results = benchmark.run_memory_benchmark()
                    st.session_state.memory_results = memory_results
                    
                    # Display memory results
                    st.metric("Model Memory Overhead", f"{memory_results['memory_overhead_mb']:.1f} MB")
                    st.metric("Peak Memory Usage", f"{memory_results['peak_memory_mb']:.1f} MB")
            
            # Display benchmark results
            if 'benchmark_results' in st.session_state:
                system_metrics = system_monitor.get_current_metrics()
                create_performance_dashboard(st.session_state.benchmark_results, system_metrics)
    
    with tab3:
        if show_system_monitoring:
            st.header("Real-Time System Monitoring")
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox("🔄 Auto-refresh every 5 seconds", value=False)
            
            if auto_refresh:
                time.sleep(5)
                st.rerun()
            
            # Get current metrics
            current_metrics = system_monitor.get_current_metrics()
            
            if current_metrics:
                # System overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "CPU Usage",
                        f"{current_metrics['cpu_percent']:.1f}%",
                        delta=f"Target: <80%"
                    )
                
                with col2:
                    st.metric(
                        "Memory Usage",
                        f"{current_metrics['memory_percent']:.1f}%",
                        delta=f"{current_metrics['memory_used_mb']:.0f} MB"
                    )
                
                with col3:
                    if current_metrics['gpu_available']:
                        st.metric(
                            "GPU Usage",
                            f"{current_metrics['gpu_utilization']:.1f}%",
                            delta=f"{current_metrics['gpu_memory_used_mb']:.0f} MB"
                        )
                    else:
                        st.metric("GPU", "Not Available", delta="Using CPU")
                
                with col4:
                    uptime_hours = (time.time() - system_monitor.start_time) / 3600
                    st.metric(
                        "Uptime",
                        f"{uptime_hours:.1f}h",
                        delta="System running"
                    )
                
                # Historical charts
                if len(system_monitor.metrics_history) > 1:
                    st.subheader("📈 System Performance Trends")
                    
                    # Prepare data for plotting
                    df_metrics = pd.DataFrame(system_monitor.metrics_history)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CPU and Memory usage over time
                        fig_cpu_mem = go.Figure()
                        fig_cpu_mem.add_trace(go.Scatter(
                            x=df_metrics['timestamp'],
                            y=df_metrics['cpu_percent'],
                            mode='lines',
                            name='CPU %',
                            line=dict(color='blue')
                        ))
                        fig_cpu_mem.add_trace(go.Scatter(
                            x=df_metrics['timestamp'],
                            y=df_metrics['memory_percent'],
                            mode='lines',
                            name='Memory %',
                            line=dict(color='red'),
                            yaxis='y2'
                        ))
                        fig_cpu_mem.update_layout(
                            title='CPU and Memory Usage Over Time',
                            xaxis_title='Time',
                            yaxis_title='CPU %',
                            yaxis2=dict(title='Memory %', overlaying='y', side='right'),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_cpu_mem, use_container_width=True)
                    
                    with col2:
                        # Memory usage details
                        fig_mem = px.line(
                            df_metrics,
                            x='timestamp',
                            y='memory_used_mb',
                            title='Memory Usage (MB)',
                            labels={'memory_used_mb': 'Memory (MB)', 'timestamp': 'Time'}
                        )
                        fig_mem.add_hline(y=400, line_dash="dash", line_color="red",
                                        annotation_text="Target: 400MB")
                        st.plotly_chart(fig_mem, use_container_width=True)
    
    with tab4:
        st.header("Analysis & Comprehensive Reports")
        
        # Summary analysis
        st.subheader("🎯 Requirements Fulfillment Analysis")
        
        # Mock analysis data
        fulfillment_data = {
            'Requirement Category': ['Real-Time Performance', 'Resource Efficiency', 'Scalability', 
                                   'Privacy & Security', 'Reliability', 'Usability'],
            'Fulfillment %': [85, 92, 78, 100, 88, 75],
            'Status': ['Good', 'Excellent', 'Needs Improvement', 'Excellent', 'Good', 'Fair']
        }
        
        df_fulfillment = pd.DataFrame(fulfillment_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fulfillment bar chart
            fig_fulfillment = px.bar(
                df_fulfillment,
                x='Requirement Category',
                y='Fulfillment %',
                color='Status',
                title='Requirements Fulfillment by Category',
                color_discrete_map={
                    'Excellent': 'green',
                    'Good': 'blue',
                    'Fair': 'orange',
                    'Needs Improvement': 'red'
                }
            )
            fig_fulfillment.add_hline(y=80, line_dash="dash", line_color="orange",
                                    annotation_text="Target: 80%")
            st.plotly_chart(fig_fulfillment, use_container_width=True)
        
        with col2:
            # Requirements overview pie chart
            req_status_data = {
                'Status': ['Implemented', 'In Progress', 'Planned', 'Testing'],
                'Count': [6, 3, 2, 5]
            }
            
            fig_req_pie = px.pie(
                values=req_status_data['Count'],
                names=req_status_data['Status'],
                title='Overall Requirements Status',
                color_discrete_map={
                    'Implemented': 'green',
                    'In Progress': 'orange',
                    'Planned': 'gray',
                    'Testing': 'blue'
                }
            )
            st.plotly_chart(fig_req_pie, use_container_width=True)

if __name__ == "__main__":
    main()
