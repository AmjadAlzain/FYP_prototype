# Module6/container_detection_gui_realtime.py
"""container_detection_gui_realtime_v3.py
=========================================
A *single‑file* PyQt6 application that performs **real‑time multi‑container anomaly
(damage) detection** on camera, video file, or RTSP streams using the shared
``EnhancedInferenceEngine``.  It includes several improvements over the v2 draft
sent by your teammate:

* **Exclusive source selection** – camera / file / RTSP via a radio‑button group
  so only one source can be active at a time.
* **Optional MP4 recording** – save annotated output to ``./recordings/`` with a
  timestamped filename.
* **Theme toggle** – built‑in dark & light palettes.
* **FPS control** – user‑settable target FPS (5‑30 fps).
* **Graceful shutdown** – threaded capture + writer close even on abrupt stop.
* **Improved overlays** – per‑detection labels show ``box`` & ``damage`` confidences
  in a rounded, semi‑transparent pill; red = damaged, green = ok.
* **GPU/CPU status** – shows whether inference is running on CUDA or CPU in the
  status‑bar.

The file is self‑contained – drop it next to your existing
``local_inference_enhanced.py`` and run ``python container_detection_gui_realtime_v3.py``.
"""

from __future__ import annotations

import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Union

import numpy as np
import cv2

from PyQt6.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QMutex,
    QMutexLocker,
)
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QFont,
    QPalette,
    QColor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QSlider,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QRadioButton,
    QButtonGroup,
    QCheckBox,
    QSpinBox,
    QLineEdit,
    QSplitter,
)

# ---------------------------------------------------------------------------
# Qt application guard – make sure a QApplication exists before any QWidget
# ---------------------------------------------------------------------------
_QT_APP_SINGLETON = QApplication.instance() or QApplication(sys.argv)


# -----------------------------------------------------------------------------
# Local inference engine import
# -----------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

try:
    from local_inference_enhanced import (
        EnhancedInferenceEngine,
        create_inference_engine,
        DetectionResult,
    )
except Exception as e:  # pragma: no cover – fatal at runtime
    QMessageBox.critical(None, "Import Error", f"Failed to import inference module: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------
COLORS = {
    "primary": "#0078D4",
    "success": "#107C10",
    "danger": "#D13438",
    "surface": "#1E1E1E",
    "text_light": "#FFFFFF",
    "text_dark": "#323130",
}


def apply_dark_palette(app: QApplication) -> None:
    """Apply a nice dark palette to ``app``."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#121212"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#202124"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2A2D2E"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2D2F31"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#FF0000"))
    palette.setColor(QPalette.ColorRole.Link, QColor(COLORS["primary"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["primary"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#000000"))
    app.setPalette(palette)


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------
class RealTimeVideoProcessor(QThread):
    """Process frames in a background thread and emit annotated results."""

    frame_ready = pyqtSignal(np.ndarray)  # already RGB & annotated
    error_occurred = pyqtSignal(str)
    fps_computed = pyqtSignal(float)

    def __init__(self, engine: EnhancedInferenceEngine):
        super().__init__()
        self.engine = engine
        self._source: Union[int, str, None] = None
        self._running = False
        self._mutex = QMutex()
        self.target_fps = 15
        self._confidence = 0.4
        # Recording
        self._record = False
        self._vw: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[Path] = None

    # ------------------------------------------------------------------ public
    def configure(
        self,
        source: Union[int, str],
        confidence: float,
        target_fps: int,
        record: bool,
        record_dir: Path,
    ) -> None:
        """Configure run parameters. Should be called *before* ``start()``."""
        with QMutexLocker(self._mutex):
            self._source = source
            self._confidence = confidence
            self.target_fps = target_fps
            self._record = record
            if record:
                record_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._record_path = record_dir / f"container_detection_{ts}.mp4"

    def stop(self) -> None:  # slot
        with QMutexLocker(self._mutex):
            self._running = False
        self.wait(2000)  # wait for thread to finish (max 2 s)

    # --------------------------------------------------------------- QThread
    def run(self) -> None:  # noqa: C901 – okay for a worker loop
        if self._source is None:
            self.error_occurred.emit("No source configured")
            return

        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            self.error_occurred.emit(f"Cannot open source: {self._source}")
            return

        # prepare writer if recording
        if self._record and self._record_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = max(5, min(30, self.target_fps))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._vw = cv2.VideoWriter(str(self._record_path), fourcc, fps, (w, h))
            if not self._vw.isOpened():
                self.error_occurred.emit("Failed to open video writer – recording disabled")
                self._record = False

        self._running = True
        frame_interval = 1.0 / self.target_fps
        try:
            while True:
                with QMutexLocker(self._mutex):
                    if not self._running:
                        break
                start = time.time()
                ok, frame_bgr = cap.read()
                if not ok:
                    break  # EOF or error

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # inference – returns list[DetectionResult]
                self.engine.confidence_threshold = self._confidence
                results = self.engine.process_full_image(frame_rgb)

                annotated = self._draw_detections(frame_rgb.copy(), results.get("detections", []))
                self.frame_ready.emit(annotated)

                # recording (RGB→BGR for OpenCV)
                if self._record and self._vw is not None:
                    self._vw.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                elapsed = time.time() - start
                self.fps_computed.emit(1.0 / elapsed if elapsed else 0)
                sleep_time = max(0.0, frame_interval - elapsed)
                if sleep_time:
                    self.msleep(int(sleep_time * 1000))
        except Exception as exc:  # pragma: no cover
            self.error_occurred.emit(f"Processing error: {exc}")
        finally:
            cap.release()
            if self._vw:
                self._vw.release()

    # ---------------------------------------------------------------- private
    @staticmethod
    def _draw_detections(frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Overlay bounding boxes & labels on *frame* (RGB)."""
        for det in detections:
            color = (209, 52, 56) if det.is_damaged else (16, 124, 16)  # RGB
            x, y, w, h = det.x, det.y, det.w, det.h
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

            # label text
            dmg_lbl = det.damage_type.upper() if det.is_damaged else "OK"
            txt = f"{dmg_lbl} | box {det.box_confidence*100:.0f}% | dmg {det.damage_confidence*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # rounded pill background
            bg = frame.copy()
            cv2.rectangle(bg, (x, max(0, y - th - 12)), (x + tw + 14, y), color, -1, cv2.LINE_AA)
            cv2.addWeighted(bg, 0.4, frame, 0.6, 0, frame)

            # text (black for contrast)
            cv2.putText(
                frame,
                txt,
                (x + 7, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return frame


# -----------------------------------------------------------------------------
# Display label
# -----------------------------------------------------------------------------
class VideoCanvas(QLabel):
    """Video display widget scaling content while keeping aspect ratio."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color:#000; border:2px solid #444; border-radius:8px;")
        self.setText("Select a source and press *Start*…")
        self.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))

    def show_frame(self, frame_rgb: np.ndarray) -> None:
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Real‑Time Container Detection v3")
        self.resize(1450, 850)

        # inference engine
        self.engine = create_inference_engine()
        if not self.engine:
            QMessageBox.critical(self, "Inference Error", "Failed to initialise inference engine – exiting.")
            sys.exit(2)
        device_is_cuda: bool
        if getattr(self.engine, "using_gpu", None) is not None:
            device_is_cuda = bool(self.engine.using_gpu)
        elif hasattr(self.engine, "device"):
            device_is_cuda = str(self.engine.device).lower().startswith("cuda")
        else:
            device_is_cuda = False
        gpu_flag = "CUDA" if device_is_cuda else "CPU"

        # widgets -----------------------------------------------------
        self.video = VideoCanvas()

        # --- controls frame
        ctrl_box = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_box)

        # Source selection group --------------------------------------
        src_grp = QGroupBox("Source")
        src_layout = QGridLayout(src_grp)
        self.r_cam = QRadioButton("Camera")
        self.r_file = QRadioButton("Video file")
        self.r_rtsp = QRadioButton("RTSP URL")
        self.r_cam.setChecked(True)
        self.src_buttons = QButtonGroup()
        for rb in (self.r_cam, self.r_file, self.r_rtsp):
            self.src_buttons.addButton(rb)
        self.combo_cam = QComboBox()
        self._populate_cameras()
        self.le_file = QLineEdit()
        self.btn_browse = QPushButton("Browse…")
        self.le_rtsp = QLineEdit()
        self.le_rtsp.setPlaceholderText("rtsp://<user>:<pass>@host:554/stream")

        src_layout.addWidget(self.r_cam, 0, 0)
        src_layout.addWidget(self.combo_cam, 0, 1)
        src_layout.addWidget(self.r_file, 1, 0)
        src_layout.addWidget(self.le_file, 1, 1)
        src_layout.addWidget(self.btn_browse, 1, 2)
        src_layout.addWidget(self.r_rtsp, 2, 0)
        src_layout.addWidget(self.le_rtsp, 2, 1, 1, 2)

        # Processing group --------------------------------------------
        proc_grp = QGroupBox("Processing")
        proc_layout = QGridLayout(proc_grp)
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(40)
        self.lbl_conf = QLabel("Confidence: 40%")
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf.setText(f"Confidence: {v}%"))

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(5, 30)
        self.spin_fps.setValue(15)
        self.chk_record = QCheckBox("Record annotated output")

        proc_layout.addWidget(self.lbl_conf, 0, 0)
        proc_layout.addWidget(self.slider_conf, 0, 1)
        proc_layout.addWidget(QLabel("FPS:"), 1, 0)
        proc_layout.addWidget(self.spin_fps, 1, 1)
        proc_layout.addWidget(self.chk_record, 2, 0, 1, 2)

        # Theme + Start/Stop -----------------------------------------
        self.chk_dark = QCheckBox("Dark theme")
        self.chk_dark.setChecked(True)
        self.btn_start = QPushButton("▶ Start")
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setEnabled(False)

        ctrl_layout.addWidget(src_grp)
        ctrl_layout.addWidget(proc_grp)
        ctrl_layout.addWidget(self.chk_dark)
        ctrl_layout.addStretch(1)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)

        # splitter ----------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(ctrl_box)
        splitter.setSizes([1100, 350])
        self.setCentralWidget(splitter)

        # status bar --------------------------------------------------
        self.label_fps = QLabel("FPS: –")
        self.statusBar().addPermanentWidget(self.label_fps)
        self.statusBar().showMessage(f"Models ready ({gpu_flag}) – select source and press Start…")

        # theme apply
        apply_dark_palette(QApplication.instance())

        # worker thread ----------------------------------------------
        self.worker: Optional[RealTimeVideoProcessor] = None

        # connections -------------------------------------------------
        self.btn_browse.clicked.connect(self._select_file)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.chk_dark.toggled.connect(self._toggle_theme)
        self.src_buttons.buttonToggled.connect(self._update_src_widgets)

        self._update_src_widgets()  # enable/disable widgets

    # ---------------------------------------------------------------- private helpers
    def _populate_cameras(self) -> None:
        self.combo_cam.clear()
        found = False
        for idx in range(5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.combo_cam.addItem(f"Camera {idx}", idx)
                found = True
            cap.release()
        if not found:
            self.combo_cam.addItem("<no camera>", -1)

    def _update_src_widgets(self) -> None:
        cam = self.r_cam.isChecked()
        file_ = self.r_file.isChecked()
        rtsp = self.r_rtsp.isChecked()
        self.combo_cam.setEnabled(cam)
        self.le_file.setEnabled(file_)
        self.btn_browse.setEnabled(file_)
        self.le_rtsp.setEnabled(rtsp)

    def _select_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Videos (*.mp4 *.avi *.mov)")
        if fn:
            self.le_file.setText(fn)

    # ---------------------------------------------------------------- start/stop
    def _start(self):
        # resolve source
        if self.r_cam.isChecked():
            idx = int(self.combo_cam.currentData())
            if idx == -1:
                QMessageBox.warning(self, "No Camera", "No camera found or selected.")
                return
            source: Union[int, str] = idx
        elif self.r_file.isChecked():
            path = self.le_file.text().strip()
            if not path or not Path(path).exists():
                QMessageBox.warning(self, "Invalid file", "Please select a valid video file.")
                return
            source = path
        else:  # RTSP
            url = self.le_rtsp.text().strip()
            if not url.lower().startswith("rtsp://"):
                QMessageBox.warning(self, "Invalid RTSP", "Please enter a valid RTSP URL starting with rtsp://")
                return
            source = url

        # instantiate worker
        self.worker = RealTimeVideoProcessor(self.engine)
        self.worker.frame_ready.connect(self.video.show_frame)
        self.worker.error_occurred.connect(lambda msg: QMessageBox.critical(self, "Processing error", msg))
        self.worker.fps_computed.connect(lambda fps: self.label_fps.setText(f"FPS: {fps:.1f}"))

        self.worker.configure(
            source=source,
            confidence=self.slider_conf.value() / 100.0,
            target_fps=self.spin_fps.value(),
            record=self.chk_record.isChecked(),
            record_dir=Path("./recordings"),
        )
        self.worker.start()

        # UI state
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Processing… press Stop to end.")

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.label_fps.setText("FPS: –")
        self.statusBar().showMessage("Stopped.")

    def closeEvent(self, event):  # noqa: N802 – Qt naming style
        self._stop()
        super().closeEvent(event)

    # ---------------------------------------------------------------- theme
    def _toggle_theme(self, dark: bool):
        if dark:
            apply_dark_palette(QApplication.instance())
        else:
            QApplication.instance().setPalette(QApplication().style().standardPalette())


# -----------------------------------------------------------------------------
# entry‑point
# -----------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    """Run the application."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
