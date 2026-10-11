"""Sage Model Chain Studio -- Video Cleanup / Upscaling (Real-ESRGAN)"""
import os
import time
import shutil
import subprocess

from PyQt6.QtCore import QThread, pyqtSignal, QTime, QDateTime, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QFileDialog, QGroupBox, QFormLayout, QProgressBar,
    QPlainTextEdit, QCheckBox, QDoubleSpinBox, QSpinBox, QMessageBox,
    QTimeEdit, QDateTimeEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)

from .constants import (
    REALESRGAN_MODEL_DIR, REALESRGAN_WEIGHTS, GFPGAN_WEIGHT_URL,
    REALESRGAN_PIP_HINT,
)
from .utils import _cuda_available

PRESETS = {
    "Light Cleanup (no resize)": {
        "model": "realesr-general-x4v3", "outscale": 1.0,
        "denoise_strength": 0.4, "face_enhance": False,
        "desc": ("Gentle denoise + detail restoration. Keeps original resolution. "
                 "Good default for decent-quality footage with mild noise or softness."),
    },
    "Strong Cleanup (no resize)": {
        "model": "realesr-general-x4v3", "outscale": 1.0,
        "denoise_strength": 0.8, "face_enhance": False,
        "desc": ("Heavier denoise + compression-artifact removal for noisier/blockier "
                 "footage. Keeps original resolution."),
    },
    "Cleanup + Face Fix (no resize)": {
        "model": "realesr-general-x4v3", "outscale": 1.0,
        "denoise_strength": 0.5, "face_enhance": True,
        "desc": ("Same as Light Cleanup, plus GFPGAN face restoration. Best for "
                 "interviews / talking-head footage where faces look soft or waxy."),
    },
    "Cleanup + 2x Upscale": {
        "model": "realesr-general-x4v3", "outscale": 2.0,
        "denoise_strength": 0.5, "face_enhance": False,
        "desc": "Denoise/sharpen and double the resolution (e.g. 480p to 960p).",
    },
    "Max Upscale 4x (general)": {
        "model": "RealESRGAN_x4plus", "outscale": 4.0,
        "denoise_strength": 0.5, "face_enhance": False,
        "desc": "Full 4x AI super-resolution for general/live-action footage.",
    },
    "Max Upscale 4x (anime/cartoon)": {
        "model": "RealESRGAN_x4plus_anime_6B", "outscale": 4.0,
        "denoise_strength": 0.5, "face_enhance": False,
        "desc": "4x upscale tuned specifically for anime/flat-shaded 2D content.",
    },
    "Custom": {
        "model": "realesr-general-x4v3", "outscale": 1.0,
        "denoise_strength": 0.5, "face_enhance": False,
        "desc": "Set every option manually below.",
    },
}


def _format_eta(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class VideoUpscaleWorker(QThread):
    progress = pyqtSignal(int, int, float, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, input_path, output_path, model_name, outscale,
                 denoise_strength, face_enhance, tile, trim_start=0.0, trim_end=0.0):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.outscale = outscale
        self.denoise_strength = denoise_strength
        self.face_enhance = face_enhance
        self.tile = tile
        self.trim_start = max(0.0, trim_start)
        self.trim_end = max(0.0, trim_end)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")

    def _build_upsampler(self):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer

        info = REALESRGAN_WEIGHTS[self.model_name]
        arch = info["arch"]
        dni_weight = None

        if arch == "srvgg":
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=32, upscale=4, act_type="prelu",
            )
            model_path = load_file_from_url(info["url"], model_dir=REALESRGAN_MODEL_DIR, progress=True)
            if self.denoise_strength < 1.0 and "wdn_url" in info:
                wdn_path = load_file_from_url(info["wdn_url"], model_dir=REALESRGAN_MODEL_DIR, progress=True)
                dni_weight = [self.denoise_strength, 1 - self.denoise_strength]
                model_path = [model_path, wdn_path]
        elif arch == "rrdbnet_anime":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=6, num_grow_ch=32, scale=4)
            model_path = load_file_from_url(info["url"], model_dir=REALESRGAN_MODEL_DIR, progress=True)
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                             num_block=23, num_grow_ch=32, scale=4)
            model_path = load_file_from_url(info["url"], model_dir=REALESRGAN_MODEL_DIR, progress=True)

        half = _cuda_available()
        return RealESRGANer(
            scale=4, model_path=model_path, dni_weight=dni_weight,
            model=model, tile=self.tile, tile_pad=10, pre_pad=0,
            half=half, gpu_id=None,
        )

    def _build_face_enhancer(self, upsampler):
        from gfpgan import GFPGANer
        from basicsr.utils.download_util import load_file_from_url
        model_path = load_file_from_url(GFPGAN_WEIGHT_URL, model_dir=REALESRGAN_MODEL_DIR, progress=True)
        return GFPGANer(
            model_path=model_path, upscale=self.outscale, arch="clean",
            channel_multiplier=2, bg_upsampler=upsampler,
        )

    def _run(self):
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python is not installed.\n\n" + REALESRGAN_PIP_HINT)

        try:
            self.log.emit("Loading Real-ESRGAN model (first run downloads weights, be patient)...")
            upsampler = self._build_upsampler()
        except ImportError:
            raise RuntimeError("realesrgan/basicsr are not installed.\n\n" + REALESRGAN_PIP_HINT)

        if _cuda_available():
            self.log.emit("Inference precision: FP16 (half precision, GPU/CUDA detected).")
        else:
            self.log.emit("Inference precision: FP32 (CPU only -- FP16 requires a CUDA GPU).")

        face_enhancer = None
        if self.face_enhance:
            self.log.emit("Loading GFPGAN face-restoration model...")
            try:
                face_enhancer = self._build_face_enhancer(upsampler)
            except ImportError:
                raise RuntimeError("gfpgan is not installed.\n\n" + REALESRGAN_PIP_HINT)

        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        start_frame = int(round(self.trim_start * fps)) if self.trim_start > 0 else 0
        end_frame = int(round(self.trim_end * fps)) if self.trim_end > 0 else None
        if total_frames:
            start_frame = min(start_frame, max(total_frames - 1, 0))
            if end_frame is not None:
                end_frame = min(end_frame, total_frames)
        if end_frame is not None and end_frame <= start_frame:
            raise RuntimeError("Trim end time must be after the start time.")
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.log.emit(f"Trimming: starting at {self.trim_start:.1f}s (frame {start_frame}).")
        if end_frame is not None:
            self.log.emit(f"Trimming: stopping at {self.trim_end:.1f}s (frame {end_frame}).")
        trimmed_total = (end_frame - start_frame) if end_frame is not None else (
            (total_frames - start_frame) if total_frames else 0
        )

        out_dir = os.path.dirname(os.path.abspath(self.output_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        tmp_video = self.output_path + ".noaudio.mp4"

        writer = None
        frame_idx = 0
        start_time = time.time()

        while True:
            if self._stop_requested:
                self.log.emit("Stop requested -- finishing current frame and exiting...")
                break
            if end_frame is not None and (start_frame + frame_idx) >= end_frame:
                self.log.emit("Reached trim end time.")
                break
            ok, frame = cap.read()
            if not ok:
                break

            if face_enhancer is not None:
                _, _, enhanced = face_enhancer.enhance(
                    frame, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                enhanced, _ = upsampler.enhance(frame, outscale=self.outscale)

            if writer is None:
                out_h, out_w = enhanced.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp_video, fourcc, fps, (out_w, out_h))

            writer.write(enhanced)
            frame_idx += 1
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
            remaining_frames = max(0, trimmed_total - frame_idx) if trimmed_total else 0
            eta_str = _format_eta(remaining_frames / fps_proc) if (fps_proc > 0 and trimmed_total) else "--:--:--"
            if trimmed_total:
                self.progress.emit(frame_idx, trimmed_total, fps_proc, eta_str)
            if frame_idx % 30 == 0:
                self.log.emit(
                    f"Processed {frame_idx}/{trimmed_total or '?'} frames "
                    f"({fps_proc:.2f} fps, ETA {eta_str})"
                )

        cap.release()
        if writer is not None:
            writer.release()

        if frame_idx == 0:
            raise RuntimeError("No frames were processed (empty, unreadable, or fully-trimmed-out video).")

        self.log.emit("Muxing original audio back into output...")
        self._mux_audio(tmp_video, self.input_path, self.output_path, self.trim_start, self.trim_end)
        try:
            os.remove(tmp_video)
        except OSError:
            pass

        self.finished_ok.emit(self.output_path)

    def _mux_audio(self, video_only_path, original_path, final_path, trim_start=0.0, trim_end=0.0):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            shutil.copyfile(video_only_path, final_path)
            self.log.emit("ffmpeg not found on PATH -- output saved WITHOUT audio.")
            return
        cmd = [ffmpeg, "-y", "-i", video_only_path]
        if trim_start > 0:
            cmd += ["-ss", str(trim_start)]
        cmd += ["-i", original_path]
        cmd += [
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest", final_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.log.emit("Audio mux failed, saving video-only output.\n" + result.stderr[-800:])
            shutil.copyfile(video_only_path, final_path)


# Table columns for the batch queue
COL_FILE = 0
COL_START = 1
COL_END = 2
COL_STATUS = 3


class VideoPanel(QWidget):
    def __init__(self, persistence):
        super().__init__()
        self.persistence = persistence
        self.worker = None
        self.queue = []          # list of dicts: {path, output, start_edit, end_edit}
        self.queue_index = -1    # index currently processing, -1 = idle
        self._batch_running = False
        self._schedule_armed = False

        layout = QVBoxLayout(self)

        title = QLabel("\U0001F3AC Video Cleanup / Upscaling (Real-ESRGAN, local & free)")
        title.setObjectName("sectionHint")
        layout.addWidget(title)

        gpu_txt = ("GPU (CUDA) available -- will run at full speed" if _cuda_available()
                   else "CPU only -- this will be slow, a GPU is strongly recommended")
        gpu_label = QLabel(f"Device: {gpu_txt}")
        gpu_label.setObjectName("sectionHint")
        layout.addWidget(gpu_label)

        # ---- Batch queue ----
        queue_group = QGroupBox("Batch Queue")
        queue_layout = QVBoxLayout(queue_group)

        queue_btn_row = QHBoxLayout()
        add_files_btn = QPushButton("+ Add Files...")
        add_files_btn.setObjectName("accent")
        add_files_btn.clicked.connect(self._add_files)
        queue_btn_row.addWidget(add_files_btn)
        remove_selected_btn = QPushButton("Remove Selected")
        remove_selected_btn.clicked.connect(self._remove_selected_queue_rows)
        queue_btn_row.addWidget(remove_selected_btn)
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setObjectName("danger")
        clear_all_btn.clicked.connect(self._clear_queue)
        queue_btn_row.addWidget(clear_all_btn)
        queue_layout.addLayout(queue_btn_row)

        self.queue_table = QTableWidget(0, 4)
        self.queue_table.setHorizontalHeaderLabels(["File", "Start Time", "End Time", "Status"])
        self.queue_table.horizontalHeader().setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setMaximumHeight(220)
        queue_layout.addWidget(self.queue_table)

        queue_hint = QLabel(
            "Add one or more video files. All process with the same settings below -- only "
            "Start/End trim time is per-file (leave both at 00:00:00 to use the whole file). "
            "Output is saved next to each input as '<name>_enhanced.mp4'. To select a row for "
            "removal, click its File or Status cell (not the time fields)."
        )
        queue_hint.setWordWrap(True)
        queue_hint.setObjectName("sectionHint")
        queue_layout.addWidget(queue_hint)

        layout.addWidget(queue_group)

        # ---- Schedule ----
        schedule_group = QGroupBox("Schedule (optional -- e.g. run overnight)")
        schedule_form = QFormLayout(schedule_group)

        self.schedule_check = QCheckBox("Start automatically at a scheduled time")
        self.schedule_check.toggled.connect(self._on_schedule_toggled)
        schedule_form.addRow(self.schedule_check)

        self.schedule_datetime = QDateTimeEdit()
        self.schedule_datetime.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.schedule_datetime.setDateTime(QDateTime.currentDateTime().addSecs(3600))
        self.schedule_datetime.setCalendarPopup(True)
        self.schedule_datetime.setEnabled(False)
        schedule_form.addRow("Start At:", self.schedule_datetime)

        self.schedule_status_label = QLabel("")
        self.schedule_status_label.setObjectName("sectionHint")
        schedule_form.addRow(self.schedule_status_label)

        layout.addWidget(schedule_group)

        self._schedule_timer = QTimer(self)
        self._schedule_timer.timeout.connect(self._check_schedule)
        self._schedule_timer.start(1000)

        # ---- Preset ----
        preset_group = QGroupBox("Preset")
        preset_layout = QVBoxLayout(preset_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        self.preset_desc = QLabel()
        self.preset_desc.setWordWrap(True)
        self.preset_desc.setObjectName("sectionHint")
        preset_layout.addWidget(self.preset_desc)
        layout.addWidget(preset_group)

        # ---- Advanced ----
        adv_group = QGroupBox("Advanced (editable for any preset, required for Custom -- applies to every queued file)")
        adv_form = QFormLayout(adv_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(list(REALESRGAN_WEIGHTS.keys()))
        adv_form.addRow("Model:", self.model_combo)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(1.0, 4.0)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setToolTip(
            "Output size relative to input.\n"
            "1.0 = keep original resolution (cleanup/sharpen only).\n"
            "2.0-4.0 = also upscale resolution."
        )
        adv_form.addRow("Scale Factor:", self.scale_spin)

        self.denoise_spin = QDoubleSpinBox()
        self.denoise_spin.setRange(0.0, 1.0)
        self.denoise_spin.setSingleStep(0.1)
        self.denoise_spin.setToolTip(
            "Only applies to the 'realesr-general-x4v3' model.\n"
            "0.0 = sharpest/most detail, least noise removal.\n"
            "1.0 = strongest denoise, softer result.\n"
            "0.4-0.6 is a good starting point for compressed footage."
        )
        adv_form.addRow("Denoise Strength:", self.denoise_spin)

        self.face_check = QCheckBox("Restore faces with GFPGAN (best for interviews/talking-head footage)")
        adv_form.addRow(self.face_check)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 2048)
        self.tile_spin.setValue(0)
        self.tile_spin.setToolTip(
            "Splits each frame into tiles to reduce VRAM use.\n"
            "0 = no tiling (fastest, needs the most VRAM).\n"
            "If you get CUDA out-of-memory errors, try 256 or 512."
        )
        adv_form.addRow("Tile Size (0 = off):", self.tile_spin)

        layout.addWidget(adv_group)

        # ---- Run controls ----
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25B6 Start Processing")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._on_start_clicked)
        self.cancel_btn = QPushButton("\u23F9 Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.batch_progress_label = QLabel("")
        self.batch_progress_label.setObjectName("sectionHint")
        layout.addWidget(self.batch_progress_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        self.eta_label.setObjectName("sectionHint")
        layout.addWidget(self.eta_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        layout.addWidget(self.log_view, 1)

        self._on_preset_changed(self.preset_combo.currentText())

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select video file(s)", "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*)"
        )
        for path in paths:
            self._add_queue_row(path)

    def _add_queue_row(self, path):
        row = self.queue_table.rowCount()
        self.queue_table.insertRow(row)

        self.queue_table.setItem(row, COL_FILE, QTableWidgetItem(os.path.basename(path)))
        self.queue_table.item(row, COL_FILE).setToolTip(path)

        start_edit = QTimeEdit()
        start_edit.setDisplayFormat("HH:mm:ss")
        start_edit.setTime(QTime(0, 0, 0))
        self.queue_table.setCellWidget(row, COL_START, start_edit)

        end_edit = QTimeEdit()
        end_edit.setDisplayFormat("HH:mm:ss")
        end_edit.setTime(QTime(0, 0, 0))
        self.queue_table.setCellWidget(row, COL_END, end_edit)

        self.queue_table.setItem(row, COL_STATUS, QTableWidgetItem("Queued"))

        base, ext = os.path.splitext(path)
        output_path = f"{base}_enhanced{ext or '.mp4'}"

        self.queue.append({
            "path": path,
            "output": output_path,
            "start_edit": start_edit,
            "end_edit": end_edit,
        })

    def _remove_selected_queue_rows(self):
        if self._batch_running:
            QMessageBox.information(self, "Batch running", "Stop or wait for the current batch to finish first.")
            return
        rows = sorted({idx.row() for idx in self.queue_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.queue_table.removeRow(row)
            del self.queue[row]

    def _clear_queue(self):
        if self._batch_running:
            QMessageBox.information(self, "Batch running", "Stop or wait for the current batch to finish first.")
            return
        self.queue_table.setRowCount(0)
        self.queue = []

    def _set_row_status(self, row, text):
        item = self.queue_table.item(row, COL_STATUS)
        if item is not None:
            item.setText(text)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def _on_schedule_toggled(self, checked):
        self.schedule_datetime.setEnabled(checked)
        if not checked:
            self._schedule_armed = False
            self.schedule_status_label.setText("")

    def _check_schedule(self):
        if not self._schedule_armed:
            return
        target = self.schedule_datetime.dateTime()
        now = QDateTime.currentDateTime()
        remaining = now.secsTo(target)
        if remaining <= 0:
            self._schedule_armed = False
            self.schedule_status_label.setText("Scheduled time reached -- starting now.")
            self._begin_batch()
        else:
            self.schedule_status_label.setText(f"Starts in {_format_eta(remaining)} (at {target.toString('yyyy-MM-dd HH:mm:ss')})")

    # ------------------------------------------------------------------
    # Preset handling
    # ------------------------------------------------------------------
    def _on_preset_changed(self, name):
        preset = PRESETS.get(name)
        if not preset:
            return
        self.preset_desc.setText(preset["desc"])
        idx = self.model_combo.findText(preset["model"])
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.scale_spin.setValue(preset["outscale"])
        self.denoise_spin.setValue(preset["denoise_strength"])
        self.face_check.setChecked(preset["face_enhance"])

    # ------------------------------------------------------------------
    # Run / batch orchestration
    # ------------------------------------------------------------------
    def _on_start_clicked(self):
        if not self.queue:
            QMessageBox.warning(self, "No files queued", "Add at least one video file to the batch queue first.")
            return

        if self.schedule_check.isChecked():
            target = self.schedule_datetime.dateTime()
            if QDateTime.currentDateTime().secsTo(target) <= 0:
                QMessageBox.warning(self, "Invalid schedule", "Scheduled start time must be in the future.")
                return
            self._schedule_armed = True
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self.schedule_status_label.setText("Schedule armed -- waiting...")
            self.log_view.appendPlainText(
                f"Schedule armed: batch will start at {target.toString('yyyy-MM-dd HH:mm:ss')}."
            )
            return

        self._begin_batch()

    def _begin_batch(self):
        for row in range(self.queue_table.rowCount()):
            self._set_row_status(row, "Queued")
        self.log_view.clear()
        self.batch_progress_label.setText(f"Starting batch: 0/{len(self.queue)} files done.")
        self.eta_label.setText("Starting...")
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._batch_running = True
        self.queue_index = -1
        self._start_next_in_queue()

    def _start_next_in_queue(self):
        self.queue_index += 1
        if self.queue_index >= len(self.queue):
            self._finish_batch()
            return

        item = self.queue[self.queue_index]
        model_name = self.model_combo.currentText()
        outscale = self.scale_spin.value()
        denoise_strength = self.denoise_spin.value()
        face_enhance = self.face_check.isChecked()
        tile = self.tile_spin.value()
        trim_start = self._time_to_seconds(item["start_edit"].time())
        trim_end = self._time_to_seconds(item["end_edit"].time())

        self._set_row_status(self.queue_index, "Processing...")
        self.batch_progress_label.setText(
            f"Processing file {self.queue_index + 1}/{len(self.queue)}: {os.path.basename(item['path'])}"
        )
        self.progress_bar.setValue(0)

        self.worker = VideoUpscaleWorker(
            item["path"], item["output"], model_name, outscale,
            denoise_strength, face_enhance, tile,
            trim_start=trim_start, trim_end=trim_end,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished_ok.connect(self._on_item_finished_ok)
        self.worker.finished_err.connect(self._on_item_finished_err)
        self.worker.start()

    def _finish_batch(self):
        self._batch_running = False
        self.queue_index = -1
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.eta_label.setText("Batch complete.")
        self.batch_progress_label.setText(f"Batch complete: {len(self.queue)}/{len(self.queue)} files done.")
        self.log_view.appendPlainText("All queued files processed.")
        QMessageBox.information(self, "Batch complete", f"Finished processing {len(self.queue)} file(s).")

    def _cancel(self):
        if self._schedule_armed:
            self._schedule_armed = False
            self.schedule_status_label.setText("Schedule cancelled.")
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            return
        if self.worker is not None:
            self.worker.request_stop()
        self._batch_running = False
        self.cancel_btn.setEnabled(False)

    @staticmethod
    def _time_to_seconds(qtime):
        return qtime.hour() * 3600 + qtime.minute() * 60 + qtime.second()

    def _on_progress(self, frame_idx, total_frames, fps_proc, eta_str):
        if total_frames:
            self.progress_bar.setMaximum(total_frames)
            self.progress_bar.setValue(frame_idx)
            self.eta_label.setText(f"{fps_proc:.2f} fps \u00b7 ETA (this file) {eta_str}")

    def _on_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_item_finished_ok(self, output_path):
        self._set_row_status(self.queue_index, "Done")
        self.log_view.appendPlainText(f"Done. Saved to: {output_path}")
        if self._batch_running:
            self._start_next_in_queue()
        else:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def _on_item_finished_err(self, err):
        self._set_row_status(self.queue_index, "Failed")
        self.log_view.appendPlainText(f"ERROR: {err}")
        if self._batch_running:
            self._start_next_in_queue()
        else:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
