"""Sage Model Chain Studio -- MusicStem (Demucs stem separation)"""
import os
import time

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QFileDialog, QGroupBox, QFormLayout, QProgressBar,
    QPlainTextEdit, QCheckBox, QMessageBox,
)

from .constants import DEMUCS_MODEL_DIR, DEMUCS_PIP_HINT
from .utils import _cuda_available

MODELS = {
    "htdemucs_ft (High Quality, slower)": {
        "model": "htdemucs_ft",
        "desc": ("Fine-tuned 4-model ensemble. Best separation quality of the free Demucs "
                 "models, at roughly 4x the compute cost of htdemucs (it runs 4 sub-models "
                 "and averages them)."),
    },
    "htdemucs (Fast)": {
        "model": "htdemucs",
        "desc": "Single-model, hybrid transformer architecture. Good quality, noticeably faster than htdemucs_ft.",
    },
    "htdemucs_6s (6 stems)": {
        "model": "htdemucs_6s",
        "desc": "Like htdemucs, but also splits out Piano and Guitar as separate stems (6 total instead of 4).",
    },
}

STEM_NAMES_4 = ["vocals", "drums", "bass", "other"]
STEM_NAMES_6 = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def _format_eta(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _CancelledError(Exception):
    pass


class StemSeparationWorker(QThread):
    progress = pyqtSignal(float, str)   # percent (0-100), eta_str
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(list)       # list of saved file paths
    finished_err = pyqtSignal(str)

    def __init__(self, input_path, output_dir, model_name, stems_to_save, two_stem_vocals_only):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.stems_to_save = stems_to_save
        self.two_stem_vocals_only = two_stem_vocals_only
        self._stop_requested = False
        self._start_time = None

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run()
        except _CancelledError:
            self.log.emit("Cancelled -- stopping after the current chunk.")
            self.finished_err.emit("Cancelled by user.")
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")

    def _progress_callback(self, info):
        if self._stop_requested:
            raise _CancelledError()
        models = max(1, info.get("models", 1))
        model_idx = info.get("model_idx_in_bag", 0)
        offset = info.get("audio_length", 0) and info.get("segment_offset", 0)
        audio_length = info.get("audio_length", 0)
        frac = (model_idx + (offset / audio_length if audio_length else 0.0)) / models
        percent = max(0.0, min(100.0, frac * 100.0))

        elapsed = time.time() - self._start_time if self._start_time else 0.0
        eta_str = "--:--:--"
        if percent > 0.5 and elapsed > 0:
            total_est = elapsed / (percent / 100.0)
            eta_str = _format_eta(total_est - elapsed)
        self.progress.emit(percent, eta_str)

    def _run(self):
        try:
            import demucs.api
        except ImportError:
            raise RuntimeError("demucs is not installed.\n\n" + DEMUCS_PIP_HINT)

        device = "cuda" if _cuda_available() else "cpu"
        self.log.emit(f"Loading Demucs model '{self.model_name}' on device: {device}...")
        self.log.emit("(first run downloads model weights, be patient)")

        try:
            separator = demucs.api.Separator(
                model=self.model_name,
                device=device,
                progress=False,
                callback=self._progress_callback,
            )
        except Exception as e:
            raise RuntimeError(f"Could not load Demucs model '{self.model_name}': {e}")

        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.input_path))[0]

        self._start_time = time.time()
        self.log.emit(f"Separating '{os.path.basename(self.input_path)}'...")
        try:
            _origin, separated = separator.separate_audio_file(self.input_path)
        except _CancelledError:
            raise
        except Exception as e:
            raise RuntimeError(f"Separation failed: {e}")

        if self._stop_requested:
            raise _CancelledError()

        saved_paths = []
        stems_to_write = separated.items()
        if self.two_stem_vocals_only:
            stems_to_write = [(k, v) for k, v in separated.items() if k == "vocals"]
        elif self.stems_to_save:
            stems_to_write = [(k, v) for k, v in separated.items() if k in self.stems_to_save]

        for stem_name, source in stems_to_write:
            out_path = os.path.join(self.output_dir, f"{base_name}_{stem_name}.wav")
            demucs.api.save_audio(source, out_path, samplerate=separator.samplerate)
            saved_paths.append(out_path)
            self.log.emit(f"Saved: {out_path}")

        if not saved_paths:
            raise RuntimeError("No stems were saved (check your stem selection).")

        self.progress.emit(100.0, "00:00:00")
        self.finished_ok.emit(saved_paths)


class MusicStemPanel(QWidget):
    def __init__(self, persistence):
        super().__init__()
        self.persistence = persistence
        self.worker = None

        layout = QVBoxLayout(self)

        title = QLabel("\U0001F3B5 MusicStem -- Stem Separation (Demucs, local & free)")
        title.setObjectName("sectionHint")
        layout.addWidget(title)

        gpu_txt = ("GPU (CUDA) available -- will run faster" if _cuda_available()
                   else "CPU only -- this will be slow, a GPU is strongly recommended")
        gpu_label = QLabel(f"Device: {gpu_txt}")
        gpu_label.setObjectName("sectionHint")
        layout.addWidget(gpu_label)

        file_group = QGroupBox("Input / Output")
        file_form = QFormLayout(file_group)

        in_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        in_browse = QPushButton("Browse...")
        in_browse.clicked.connect(self._pick_input)
        in_row.addWidget(self.input_edit, 1)
        in_row.addWidget(in_browse)
        file_form.addRow("Input Audio:", in_row)

        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._pick_output_dir)
        out_row.addWidget(self.output_dir_edit, 1)
        out_row.addWidget(out_browse)
        file_form.addRow("Output Folder:", out_row)

        layout.addWidget(file_group)

        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODELS.keys()))
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        self.model_desc = QLabel()
        self.model_desc.setWordWrap(True)
        self.model_desc.setObjectName("sectionHint")
        model_layout.addWidget(self.model_desc)
        layout.addWidget(model_group)

        stems_group = QGroupBox("Stems to Save")
        stems_layout = QVBoxLayout(stems_group)

        self.vocals_only_check = QCheckBox("Vocals only (2-stem mode -- just extract vocals, skip the rest)")
        self.vocals_only_check.toggled.connect(self._on_vocals_only_toggled)
        stems_layout.addWidget(self.vocals_only_check)

        self.stem_checks = {}
        stem_row = QHBoxLayout()
        for stem_name in STEM_NAMES_6:
            cb = QCheckBox(stem_name.capitalize())
            cb.setChecked(stem_name in STEM_NAMES_4)
            self.stem_checks[stem_name] = cb
            stem_row.addWidget(cb)
        stems_layout.addLayout(stem_row)

        stems_hint = QLabel(
            "Guitar/Piano checkboxes only apply if you pick the 6-stem model above. "
            "Unchecked stems are still separated internally but not written to disk."
        )
        stems_hint.setWordWrap(True)
        stems_hint.setObjectName("sectionHint")
        stems_layout.addWidget(stems_hint)

        layout.addWidget(stems_group)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25B6 Start Separation")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("\u23F9 Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        self.eta_label.setObjectName("sectionHint")
        layout.addWidget(self.eta_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        layout.addWidget(self.log_view, 1)

        self._on_model_changed(self.model_combo.currentText())

    def _pick_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an audio file", "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg *.aac);;All Files (*)"
        )
        if not path:
            return
        self.input_edit.setText(path)
        if not self.output_dir_edit.text():
            self.output_dir_edit.setText(os.path.dirname(path))

    def _pick_output_dir(self):
        default = self.output_dir_edit.text() or ""
        path = QFileDialog.getExistingDirectory(self, "Select output folder", default)
        if path:
            self.output_dir_edit.setText(path)

    def _on_model_changed(self, name):
        info = MODELS.get(name)
        if not info:
            return
        self.model_desc.setText(info["desc"])
        is_6s = info["model"] == "htdemucs_6s"
        for stem_name in ("guitar", "piano"):
            self.stem_checks[stem_name].setEnabled(is_6s)

    def _on_vocals_only_toggled(self, checked):
        for stem_name, cb in self.stem_checks.items():
            cb.setEnabled(not checked and (stem_name in STEM_NAMES_4 or self._current_is_6s()))

    def _current_is_6s(self):
        info = MODELS.get(self.model_combo.currentText())
        return bool(info and info["model"] == "htdemucs_6s")

    def _start(self):
        input_path = self.input_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not input_path or not os.path.exists(input_path):
            QMessageBox.warning(self, "No input audio", "Choose a valid input audio file first.")
            return
        if not output_dir:
            QMessageBox.warning(self, "No output folder", "Choose an output folder first.")
            return

        model_info = MODELS[self.model_combo.currentText()]
        model_name = model_info["model"]
        vocals_only = self.vocals_only_check.isChecked()
        selected_stems = [name for name, cb in self.stem_checks.items() if cb.isChecked() and cb.isEnabled()]

        if not vocals_only and not selected_stems:
            QMessageBox.warning(self, "No stems selected", "Check at least one stem to save, or enable Vocals Only.")
            return

        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.eta_label.setText("Starting...")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = StemSeparationWorker(
            input_path, output_dir, model_name, selected_stems, vocals_only,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()

    def _cancel(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, percent, eta_str):
        self.progress_bar.setValue(int(percent))
        self.eta_label.setText(f"{percent:.1f}% \u00b7 ETA {eta_str}")

    def _on_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_finished_ok(self, saved_paths):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.eta_label.setText("Done.")
        self.log_view.appendPlainText(f"Done. Saved {len(saved_paths)} stem file(s).")
        QMessageBox.information(
            self, "Done",
            "Saved stems:\n" + "\n".join(saved_paths)
        )

    def _on_finished_err(self, err):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.eta_label.setText("Failed." if "Cancelled" not in err else "Cancelled.")
        self.log_view.appendPlainText(f"ERROR: {err}")
        if "Cancelled" not in err:
            QMessageBox.critical(self, "Separation failed", err)
