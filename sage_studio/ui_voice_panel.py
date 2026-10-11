"""Sage Model Chain Studio — Ui Voice Panel"""

import os, sys, subprocess
from datetime import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QCheckBox, QGroupBox, QFormLayout, QPlainTextEdit, QSlider,
    QMessageBox,
)
from .constants import AUDIO_DIR
from .utils import _cuda_available, clean_text_for_speech
from .workers import VoiceWorker
from .persistence import PersistenceManager


class VoicePanel(QWidget):
    KOKORO_VOICES = [
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
    ]

    def __init__(self, persistence, get_output_callback):
        super().__init__()
        self.persistence = persistence
        self.get_output_callback = get_output_callback
        self.settings = self.persistence.load_voice_settings()
        self.worker = None
        root = QVBoxLayout(self)

        title = QLabel("Voice Output")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        root.addWidget(title)

        gpu_status = "GPU (CUDA) available" if _cuda_available() else "CPU only"
        hint = QLabel(
            f"Turn chain output into speech with Kokoro TTS (82M). "
            f"All audio saved in sage_data/audio/. Model cache saved in sage_data/kokoro_model/. "
            f"Install: pip install kokoro soundfile numpy. <b>{gpu_status}</b>"
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        config = QGroupBox("TTS Settings")
        form = QFormLayout(config)

        self.enabled_check = QCheckBox("Auto-synthesize final chain output")
        self.enabled_check.setChecked(bool(self.settings.get("enabled", False)))
        self.enabled_check.toggled.connect(self._save_settings)
        form.addRow("Auto TTS:", self.enabled_check)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Kokoro (local)")
        form.addRow("Engine:", self.engine_combo)

        self.voice_combo = QComboBox()
        self.voice_combo.addItems(self.KOKORO_VOICES)
        saved_voice = self.settings.get("voice", "af_heart")
        if saved_voice in self.KOKORO_VOICES:
            self.voice_combo.setCurrentText(saved_voice)
        self.voice_combo.currentTextChanged.connect(self._save_settings)
        form.addRow("Voice:", self.voice_combo)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)
        self.speed_slider.setValue(int(float(self.settings.get("speed", 1.0)) * 100))
        self.speed_label = QLabel()
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.speed_slider, 1)
        speed_row.addWidget(self.speed_label)
        form.addRow("Speed:", speed_row)

        gpu_row = QHBoxLayout()
        self.gpu_check = QCheckBox("Use GPU (CUDA)")
        self.gpu_check.setChecked(bool(self.settings.get("use_gpu", True)))
        self.gpu_check.setToolTip("Uses CUDA if available. Falls back to CPU if not.")
        self.gpu_check.toggled.connect(self._save_settings)
        gpu_row.addWidget(self.gpu_check)
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setObjectName("sectionHint")
        gpu_row.addStretch(1)
        gpu_row.addWidget(self.gpu_status_label)
        form.addRow("Device:", gpu_row)

        self.auto_play_check = QCheckBox("Play WAV automatically after synthesis")
        self.auto_play_check.setChecked(bool(self.settings.get("auto_play", True)))
        self.auto_play_check.toggled.connect(self._save_settings)
        form.addRow("Playback:", self.auto_play_check)

        self.clean_check = QCheckBox("Clean text for speech (remove emoji, *actions*, **bold**, [narration])")
        self.clean_check.setChecked(bool(self.settings.get("clean_text", True)))
        self.clean_check.setToolTip(
            "Strips emoticons, Unicode emoji, markdown formatting (*text*, **text**), "
            "and stage directions like [laughs] or (whispers) before synthesis."
        )
        self.clean_check.toggled.connect(self._save_settings)
        form.addRow("Text cleaning:", self.clean_check)
        root.addWidget(config)

        self._update_gpu_status()
        self._on_speed_changed(self.speed_slider.value())

        text_group = QGroupBox("Text to Speak")
        text_layout = QVBoxLayout(text_group)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Paste text here, or click Use Latest Chain Output...")
        text_layout.addWidget(self.text_edit, 1)

        text_btns = QHBoxLayout()
        latest_btn = QPushButton("Use Latest Chain Output")
        latest_btn.clicked.connect(self._use_latest_output)
        text_btns.addWidget(latest_btn)
        clean_btn = QPushButton("Preview Cleaned Text")
        clean_btn.clicked.connect(self._preview_cleaned)
        text_btns.addWidget(clean_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.text_edit.clear)
        text_btns.addWidget(clear_btn)
        text_layout.addLayout(text_btns)
        root.addWidget(text_group, 1)

        action_row = QHBoxLayout()
        self.speak_btn = QPushButton("\U0001F50A Speak / Save WAV")
        self.speak_btn.setObjectName("accent")
        self.speak_btn.clicked.connect(self.synthesize)
        action_row.addWidget(self.speak_btn, 1)
        play_btn = QPushButton("Play Last WAV")
        play_btn.clicked.connect(self.play_last)
        action_row.addWidget(play_btn)
        delete_audio_btn = QPushButton("\U0001F5D1 Delete Audio Files")
        delete_audio_btn.setObjectName("danger")
        delete_audio_btn.clicked.connect(self._delete_all_audio)
        action_row.addWidget(delete_audio_btn)
        open_btn = QPushButton("Open Audio Folder")
        open_btn.clicked.connect(self.open_audio_folder)
        action_row.addWidget(open_btn)
        root.addLayout(action_row)

        self.status = QLabel("Ready.")
        self.status.setObjectName("sectionHint")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    def _update_gpu_status(self):
        if _cuda_available():
            self.gpu_status_label.setText("CUDA detected \u2705")
            self.gpu_status_label.setStyleSheet("color: #4ade80;")
        else:
            self.gpu_status_label.setText("CUDA not detected \u274c")
            self.gpu_status_label.setStyleSheet("color: #f87171;")

    def _on_speed_changed(self, value):
        self.speed_label.setText(f"{value / 100:.2f}x")
        self._save_settings()

    def _save_settings(self, *args):
        self.settings.update({
            "enabled": self.enabled_check.isChecked(), "engine": "Kokoro (local)",
            "voice": self.voice_combo.currentText(), "speed": self.speed_slider.value() / 100,
            "auto_play": self.auto_play_check.isChecked(),
            "use_gpu": self.gpu_check.isChecked(), "clean_text": self.clean_check.isChecked(),
        })
        self.persistence.save_voice_settings(self.settings)

    def _use_latest_output(self):
        text = self.get_output_callback()
        if not text:
            QMessageBox.information(self, "No chain output", "Run a chain first, or paste text into the box.")
            return
        self.text_edit.setPlainText(text)

    def _preview_cleaned(self):
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "No text", "Paste some text first.")
            return
        cleaned = clean_text_for_speech(text)
        self.text_edit.setPlainText(cleaned)
        self.status.setText(f"Cleaned: {len(text)} chars \u2192 {len(cleaned)} chars")

    def speak_text(self, text):
        if not text.strip():
            return
        self.text_edit.setPlainText(text)
        self.synthesize()

    def synthesize(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "No text", "Paste text or use the latest chain output first.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(AUDIO_DIR, f"sage_tts_{timestamp}.wav")
        self.speak_btn.setEnabled(False)
        self.speak_btn.setText("Synthesizing...")
        device_label = "GPU (CUDA)" if (self.gpu_check.isChecked() and _cuda_available()) else "CPU"
        self.status.setText(f"Loading Kokoro on {device_label} and generating audio...")
        self.worker = VoiceWorker(
            text, self.voice_combo.currentText(),
            self.speed_slider.value() / 100, output_path,
            self.gpu_check.isChecked(), self.clean_check.isChecked(),
        )
        self.worker.finished_ok.connect(self._on_voice_ok)
        self.worker.finished_err.connect(self._on_voice_err)
        self.worker.start()

    def _on_voice_ok(self, path):
        self.speak_btn.setEnabled(True)
        self.speak_btn.setText("\U0001F50A Speak / Save WAV")
        self.settings["last_output"] = path
        self.persistence.save_voice_settings(self.settings)
        self.status.setText(f"Saved: {os.path.basename(path)}")
        if self.auto_play_check.isChecked():
            self._play_path(path)

    def _on_voice_err(self, error):
        self.speak_btn.setEnabled(True)
        self.speak_btn.setText("\U0001F50A Speak / Save WAV")
        self.status.setText("Voice generation failed.")
        QMessageBox.critical(self, "Kokoro TTS failed", error)

    def _play_path(self, path):
        if not path or not os.path.exists(path):
            QMessageBox.information(self, "No audio file", "No generated WAV file is available yet.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(self, "Could not play audio", str(e))

    def play_last(self):
        self._play_path(self.settings.get("last_output", ""))

    def open_audio_folder(self):
        try:
            if sys.platform.startswith("win"):
                os.startfile(AUDIO_DIR)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", AUDIO_DIR])
            else:
                subprocess.Popen(["xdg-open", AUDIO_DIR])
        except Exception as e:
            QMessageBox.warning(self, "Could not open folder", str(e))

    def _delete_all_audio(self):
        try:
            wav_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
        except OSError as e:
            QMessageBox.critical(self, "Could not read audio folder", str(e))
            return
        if not wav_files:
            QMessageBox.information(self, "No audio files", "There are no WAV files to delete.")
            return
        confirm = QMessageBox.question(
            self,
            "Delete all audio files",
            f"Delete all {len(wav_files)} WAV file(s)? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        deleted, failed = 0, []
        for fname in wav_files:
            path = os.path.join(AUDIO_DIR, fname)
            try:
                os.remove(path)
                deleted += 1
            except OSError as e:
                failed.append(f"{fname}: {e}")
        last_output = self.settings.get("last_output", "")
        if last_output and not os.path.exists(last_output):
            self.settings["last_output"] = ""
            self.persistence.save_voice_settings(self.settings)
        msg = f"Deleted {deleted} WAV file(s)."
        if failed:
            msg += "\n\nFailed:\n" + "\n".join(failed)
        self.status.setText(f"Deleted {deleted} WAV file(s).")
        QMessageBox.information(self, "Deleted", msg)


