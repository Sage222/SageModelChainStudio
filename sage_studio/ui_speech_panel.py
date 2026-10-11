"""Sage Model Chain Studio -- SpeechText (Parakeet / faster-whisper speech-to-text)"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QFileDialog, QGroupBox, QFormLayout, QProgressBar,
    QPlainTextEdit, QSpinBox, QMessageBox,
)

from .constants import WHISPER_MODEL_DIR, SPEECH_PIP_HINT
from .utils import _cuda_available

MODELS = {
    "Parakeet-TDT-0.6B-v3 (Fast, English only)": {"engine": "parakeet", "model_id": "nvidia/parakeet-tdt-0.6b-v3"},
    "faster-whisper large-v3 (Best accuracy, 99 languages)": {"engine": "whisper", "model_size": "large-v3"},
    "faster-whisper medium": {"engine": "whisper", "model_size": "medium"},
    "faster-whisper small": {"engine": "whisper", "model_size": "small"},
    "faster-whisper base": {"engine": "whisper", "model_size": "base"},
    "faster-whisper tiny (Fastest)": {"engine": "whisper", "model_size": "tiny"},
}


def _format_eta(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _CancelledError(Exception):
    pass


class SpeechToTextWorker(QThread):
    progress = pyqtSignal(int, int, str)  # chunks_done, total_chunks, eta_str
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    finished_err = pyqtSignal(str)

    def __init__(self, input_path, engine, model_ref, chunk_seconds):
        super().__init__()
        self.input_path = input_path
        self.engine = engine          # "parakeet" or "whisper"
        self.model_ref = model_ref    # HF model id, or whisper model size
        self.chunk_seconds = chunk_seconds
        self._stop_requested = False
        self._tmp_wav = None

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._run()
        except _CancelledError:
            self.log.emit("Cancelled by user.")
            self.finished_err.emit("Cancelled by user.")
        except Exception as e:
            self.finished_err.emit(f"{type(e).__name__}: {e}")
        finally:
            if self._tmp_wav and os.path.exists(self._tmp_wav):
                try:
                    os.remove(self._tmp_wav)
                except OSError:
                    pass

    def _check_cancel(self):
        if self._stop_requested:
            raise _CancelledError()

    def _normalize_to_wav(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg is not on your PATH. It's required to decode the input file "
                "into 16kHz mono audio for transcription.\n\n"
                "https://ffmpeg.org/download.html"
            )
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._tmp_wav = tmp_path
        cmd = [
            ffmpeg, "-y", "-i", self.input_path,
            "-ac", "1", "-ar", "16000", "-vn", tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode input file:\n{result.stderr[-800:]}")
        return tmp_path

    def _build_engine(self):
        if self.engine == "parakeet":
            try:
                from transformers import pipeline
            except ImportError:
                raise RuntimeError("transformers is not installed.\n\n" + SPEECH_PIP_HINT)
            device = 0 if _cuda_available() else -1
            self.log.emit(f"Loading Parakeet model '{self.model_ref}' (first run downloads weights)...")
            pipe = pipeline("automatic-speech-recognition", model=self.model_ref, device=device)
            return ("parakeet", pipe)
        else:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError("faster-whisper is not installed.\n\n" + SPEECH_PIP_HINT)
            device = "cuda" if _cuda_available() else "cpu"
            compute_type = "float16" if _cuda_available() else "int8"
            self.log.emit(f"Loading faster-whisper model '{self.model_ref}' (first run downloads weights)...")
            model = WhisperModel(
                self.model_ref, device=device, compute_type=compute_type,
                download_root=WHISPER_MODEL_DIR,
            )
            return ("whisper", model)

    def _transcribe_chunk(self, kind, engine_obj, chunk_array, sample_rate):
        if kind == "parakeet":
            result = engine_obj({"array": chunk_array, "sampling_rate": sample_rate})
            return (result.get("text") or "").strip()
        else:
            segments, _info = engine_obj.transcribe(chunk_array, language="en")
            return " ".join(seg.text.strip() for seg in segments).strip()

    def _run(self):
        try:
            import soundfile as sf
            import numpy as np
        except ImportError:
            raise RuntimeError("soundfile/numpy are not installed.\n\n" + SPEECH_PIP_HINT)

        if _cuda_available():
            self.log.emit("Device: GPU (CUDA detected).")
        else:
            self.log.emit("Device: CPU (no CUDA detected -- this will be slow).")

        self.log.emit("Decoding input audio to 16kHz mono WAV...")
        wav_path = self._normalize_to_wav()
        self._check_cancel()

        audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        total_samples = len(audio)
        if total_samples == 0:
            raise RuntimeError("No audio data found (empty or unreadable file).")

        chunk_samples = max(1, int(self.chunk_seconds * sample_rate))
        total_chunks = max(1, (total_samples + chunk_samples - 1) // chunk_samples)

        kind, engine_obj = self._build_engine()
        self._check_cancel()

        import time
        start_time = time.time()
        pieces = []
        for i in range(total_chunks):
            self._check_cancel()
            start = i * chunk_samples
            end = min(total_samples, start + chunk_samples)
            chunk = audio[start:end]

            text = self._transcribe_chunk(kind, engine_obj, chunk, sample_rate)
            if text:
                pieces.append(text)

            done = i + 1
            elapsed = time.time() - start_time
            rate = elapsed / done if done else 0.0
            remaining = (total_chunks - done) * rate
            eta_str = _format_eta(remaining) if done < total_chunks else "00:00:00"
            self.progress.emit(done, total_chunks, eta_str)
            self.log.emit(f"Transcribed chunk {done}/{total_chunks}.")

        transcript = " ".join(pieces).strip()
        if not transcript:
            raise RuntimeError("Transcription produced no text (silent or unreadable audio?).")

        self.finished_ok.emit(transcript)


class MicrophoneRecorder(QThread):
    recorded = pyqtSignal(str)
    failed = pyqtSignal(str)
    elapsed = pyqtSignal(int)

    def __init__(self, device_index, output_path):
        super().__init__()
        self.device_index = device_index
        self.output_path = output_path
        self._stop_event = threading.Event()

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        try:
            import sounddevice as sd
            import soundfile as sf
            info = sd.query_devices(self.device_index, 'input')
            if int(info['max_input_channels']) < 1:
                raise RuntimeError('Selected device has no microphone input channels.')
            rate = int(round(info['default_samplerate']))
            if rate < 8000:
                raise RuntimeError('Microphone reported an invalid sample rate.')
            frames_written = 0
            started = time.monotonic()
            shown_second = -1
            with sf.SoundFile(self.output_path, mode='x', samplerate=rate,
                              channels=1, subtype='PCM_16') as wav:
                with sd.InputStream(device=self.device_index, samplerate=rate,
                                    channels=1, dtype='float32', blocksize=1024) as stream:
                    while not self._stop_event.is_set():
                        data, overflowed = stream.read(1024)
                        wav.write(data)
                        frames_written += len(data)
                        seconds = int(time.monotonic() - started)
                        if seconds != shown_second:
                            self.elapsed.emit(seconds)
                            shown_second = seconds
                        if overflowed:
                            pass  # Keep recording; a brief input overflow is recoverable.
            if frames_written == 0:
                raise RuntimeError('No microphone audio was recorded.')
            self.recorded.emit(self.output_path)
        except Exception as exc:
            try:
                os.remove(self.output_path)
            except OSError:
                pass
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class SpeechTextPanel(QWidget):
    def __init__(self, persistence, set_chain_input_callback=None):
        super().__init__()
        self.persistence = persistence
        self.set_chain_input_callback = set_chain_input_callback
        self.worker = None
        self.last_transcript = ""
        self.recorder = None
        self._closing = False

        layout = QVBoxLayout(self)

        title = QLabel("\U0001F3A4 SpeechText -- Speech-to-Text (Parakeet / faster-whisper, local & free)")
        title.setObjectName("sectionHint")
        layout.addWidget(title)

        gpu_txt = ("GPU (CUDA) available -- will run faster" if _cuda_available()
                   else "CPU only -- this will be slow, a GPU is strongly recommended")
        gpu_label = QLabel(f"Device: {gpu_txt}")
        gpu_label.setObjectName("sectionHint")
        layout.addWidget(gpu_label)

        file_group = QGroupBox("Input")
        file_form = QFormLayout(file_group)
        in_row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        in_browse = QPushButton("Browse...")
        in_browse.clicked.connect(self._pick_input)
        self.browse_btn = in_browse
        in_row.addWidget(self.input_edit, 1)
        in_row.addWidget(in_browse)
        file_form.addRow("Audio/Video File:", in_row)
        mic_row = QHBoxLayout()
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(220)
        self.refresh_mics_btn = QPushButton('Refresh mics')
        self.refresh_mics_btn.clicked.connect(self._refresh_microphones)
        mic_row.addWidget(self.mic_combo, 1)
        mic_row.addWidget(self.refresh_mics_btn)
        file_form.addRow('Microphone:', mic_row)
        record_row = QHBoxLayout()
        self.record_btn = QPushButton('● Record from mic')
        self.record_btn.clicked.connect(self._start_recording)
        self.stop_record_btn = QPushButton('■ Stop and transcribe')
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.clicked.connect(self._stop_recording)
        self.recording_label = QLabel('')
        record_row.addWidget(self.record_btn)
        record_row.addWidget(self.stop_record_btn)
        record_row.addWidget(self.recording_label)
        record_row.addStretch(1)
        file_form.addRow('Record:', record_row)
        layout.addWidget(file_group)

        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODELS.keys()))
        model_layout.addWidget(self.model_combo)

        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Chunk Length:"))
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(10, 300)
        self.chunk_spin.setValue(60)
        self.chunk_spin.setSuffix(" sec")
        self.chunk_spin.setToolTip(
            "Audio is processed in fixed-length chunks rather than all at once.\n"
            "This keeps VRAM use bounded regardless of file length -- important for\n"
            "Parakeet specifically, whose memory use scales with clip duration.\n"
            "Smaller chunks = lower peak VRAM, more overhead. 60s is a good default."
        )
        chunk_row.addWidget(self.chunk_spin)
        chunk_row.addStretch(1)
        model_layout.addLayout(chunk_row)
        layout.addWidget(model_group)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25B6 Start Transcription")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("\u23F9 Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.eta_label = QLabel("")
        self.eta_label.setObjectName("sectionHint")
        layout.addWidget(self.eta_label)

        transcript_group = QGroupBox("Transcript")
        transcript_layout = QVBoxLayout(transcript_group)
        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setReadOnly(True)
        transcript_layout.addWidget(self.transcript_edit, 1)

        transcript_btn_row = QHBoxLayout()
        save_btn = QPushButton("Save as .txt")
        save_btn.clicked.connect(self._save_transcript)
        transcript_btn_row.addWidget(save_btn)
        send_btn = QPushButton("Send to Chat Input")
        send_btn.clicked.connect(self._send_to_chat)
        transcript_btn_row.addWidget(send_btn)
        transcript_layout.addLayout(transcript_btn_row)

        layout.addWidget(transcript_group, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setMaximumHeight(140)
        layout.addWidget(self.log_view)
        self._refresh_microphones()

    def _refresh_microphones(self):
        previous = self.mic_combo.currentData() if self.mic_combo.count() else None
        self.mic_combo.clear()
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            for index, device in enumerate(devices):
                if int(device['max_input_channels']) > 0:
                    self.mic_combo.addItem(f"{index}: {device['name']}", index)
            selected = self.mic_combo.findData(previous)
            if selected < 0:
                selected = self.mic_combo.findData(default_input)
            if selected >= 0:
                self.mic_combo.setCurrentIndex(selected)
            self.record_btn.setEnabled(self.mic_combo.count() > 0 and
                                       not (self.worker and self.worker.isRunning()))
            if not self.mic_combo.count():
                self.mic_combo.addItem('No microphone input found', None)
                self.record_btn.setEnabled(False)
        except Exception as exc:
            self.mic_combo.addItem(f'Microphone unavailable: {exc}', None)
            self.record_btn.setEnabled(False)

    def _start_recording(self):
        if self.recorder and self.recorder.isRunning():
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, 'Busy', 'Finish or cancel transcription first.')
            return
        device_index = self.mic_combo.currentData()
        if device_index is None:
            QMessageBox.warning(self, 'No microphone', 'Choose an input device and click Refresh mics.')
            return
        try:
            from .constants import DATA_DIR
            recordings_dir = os.path.join(DATA_DIR, 'recordings')
            os.makedirs(recordings_dir, exist_ok=True)
            name = 'mic_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.wav'
            output_path = os.path.join(recordings_dir, name)
        except OSError as exc:
            QMessageBox.critical(self, 'Recording failed', str(exc))
            return
        self.recorder = MicrophoneRecorder(device_index, output_path)
        self.recorder.recorded.connect(self._recording_saved)
        self.recorder.failed.connect(self._recording_failed)
        self.recorder.elapsed.connect(self._recording_elapsed)
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)
        self.mic_combo.setEnabled(False)
        self.refresh_mics_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.recording_label.setText('Recording... 00:00:00')
        self.recorder.start()

    def _stop_recording(self):
        if self.recorder and self.recorder.isRunning():
            self.stop_record_btn.setEnabled(False)
            self.recording_label.setText('Saving recording...')
            self.recorder.request_stop()

    def _recording_elapsed(self, seconds):
        self.recording_label.setText('Recording... ' + _format_eta(seconds))

    def _recording_controls_reset(self):
        self.stop_record_btn.setEnabled(False)
        self.mic_combo.setEnabled(True)
        self.refresh_mics_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        busy = bool(self.worker and self.worker.isRunning())
        self.record_btn.setEnabled(self.mic_combo.currentData() is not None and not busy)
        self.run_btn.setEnabled(not busy)

    def _recording_saved(self, path):
        if self._closing:
            return
        self._recording_controls_reset()
        self.recording_label.setText('Saved recording')
        self.input_edit.setText(path)
        self.log_view.appendPlainText(f'Microphone recording saved: {path}')
        self._start()

    def _recording_failed(self, message):
        if self._closing:
            return
        self._recording_controls_reset()
        self.recording_label.setText('Recording failed')
        self.log_view.appendPlainText(f'Microphone error: {message}')
        QMessageBox.critical(self, 'Microphone recording failed', message)

    def stop_recording_for_close(self):
        self._closing = True
        if self.recorder and self.recorder.isRunning():
            self.recorder.request_stop()
            self.recorder.wait()

    def _pick_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an audio or video file", "",
            "Audio/Video Files (*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.mp4 *.mkv *.mov *.webm);;All Files (*)"
        )
        if path:
            self.input_edit.setText(path)

    def _start(self):
        if self.stop_record_btn.isEnabled() or self.recording_label.text() == "Saving recording...":
            return
        if self.worker and self.worker.isRunning():
            return
        input_path = self.input_edit.text().strip()
        if not input_path or not os.path.exists(input_path):
            QMessageBox.warning(self, "No input file", "Choose a valid audio or video file first.")
            return

        info = MODELS[self.model_combo.currentText()]
        engine = info["engine"]
        model_ref = info.get("model_id") or info.get("model_size")
        chunk_seconds = self.chunk_spin.value()

        self.log_view.clear()
        self.transcript_edit.clear()
        self.progress_bar.setValue(0)
        self.eta_label.setText("Starting...")
        self.run_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.worker = SpeechToTextWorker(input_path, engine, model_ref, chunk_seconds)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished_err.connect(self._on_finished_err)
        self.worker.start()

    def _cancel(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.cancel_btn.setEnabled(False)

    def _on_progress(self, done, total, eta_str):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.eta_label.setText(f"Chunk {done}/{total} \u00b7 ETA {eta_str}")

    def _on_log(self, text):
        self.log_view.appendPlainText(text)

    def _on_finished_ok(self, transcript):
        self.run_btn.setEnabled(True)
        self.record_btn.setEnabled(self.mic_combo.currentData() is not None)
        self.cancel_btn.setEnabled(False)
        self.eta_label.setText("Done.")
        self.last_transcript = transcript
        self.transcript_edit.setPlainText(transcript)
        self.log_view.appendPlainText("Transcription complete.")

    def _on_finished_err(self, err):
        self.run_btn.setEnabled(True)
        self.record_btn.setEnabled(self.mic_combo.currentData() is not None)
        self.cancel_btn.setEnabled(False)
        self.eta_label.setText("Failed." if "Cancelled" not in err else "Cancelled.")
        self.log_view.appendPlainText(f"ERROR: {err}")
        if "Cancelled" not in err:
            QMessageBox.critical(self, "Transcription failed", err)

    def _save_transcript(self):
        if not self.last_transcript:
            QMessageBox.information(self, "No transcript", "Run a transcription first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript as", "transcript.txt", "Text Files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.last_transcript)
            QMessageBox.information(self, "Saved", f"Transcript saved to:\n{path}")
        except OSError as e:
            QMessageBox.critical(self, "Save failed", str(e))

    def _send_to_chat(self):
        if not self.last_transcript:
            QMessageBox.information(self, "No transcript", "Run a transcription first.")
            return
        if self.set_chain_input_callback:
            self.set_chain_input_callback(self.last_transcript)
            QMessageBox.information(self, "Sent", "Transcript sent to the Chat tab's input box.")
        else:
            QMessageBox.information(self, "Not wired up", "This panel isn't connected to the Chat tab's input.")
