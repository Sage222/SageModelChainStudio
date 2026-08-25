"""Sage Model Chain Studio — Ui Run Panel"""

from datetime import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QPlainTextEdit, QTextEdit, QMessageBox,
)
from .models import RunRecord
from .workers import ChatWorker


class RunPanel(QGroupBox):
    def __init__(self, get_steps_callback):
        super().__init__("Run")
        self.get_steps_callback = get_steps_callback
        self.worker = None
        self.on_run_finished = None
        self.on_chain_complete = None
        self.on_auto_tts_toggled = None
        self.local_model_params = {}
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Initial input:"))
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Type the starting prompt for step 1...")
        self.input_edit.setFixedHeight(120)
        self.input_edit.setStyleSheet("font-size: 15px;")
        layout.addWidget(self.input_edit)

        options_row = QHBoxLayout()
        self.retry_check = QCheckBox("Auto-retry on 429")
        self.retry_check.setChecked(True)
        options_row.addWidget(self.retry_check)
        options_row.addStretch(1)
        layout.addLayout(options_row)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6  Run Chain")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self.run_chain)
        run_row.addWidget(self.run_btn, 1)
        self.stop_btn = QPushButton("\u25a0  Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_chain)
        run_row.addWidget(self.stop_btn)
        self.auto_tts_check = QCheckBox("\U0001F50A Auto TTS")
        self.auto_tts_check.setToolTip("When enabled, the final chain output is automatically sent to Kokoro TTS.")
        self.auto_tts_check.toggled.connect(self._on_auto_tts_toggled)
        run_row.addWidget(self.auto_tts_check)
        layout.addLayout(run_row)

        layout.addWidget(QLabel("Output / history log:"))
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setStyleSheet("font-size: 15px;")
        layout.addWidget(self.output_log, 1)
        self._pending_record = None
        self._step_count = 1

    def set_local_model_params(self, params):
        self.local_model_params = params

    def _on_auto_tts_toggled(self, checked):
        if self.on_auto_tts_toggled:
            self.on_auto_tts_toggled(checked)

    def load_history(self, history, last_input=""):
        self.output_log.clear()
        for record in history:
            self._render_record(record)
        if last_input:
            self.input_edit.setPlainText(last_input)
        else:
            self.input_edit.clear()

    def _render_record(self, record):
        single_step = len(record.step_results) <= 1
        self.output_log.append(
            f"<span style='color:#757c8f'>[{record.timestamp}]</span> "
            f"<b><span style='color:#4da6ff'>Input:</span></b> "
            f"<span style='color:#4da6ff'>{record.initial_input}</span>"
        )
        for res in record.step_results:
            idx = res["step_index"]
            label = res.get("display_name") or res.get("model_id")
            if res.get("error"):
                self.output_log.append(f"<b style='color:#f87171'>Step {idx + 1} ({label}) failed:</b> {res['error']}")
            else:
                retries = res.get("retries", 0)
                rn = f" <i style='color:#facc15'>(after {retries} retr{'y' if retries==1 else 'ies'})</i>" if retries else ""
                if not single_step:
                    self.output_log.append(f"<b>\u2500\u2500 Step {idx + 1} \u2014 {label} \u2500\u2500</b>{rn}")
                elif rn:
                    self.output_log.append(rn)
                self.output_log.append(res["output"])
        if record.stopped:
            self.output_log.append("<i style='color:#facc15'>Chain was stopped by the user.</i>")
        self.output_log.append("<hr>")

    def run_chain(self):
        steps = self.get_steps_callback()
        if not steps:
            QMessageBox.information(self, "No steps", "Add at least one model to the chain first.")
            return
        initial_input = self.input_edit.toPlainText().strip()
        if not initial_input:
            QMessageBox.information(self, "No input", "Type an initial prompt first.")
            return
        self._step_count = len(steps)
        self._pending_record = RunRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            initial_input=initial_input, step_results=[],
        )
        self.output_log.append(f"<span style='color:#757c8f'>[{self._pending_record.timestamp}]</span> <b>Input:</b> {initial_input}")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("\u25a0  Stop")
        self.worker = ChatWorker(steps, initial_input, self.retry_check.isChecked(), self.local_model_params)
        self.worker.step_done.connect(self._on_step_done)
        self.worker.step_retry.connect(self._on_step_retry)
        self.worker.step_error.connect(self._on_step_error)
        self.worker.step_stopped.connect(self._on_step_stopped)
        self.worker.step_loading.connect(self._on_step_loading)
        self.worker.step_warning.connect(self._on_step_warning)
        self.worker.chain_finished.connect(self._on_finished)
        self.worker.start()

    def stop_chain(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")

    def _on_step_loading(self, idx, label):
        if getattr(self, '_step_count', 1) <= 1:
            return
        self.output_log.append(f"<i style='color:#60a5fa'>Step {idx + 1}: loading local model '{label}' into VRAM...</i>")

    def _on_step_warning(self, idx, msg):
        safe_msg = msg.replace("\n", "<br>")
        self.output_log.append(f"<b style='color:#facc15'>\u26a0 Step {idx + 1} warning:</b><br><span style='color:#facc15'>{safe_msg}</span>")

    def _on_step_retry(self, idx, attempt, wait, reason):
        self.output_log.append(f"<i style='color:#facc15'>Step {idx + 1}: {reason}, retry {attempt} in {wait:.1f}s...</i>")

    def _on_step_done(self, idx, output):
        if getattr(self, '_step_count', 1) > 1:
            self.output_log.append(f"<b>\u2500\u2500 Step {idx + 1} output \u2500\u2500</b>")
        self.output_log.append(output)
        self.output_log.append("")

    def _reset_run_controls(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("\u25b6  Run Chain")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("\u25a0  Stop")

    def _finalize_record(self, stopped=False):
        if self._pending_record is not None and self.worker is not None:
            self._pending_record.step_results = self.worker.step_results
            self._pending_record.stopped = stopped
            if self.on_run_finished:
                self.on_run_finished(self._pending_record)
            self._pending_record = None

    def _on_step_error(self, idx, err):
        self.output_log.append(f"<b style='color:#f87171'>Step {idx + 1} failed:</b> {err}")
        self._reset_run_controls()
        self._finalize_record(stopped=False)

    def _on_step_stopped(self, idx):
        self.output_log.append(f"<i style='color:#facc15'>Stopped before/at step {idx + 1} by user request.</i>")
        self.output_log.append("<hr>")
        self._reset_run_controls()
        self._finalize_record(stopped=True)

    def _on_finished(self):
        if getattr(self, '_step_count', 1) > 1:
            self.output_log.append("<b style='color:#4ade80'>Chain complete.</b>")
        self.output_log.append("<hr>")
        final_text = ""
        if self.worker is not None and self.worker.step_results:
            final_text = self.worker.step_results[-1].get("output") or ""
        self._reset_run_controls()
        self._finalize_record(stopped=False)
        if final_text and self.on_chain_complete:
            self.on_chain_complete(final_text)


