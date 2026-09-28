import time
"""Sage Model Chain Studio — Ui Run Panel"""

from datetime import datetime
from PyQt6.QtCore import Qt, QTimer, QEvent
from PyQt6.QtGui import QTextCursor, QTextOption
from PyQt6.QtWidgets import (
    QSpinBox,
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QPlainTextEdit, QTextEdit, QMessageBox,
)
from .models import RunRecord
from .workers import ChatWorker
from .utils import get_gpu_memory


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
        self.input_edit.installEventFilter(self)

        self.context_label = QLabel("")
        self.context_label.setObjectName("sectionHint")
        layout.addWidget(self.context_label)
        self.input_edit.textChanged.connect(self._update_context_meter)

        options_row = QHBoxLayout()
        self.retry_check = QCheckBox("Auto-retry on 429")
        self.retry_check.setChecked(True)
        options_row.addWidget(self.retry_check)
        self.persistent_check = QCheckBox("Persistent Session")
        self.persistent_check.setChecked(True)
        self.persistent_check.setToolTip(
            "When enabled, each run automatically prepends a rolling transcript\n"
            "of prior turns in this chat, so persona/rules prompt templates (like\n"
            "a Dungeon Master) keep track of what already happened."
        )
        options_row.addWidget(self.persistent_check)
        options_row.addWidget(QLabel("Budget:"))
        self.context_budget_spin = QSpinBox()
        self.context_budget_spin.setRange(200, 100000)
        self.context_budget_spin.setValue(12000)
        self.context_budget_spin.setSuffix(" tok")
        self.context_budget_spin.setEnabled(True)
        self.context_budget_spin.setToolTip(
            "Approximate token budget for the rolling transcript. Oldest turns\n"
            "are summarized then dropped once this is exceeded."
        )
        options_row.addWidget(self.context_budget_spin)
        self.auto_budget_check = QCheckBox("Auto")
        self.auto_budget_check.setChecked(False)
        self.auto_budget_check.setToolTip(
            "Automatically size the budget from the active local model's Context\n"
            "Length (minus Max Tokens and a safety margin). Uncheck to set a fixed\n"
            "number yourself."
        )
        self.auto_budget_check.toggled.connect(
            lambda checked: self.context_budget_spin.setEnabled(not checked)
        )
        options_row.addWidget(self.auto_budget_check)
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
        self.output_log.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.output_log.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.output_log.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.output_log, 1)

        self.speed_vram_label = QLabel("")
        self.speed_vram_label.setObjectName("sectionHint")
        layout.addWidget(self.speed_vram_label)

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._poll_vram)
        self._vram_timer.start(1000)
        self._last_tps = 0.0
        self._last_tokens = 0
        self._last_update = 0.0
        self._log_at_bottom = True
        self.output_log.verticalScrollBar().valueChanged.connect(self._on_log_scroll)
        self.output_log.textChanged.connect(self._maybe_autoscroll_log)
        self._pending_record = None
        self.history = []
        self._rolling_summary = ""
        self._summarized_upto = 0
        self._pending_raw_input = None
        self._summary_worker = None
        self._step_count = 1
        self.current_model_state = None
        self.current_model_label = ""

    def _update_context_meter(self):
        text = self.input_edit.toPlainText()
        if self.persistent_check.isChecked():
            try:
                built = self._build_persistent_input(text)
                est_tokens = max(0, len(built) // 4)
            except Exception:
                est_tokens = max(0, len(text) // 4)
        else:
            est_tokens = max(0, len(text) // 4)
        max_ctx = None
        try:
            steps = self.get_steps_callback()
        except Exception:
            steps = []
        if steps:
            first = steps[0]
            if first.kind in ("local_gguf", "local_transformers"):
                params = (self.local_model_params or {}).get(first.model_id, {})
                max_ctx = params.get("n_ctx")
        if max_ctx:
            pct = (est_tokens / max_ctx * 100) if max_ctx else 0
            note = " (incl. session history)" if self.persistent_check.isChecked() else ""
            self.context_label.setText(
                f"~{est_tokens:,} tokens{note} (~{pct:.0f}% of {max_ctx:,} local context)"
            )
        else:
            self.context_label.setText(
                f"~{est_tokens:,} tokens (estimate \u2014 remote providers have their own limits)"
            )


    def set_local_model_params(self, params):
        self.local_model_params = params

    def _on_auto_tts_toggled(self, checked):
        if self.on_auto_tts_toggled:
            self.on_auto_tts_toggled(checked)

    def load_history(self, history, last_input=""):
        self.history = list(history)
        self._rolling_summary = ""
        self._summarized_upto = 0
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
                self._append_markdown(res["output"])
        if record.stopped:
            self.output_log.append("<i style='color:#facc15'>Chain was stopped by the user.</i>")



    def _build_persistent_input(self, raw_input):
        budget = self._effective_budget()
        candidate_history = self.history[self._summarized_upto:]
        turns = []
        for record in candidate_history:
            if not record.step_results:
                continue
            final_output = record.step_results[-1].get("output") or ""
            if not final_output:
                continue
            turns.append(f"User: {record.initial_input}\nAssistant: {final_output}")

        summary_est = len(self._rolling_summary) // 4
        est_tokens = (len(raw_input) // 4) + summary_est
        included = []
        for turn in reversed(turns):
            est = len(turn) // 4
            if est_tokens + est > budget:
                break
            included.insert(0, turn)
            est_tokens += est

        transcript = ("\n\n".join(included) + "\n\n") if included else ""
        summary_block = f"Summary of earlier events: {self._rolling_summary}\n\n" if self._rolling_summary else ""
        return f"{summary_block}{transcript}User: {raw_input}\nAssistant:"

    def _turns_pending_summarization(self, raw_input):
        """Returns the list of (record) turns that would be dropped by the
        current budget and haven't been folded into the rolling summary yet."""
        budget = self._effective_budget()
        candidate_history = self.history[self._summarized_upto:]
        turns_with_records = []
        for record in candidate_history:
            if not record.step_results:
                continue
            final_output = record.step_results[-1].get("output") or ""
            if not final_output:
                continue
            turns_with_records.append((record, f"User: {record.initial_input}\nAssistant: {final_output}"))

        summary_est = len(self._rolling_summary) // 4
        est_tokens = (len(raw_input) // 4) + summary_est
        kept_count = 0
        for record, turn in reversed(turns_with_records):
            est = len(turn) // 4
            if est_tokens + est > budget:
                break
            kept_count += 1
            est_tokens += est

        drop_count = len(turns_with_records) - kept_count
        if drop_count <= 0:
            return []
        return turns_with_records[:drop_count]



    def _effective_budget(self):
        if not self.auto_budget_check.isChecked():
            return self.context_budget_spin.value()
        try:
            steps = self.get_steps_callback()
        except Exception:
            steps = []
        if steps and steps[0].kind in ("local_gguf", "local_transformers"):
            params = (self.local_model_params or {}).get(steps[0].model_id, {})
            n_ctx = params.get("n_ctx")
            max_tokens = params.get("max_tokens", 1024)
            if n_ctx:
                budget = n_ctx - max_tokens - 400
                return max(budget, 500)
        return 8000

    def run_chain(self):
        steps = self.get_steps_callback()
        if not steps:
            QMessageBox.information(self, "No steps", "Add at least one model to the chain first.")
            return
        raw_input = self.input_edit.toPlainText().strip()
        if not raw_input:
            QMessageBox.information(self, "No input", "Type an initial prompt first.")
            return
        self.input_edit.clear()

        if self.persistent_check.isChecked():
            pending = self._turns_pending_summarization(raw_input)
            if pending:
                self._pending_raw_input = raw_input
                self._start_summary_then_run(steps, pending)
                return

        self._execute_main_chain(steps, raw_input)
    def _start_summary_then_run(self, steps, pending_turns):
        old_summary = self._rolling_summary
        dropped_text = "\n\n".join(turn for _record, turn in pending_turns)
        summary_prompt = (
            (f"Existing summary of earlier events: {old_summary}\n\n" if old_summary else "")
            + "Summarize the key facts, decisions, state changes, and important details from "
            + "the following conversation turns. Be concise but do not lose important plot "
            + "points, character/inventory/state changes, or established facts. Write it as "
            + "a short narrative summary, not a list:\n\n" + dropped_text
        )
        self.output_log.append(
            "<i style='color:#60a5fa'>Summarizing earlier turns to preserve long-term context...</i>"
        )
        summary_step = steps[0]
        self._summary_worker = ChatWorker(
            [summary_step], summary_prompt, self.retry_check.isChecked(), self.local_model_params
        )
        self._summary_worker.chain_finished.connect(lambda p=pending_turns: self._on_summary_finished(p))
        self._summary_worker.step_error.connect(self._on_summary_error)
        self._summary_worker.start()

    def _on_summary_finished(self, pending_turns):
        if self._summary_worker is not None and self._summary_worker.step_results:
            summary_text = self._summary_worker.step_results[-1].get("output") or ""
            self._rolling_summary = summary_text.strip()
        self._summarized_upto += len(pending_turns)
        self.output_log.append("<i style='color:#4ade80'>Summary updated.</i>")
        raw_input = self._pending_raw_input
        self._pending_raw_input = None
        steps = self.get_steps_callback()
        self._execute_main_chain(steps, raw_input)

    def _on_summary_error(self, idx, err):
        self.output_log.append(
            f"<i style='color:#facc15'>Summarization step failed ({err}); "
            f"continuing without updating the summary.</i>"
        )
        raw_input = self._pending_raw_input
        self._pending_raw_input = None
        steps = self.get_steps_callback()
        self._execute_main_chain(steps, raw_input)

    def _execute_main_chain(self, steps, raw_input):
        initial_input = (
            self._build_persistent_input(raw_input)
            if self.persistent_check.isChecked() else raw_input
        )
        self._step_count = len(steps)
        self._pending_record = RunRecord(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            initial_input=raw_input,
            step_results=[],
        )
        self.output_log.append(
            f"<span style='color:#757c8f'>[{self._pending_record.timestamp}]</span> "
            f"<b><span style='color:#4da6ff'>Input:</span></b> "
            f"<span style='color:#4da6ff'>{raw_input}</span>"
        )
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("\u25a0 Stop")
        self.worker = ChatWorker(steps, initial_input, self.retry_check.isChecked(), self.local_model_params)
        self.worker.step_done.connect(self._on_step_done)
        self.worker.step_retry.connect(self._on_step_retry)
        self.worker.step_error.connect(self._on_step_error)
        self.worker.step_stopped.connect(self._on_step_stopped)
        self.worker.step_loading.connect(self._on_step_loading)
        self.worker.step_warning.connect(self._on_step_warning)
        self.worker.token_progress.connect(self._on_token_progress)
        self.worker.chain_finished.connect(self._on_finished)
        self.worker.start()



    def _poll_vram(self):
        mem = get_gpu_memory()
        if mem is None:
            vram_text = "VRAM: n/a"
        else:
            used, total = mem
            vram_text = f"VRAM: {used/1024:.1f} / {total/1024:.1f} GB"
        speed_text = f"{self._last_tps:.1f} tok/s \u00b7 {self._last_tokens} tokens" if self._last_tokens else ""
        self.speed_vram_label.setText(f"{speed_text}    {vram_text}".strip())

    def _on_token_progress(self, idx, tps, count):
        self.current_model_state = "running"
        try:
            steps = self.get_steps_callback()
            if 0 <= idx < len(steps):
                step = steps[idx]
                self.current_model_label = step.display_name or step.model_id
        except Exception:
            pass
        self._last_tps = tps
        self._last_tokens = count
        self._last_update = time.time()
        self._poll_vram()

    def stop_chain(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")

    def _on_step_loading(self, idx, label):
        self.current_model_state = "loading"
        self.current_model_label = label
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
        self._append_markdown(output)

    def _reset_run_controls(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("\u25b6  Run Chain")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("\u25a0  Stop")

    def _finalize_record(self, stopped=False):
        if self._pending_record is not None and self.worker is not None:
            self._pending_record.step_results = self.worker.step_results
            self._pending_record.stopped = stopped
            self.history.append(self._pending_record)
            if self.on_run_finished:
                self.on_run_finished(self._pending_record)
            self._pending_record = None

    def _on_step_error(self, idx, err):
        self.current_model_state = None
        self.current_model_label = ""
        self.output_log.append(f"<b style='color:#f87171'>Step {idx + 1} failed:</b> {err}")
        self._reset_run_controls()
        self._finalize_record(stopped=False)

    def _on_step_stopped(self, idx):
        self.current_model_state = None
        self.current_model_label = ""
        self.output_log.append(f"<i style='color:#facc15'>Stopped before/at step {idx + 1} by user request.</i>")
        self._reset_run_controls()
        self._finalize_record(stopped=True)

    def _on_finished(self):
        self.current_model_state = None
        self.current_model_label = ""
        if getattr(self, '_step_count', 1) > 1:
            self.output_log.append("<b style='color:#4ade80'>Chain complete.</b>")
        final_text = ""
        if self.worker is not None and self.worker.step_results:
            final_text = self.worker.step_results[-1].get("output") or ""
        self._reset_run_controls()
        self._finalize_record(stopped=False)
        if final_text and self.on_chain_complete:
            self.on_chain_complete(final_text)

    def get_model_status(self):
        """Short footer string describing current local-model activity, or None if idle.
        Only tracks local GGUF/safetensors steps -- remote API steps have no
        'started' signal to hook into yet."""
        if self.current_model_state == "loading":
            return f"Loading {self.current_model_label} into VRAM..."
        if self.current_model_state == "running":
            return f"Running {self.current_model_label}"
        return None

    def _on_log_scroll(self, value):
        sb = self.output_log.verticalScrollBar()
        self._log_at_bottom = value >= sb.maximum() - 4

    def _maybe_autoscroll_log(self):
        if self._log_at_bottom:
            sb = self.output_log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def eventFilter(self, obj, event):
        if obj is self.input_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False  # Shift+Enter -- let it insert a newline
                if self.run_btn.isEnabled():
                    self.run_chain()
                return True  # swallow the Enter so it doesn't also add a newline
        return super().eventFilter(obj, event)

    def _append_markdown(self, text):
        """Appends text to the output log, rendering Markdown (headers, bold,
        bullet lists, etc.) instead of dumping the raw ###/** syntax as text."""
        cursor = self.output_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.output_log.document().isEmpty():
            cursor.insertBlock()
        cursor.insertMarkdown(text or "")
        self.output_log.setTextCursor(cursor)
        self.output_log.ensureCursorVisible()


