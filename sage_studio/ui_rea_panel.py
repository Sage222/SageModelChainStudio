"""Sage Model Chain Studio -- REA tab (Reverse Engineer Anything).

Wraps the REA command-line tool (https://github.com/morluto/rea, npm package
"rea-agents") so it can be installed, updated and run from inside the app, with
the output sent to the Chat input if wanted.

Place this file in sage_studio/. It is registered as a tab by patch_rea_tab.py.

Notes
-----
* REA's main workflow is "ask an MCP-capable agent to investigate a target".
  This app is not an MCP client, so this tab uses REA's CLI commands instead
  and lets you hand the results to your chain.
* Static JavaScript/.NET inspection reads files without running them. Runtime
  capture features run the target with your user permissions.
"""
import os
import re
import shutil
import subprocess
import sys

from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox,
    QCheckBox, QPlainTextEdit, QFileDialog, QGroupBox, QSpinBox, QMessageBox,
    QApplication, QFormLayout,
)

REPO_URL = "https://github.com/morluto/rea"
DOCS_URL = "https://github.com/morluto/rea/tree/main/docs"
NPM_PACKAGE = "rea-agents"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_IS_WIN = sys.platform.startswith("win")
_NO_WINDOW = 0x08000000 if _IS_WIN else 0

PRESETS = [
    ("JavaScript / Electron app (folder or .asar)", "analyze-javascript-application", True),
    ("Custom command (arguments after 'rea')", None, False),
]


def node_version_ok(text):
    """REA needs Node 22.x (>=22.19), 24.x (>=24.11) or 26+."""
    m = re.match(r"v?(\d+)\.(\d+)", (text or "").strip())
    if not m:
        return False
    major, minor = int(m.group(1)), int(m.group(2))
    return (major == 22 and minor >= 19) or (major == 24 and minor >= 11) or major >= 26


def _run_quiet(cmd, timeout=8):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           creationflags=_NO_WINDOW)
        return (r.stdout or r.stderr or "").strip()
    except Exception:
        return ""


def _which(name):
    return shutil.which(name)


class REAPanel(QWidget):
    def __init__(self, send_to_chat=None):
        super().__init__()
        self.send_to_chat = send_to_chat
        self.proc = None
        self._label = ""

        root = QVBoxLayout(self)

        # ---- status + install ----
        status_box = QGroupBox("REA \u2014 Reverse Engineer Anything")
        sv = QVBoxLayout(status_box)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("sectionHint")
        sv.addWidget(self.status_label)

        row = QHBoxLayout()
        self.check_btn = QPushButton("Check setup")
        self.check_btn.clicked.connect(self.refresh_status)
        row.addWidget(self.check_btn)
        self.install_btn = QPushButton("Install REA (npm global)")
        self.install_btn.clicked.connect(self.install_rea)
        row.addWidget(self.install_btn)
        self.update_btn = QPushButton("Update REA")
        self.update_btn.clicked.connect(self.update_rea)
        row.addWidget(self.update_btn)
        self.setup_btn = QPushButton("Agent setup (opens terminal)")
        self.setup_btn.setToolTip("Runs 'npx rea-agents setup' in its own terminal window. "
                                  "Only needed if you also want REA inside Claude Code, Codex, "
                                  "Cursor, etc.")
        self.setup_btn.clicked.connect(self.open_agent_setup)
        row.addWidget(self.setup_btn)
        row.addStretch(1)
        self.repo_btn = QPushButton("GitHub")
        self.repo_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(REPO_URL)))
        row.addWidget(self.repo_btn)
        self.docs_btn = QPushButton("Docs")
        self.docs_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        row.addWidget(self.docs_btn)
        sv.addLayout(row)
        root.addWidget(status_box)

        # ---- run ----
        run_box = QGroupBox("Analyze")
        form = QFormLayout(run_box)
        self.preset_combo = QComboBox()
        for title, _cmd, _needs in PRESETS:
            self.preset_combo.addItem(title)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("Task:", self.preset_combo)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Absolute path to an extracted app folder or .asar")
        path_row.addWidget(self.path_edit, 1)
        b1 = QPushButton("File...")
        b1.clicked.connect(self._browse_file)
        path_row.addWidget(b1)
        b2 = QPushButton("Folder...")
        b2.clicked.connect(self._browse_folder)
        path_row.addWidget(b2)
        self.path_widget = QWidget()
        self.path_widget.setLayout(path_row)
        path_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Target:", self.path_widget)

        self.args_edit = QLineEdit()
        self.args_edit.setPlaceholderText("e.g.  --help     (runs: rea <these arguments>)")
        form.addRow("Arguments:", self.args_edit)

        self.json_check = QCheckBox("Request JSON output (--json)")
        self.json_check.setChecked(True)
        form.addRow("", self.json_check)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("\u25b6 Run")
        self.run_btn.setObjectName("accent")
        self.run_btn.clicked.connect(self.run_analysis)
        run_row.addWidget(self.run_btn, 1)
        self.help_btn = QPushButton("List commands")
        self.help_btn.clicked.connect(self.show_help)
        run_row.addWidget(self.help_btn)
        self.stop_btn = QPushButton("\u25a0 Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        run_row.addWidget(self.stop_btn)
        form.addRow("", self._wrap(run_row))
        root.addWidget(run_box)

        note = QLabel("Use REA only on software you are authorised to analyse. Static JavaScript/"
                      ".NET inspection reads files without running them; runtime-capture features "
                      "run the target with your user permissions. Analysis stays on this PC; "
                      "anything you send to the Chat tab goes to whichever model you run.")
        note.setWordWrap(True)
        note.setObjectName("sectionHint")
        root.addWidget(note)

        # ---- output ----
        root.addWidget(QLabel("Output"))
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setMaximumBlockCount(50000)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.out.setFont(mono)
        root.addWidget(self.out, 1)

        out_row = QHBoxLayout()
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.clicked.connect(self._copy)
        out_row.addWidget(self.copy_btn)
        self.save_btn = QPushButton("Save...")
        self.save_btn.clicked.connect(self._save)
        out_row.addWidget(self.save_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.out.clear)
        out_row.addWidget(self.clear_btn)
        out_row.addStretch(1)
        out_row.addWidget(QLabel("Send up to"))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(500, 200000)
        self.limit_spin.setSingleStep(500)
        self.limit_spin.setValue(6000)
        self.limit_spin.setSuffix(" chars")
        self.limit_spin.setToolTip("Local models have limited context. Large JSON output is cut "
                                   "to this length before it is placed in the Chat input.")
        out_row.addWidget(self.limit_spin)
        self.send_btn = QPushButton("Send to Chat input")
        self.send_btn.clicked.connect(self._send)
        out_row.addWidget(self.send_btn)
        root.addLayout(out_row)

        self._on_preset_changed()
        self.refresh_status()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _wrap(layout):
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _append(self, text):
        if not text:
            return
        self.out.moveCursor(QTextCursor.MoveOperation.End)
        self.out.insertPlainText(text)
        self.out.moveCursor(QTextCursor.MoveOperation.End)

    def _on_preset_changed(self, *_):
        _title, cmd, needs_path = PRESETS[self.preset_combo.currentIndex()]
        self.path_widget.setEnabled(bool(needs_path))
        self.args_edit.setEnabled(cmd is None)
        self.json_check.setEnabled(cmd is not None)

    def _browse_file(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select target file")
        if p:
            self.path_edit.setText(os.path.normpath(p))

    def _browse_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select target folder")
        if p:
            self.path_edit.setText(os.path.normpath(p))

    # ------------------------------------------------------------- status
    def refresh_status(self):
        node = _run_quiet(["node", "--version"]) if _which("node") else ""
        npm = _run_quiet(["npm", "--version"]) if _which("npm") else ""
        rea = _which("rea")
        lines = []
        if not node:
            lines.append("Node.js: NOT FOUND. Install Node 22.19+, 24.11+ or 26+ from nodejs.org.")
        elif node_version_ok(node):
            lines.append(f"Node.js: {node} (supported).")
        else:
            lines.append(f"Node.js: {node} is NOT supported. REA needs 22.19+, 24.11+ or 26+.")
        lines.append(f"npm: {npm}." if npm else "npm: NOT FOUND (comes with Node.js).")
        if rea:
            lines.append(f"REA CLI: installed ({rea}).")
        else:
            lines.append("REA CLI: not installed. Commands will run through 'npx' (slower first run), "
                         "or click 'Install REA'.")
        self.status_label.setText("\n".join(lines))
        ready = bool(node) and node_version_ok(node)
        self.run_btn.setEnabled(ready and self.proc is None)
        self.help_btn.setEnabled(ready and self.proc is None)
        self.install_btn.setEnabled(bool(npm) and self.proc is None)
        self.update_btn.setEnabled(bool(rea) and self.proc is None)
        return ready

    # ------------------------------------------------------------ running
    def _rea_prefix(self):
        rea = _which("rea")
        if rea:
            return [rea]
        npx = _which("npx") or "npx"
        return [npx, "-y", f"{NPM_PACKAGE}@latest"]

    def _build_args(self):
        _title, cmd, needs_path = PRESETS[self.preset_combo.currentIndex()]
        if cmd is None:
            raw = self.args_edit.text().strip()
            if not raw:
                return None, "Type the arguments to pass to 'rea' (for example: --help)."
            try:
                import shlex
                return shlex.split(raw, posix=not _IS_WIN), ""
            except ValueError as e:
                return None, f"Could not parse the arguments: {e}"
        path = self.path_edit.text().strip().strip('"')
        if needs_path:
            if not path:
                return None, "Choose a target path first."
            if not os.path.exists(path):
                return None, f"Path not found: {path}"
            if not os.path.isabs(path):
                path = os.path.abspath(path)
        args = [cmd, path]
        if self.json_check.isChecked():
            args.append("--json")
        return args, ""

    def run_analysis(self):
        args, err = self._build_args()
        if args is None:
            QMessageBox.information(self, "REA", err)
            return
        self._start(self._rea_prefix() + args, "REA")

    def show_help(self):
        self._start(self._rea_prefix() + ["--help"], "REA --help")

    def install_rea(self):
        npm = _which("npm")
        if not npm:
            QMessageBox.warning(self, "REA", "npm was not found. Install Node.js first.")
            return
        self._start([npm, "install", "--global", NPM_PACKAGE], "npm install --global " + NPM_PACKAGE)

    def update_rea(self):
        rea = _which("rea")
        if not rea:
            QMessageBox.information(self, "REA", "REA is not installed as a global command yet.")
            return
        self._start([rea, "update"], "rea update")

    def open_agent_setup(self):
        cmd = f"npx {NPM_PACKAGE} setup"
        if _IS_WIN:
            try:
                subprocess.Popen(["cmd.exe", "/c", "start", "REA setup", "cmd.exe", "/k", cmd])
                return
            except Exception as e:
                QMessageBox.warning(self, "REA", f"Could not open a terminal: {e}")
                return
        QMessageBox.information(self, "REA agent setup",
                                f"Run this in a terminal, then restart your agent:\n\n{cmd}")

    def _start(self, argv, label):
        if self.proc is not None:
            QMessageBox.information(self, "REA", "A command is already running.")
            return
        self._label = label
        self.out.clear()
        self._append(f"$ {' '.join(argv)}\n\n")
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("NO_COLOR", "1")
        env.insert("FORCE_COLOR", "0")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        exe = argv[0]
        if _IS_WIN and exe.lower().endswith((".cmd", ".bat")):
            line = " ".join('"' + a.replace('"', '\\"') + '"' for a in argv)
            proc.setProgram("cmd.exe")
            proc.setNativeArguments(f'/d /s /c "{line}"')
            proc.start()
        else:
            proc.start(exe, argv[1:])
        self.proc = proc
        self._set_busy(True)

    def _set_busy(self, busy):
        self.stop_btn.setEnabled(busy)
        for b in (self.run_btn, self.help_btn, self.install_btn, self.update_btn, self.check_btn):
            b.setEnabled(not busy)
        if not busy:
            self.refresh_status()

    def _on_output(self):
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append(_ANSI.sub("", data).replace("\r\n", "\n").replace("\r", "\n"))

    def _on_error(self, _err):
        if self.proc is not None and self.proc.state() == QProcess.ProcessState.NotRunning:
            self._append(f"\n[could not start the command: {self.proc.errorString()}]\n")
            self.proc = None
            self._set_busy(False)

    def _on_finished(self, code, _status):
        self._on_output()
        self._append(f"\n[{self._label} finished with exit code {code}]\n")
        self.proc = None
        self._set_busy(False)

    def stop(self):
        if self.proc is None:
            return
        pid = int(self.proc.processId() or 0)
        if _IS_WIN and pid:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, creationflags=_NO_WINDOW)
            except Exception:
                self.proc.kill()
        else:
            self.proc.kill()
        self._append("\n[stopped by user]\n")

    # -------------------------------------------------------------- output
    def _copy(self):
        QApplication.clipboard().setText(self.out.toPlainText())

    def _save(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save output", "rea_output.txt",
                                           "Text / JSON (*.txt *.json);;All files (*)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.out.toPlainText())

    def _send(self):
        text = self.out.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "REA", "There is no output to send yet.")
            return
        if self.send_to_chat is None:
            QMessageBox.information(self, "REA", "This tab is not connected to the Chat input.")
            return
        limit = int(self.limit_spin.value())
        cut = len(text) > limit
        body = text[:limit]
        msg = ("Here is output from the REA reverse-engineering tool. Summarise what it shows and "
               "point out anything notable.\n\n" + body)
        if cut:
            msg += f"\n\n[output truncated: first {limit} of {len(text)} characters]"
        self.send_to_chat(msg)
