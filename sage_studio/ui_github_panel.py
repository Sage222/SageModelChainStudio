import time
"""Sage Model Chain Studio — GitHub Vibe Coding Panel (v5: fully self-contained)

v5 change: this tab no longer touches the Chat tab at all. It runs its own
ChatWorker instance and has its own Output log. "Run Chain" reads from this
tab's Chain Input box and writes to this tab's Output box. "Push Changes"
pushes the output from THIS tab's run, not anything from Chat.

The only things shared with the rest of the app are read-only: the chain
steps configured in the Model Chain panel (Models tab) and the local model
GPU parameters — both fetched fresh via callbacks each run, never mutated.
"""

import os, re, json, base64, requests
from datetime import datetime
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QPlainTextEdit, QTextEdit, QMessageBox, QGroupBox, QComboBox,
    QFormLayout, QCheckBox, QListWidget, QListWidgetItem,
    QAbstractItemView,
)

from .workers import ChatWorker
from .ui_github_settings import GitHubSettingsPanel
from .constants import DATA_DIR
from .utils import get_gpu_memory

FILE_START = "===FILE:"
FILE_END = "===ENDFILE==="


class GitHubPanel(QWidget):
    """Fully self-contained: own chain execution, own output log, own push logic."""

    def __init__(self, persistence, get_steps_callback, get_local_model_params_callback=None):
        super().__init__()
        self.persistence = persistence
        self.get_steps_callback = get_steps_callback
        self.get_local_model_params_callback = get_local_model_params_callback

        self.tree_items = []
        self.file_cache = {}
        self.base_commit_sha = None
        self.base_tree_sha = None
        self.worker = None
        self._step_count = 1
        self._last_output = ""

        root = QVBoxLayout(self)
        # ── Repo config ─────────────────────────────────────────────────────
        self.github_settings_panel = GitHubSettingsPanel(persistence)
        root.addWidget(self.github_settings_panel)
        repo_group = QGroupBox("Repository")
        repo_form = QFormLayout(repo_group)
        self.repo_edit = QComboBox()
        self.repo_edit.setEditable(True)
        self.repo_edit.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.repo_edit.setPlaceholderText("owner/repo or full GitHub URL")
        self._recent_repos_file = os.path.join(DATA_DIR, "github_recent_repos.json")
        self._load_recent_repos()
        repo_form.addRow("Repo:", self.repo_edit)
        self.branch_edit = QLineEdit("main")
        repo_form.addRow("Branch:", self.branch_edit)

        connect_btn = QPushButton("\U0001F50C Connect & List Files")
        connect_btn.setObjectName("accent")
        connect_btn.clicked.connect(self._connect_and_list_files)
        repo_form.addRow(connect_btn)
        root.addWidget(repo_group)

        # ── File tree / selection ──────────────────────────────────────────
        files_group = QGroupBox("Repository Files (check the ones the model should see)")
        files_layout = QVBoxLayout(files_group)

        filter_row = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by path or extension (e.g. .py)")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit)
        files_layout.addLayout(filter_row)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.file_list.setMaximumHeight(140)
        files_layout.addWidget(self.file_list)

        sel_row = QHBoxLayout()
        select_all_btn = QPushButton("Check All Visible")
        select_all_btn.clicked.connect(lambda: self._check_all(True))
        sel_row.addWidget(select_all_btn)
        select_none_btn = QPushButton("Uncheck All")
        select_none_btn.clicked.connect(lambda: self._check_all(False))
        sel_row.addWidget(select_none_btn)
        files_layout.addLayout(sel_row)

        root.addWidget(files_group)

        # ── Instructions ────────────────────────────────────────────────────
        instr_group = QGroupBox("Modification Instructions")
        instr_layout = QVBoxLayout(instr_group)
        self.instructions_edit = QPlainTextEdit()
        self.instructions_edit.setPlaceholderText(
            "Describe what you want changed across the selected files."
        )
        self.instructions_edit.setFixedHeight(55)
        instr_layout.addWidget(self.instructions_edit)
        root.addWidget(instr_group)

        prepare_btn = QPushButton("\u2193 Prepare Input for Chain")
        prepare_btn.setObjectName("accent")
        prepare_btn.clicked.connect(self._prepare_input)
        root.addWidget(prepare_btn)

        # ── Chain Input ──────────────────────────────────────────────────────
        chain_input_group = QGroupBox("Chain Input")
        chain_input_layout = QVBoxLayout(chain_input_group)
        self.chain_input_edit = QPlainTextEdit()
        self.chain_input_edit.setPlaceholderText(
            "Click 'Prepare Input for Chain' above, or type/paste your own prompt here."
        )
        self.chain_input_edit.setFixedHeight(140)
        chain_input_layout.addWidget(self.chain_input_edit)

        self.chain_context_label = QLabel("")
        self.chain_context_label.setObjectName("sectionHint")
        chain_input_layout.addWidget(self.chain_context_label)
        self.chain_input_edit.textChanged.connect(self._update_chain_context_meter)
        root.addWidget(chain_input_group)

        # ── Run controls ─────────────────────────────────────────────────────
        run_options_row = QHBoxLayout()
        self.retry_check = QCheckBox("Auto-retry on 429")
        self.retry_check.setChecked(True)
        run_options_row.addWidget(self.retry_check)
        run_options_row.addStretch(1)
        root.addLayout(run_options_row)

        run_row = QHBoxLayout()
        self.run_chain_btn = QPushButton("\u25b6 Run Chain")
        self.run_chain_btn.setObjectName("accent")
        self.run_chain_btn.clicked.connect(self._run_chain)
        run_row.addWidget(self.run_chain_btn, 1)
        self.stop_btn = QPushButton("\u25a0 Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_chain)
        run_row.addWidget(self.stop_btn)
        root.addLayout(run_row)

        # ── Output (this tab's own, separate from Chat) ─────────────────────
        root.addWidget(QLabel("Output:"))
        self.output_log = QTextEdit()
        self.output_log.setReadOnly(True)
        self.output_log.setFixedHeight(220)
        self.output_log.setStyleSheet("font-size: 14px;")
        root.addWidget(self.output_log)

        self.speed_vram_label = QLabel("")
        self.speed_vram_label.setObjectName("sectionHint")
        root.addWidget(self.speed_vram_label)

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._poll_vram)
        self._vram_timer.start(1000)
        self._last_tps = 0.0
        self._last_tokens = 0
        self._last_update = 0.0

        # ── Push ─────────────────────────────────────────────────────────────
        push_btn = QPushButton("\u2191 Push Changes (multi-file commit)")
        push_btn.setObjectName("danger")
        push_btn.clicked.connect(self._push_changes)
        root.addWidget(push_btn)

        options_row = QHBoxLayout()
        self.new_branch_check = QCheckBox("Push to a new branch (recommended)")
        self.new_branch_check.setChecked(True)
        options_row.addWidget(self.new_branch_check)
        options_row.addStretch(1)
        root.addLayout(options_row)

        self.commit_msg_edit = QLineEdit()
        self.commit_msg_edit.setPlaceholderText("Commit message (auto-generated if left blank)")
        root.addWidget(QLabel("Commit message:"))
        root.addWidget(self.commit_msg_edit)

        self.status = QLabel("Ready. Connect to a repo to begin.")
        self.status.setObjectName("sectionHint")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        howto = QGroupBox("Workflow (fully contained in this tab)")
        howto_layout = QVBoxLayout(howto)
        howto_layout.addWidget(QLabel(
            "1. Enter token + repo + branch, click 'Connect & List Files'\n"
            "2. Check the files the model needs to see\n"
            "3. Describe your change, click 'Prepare Input for Chain'\n"
            "4. Click 'Run Chain' \u2014 runs here, output shown in this tab's Output box\n"
            "5. Click 'Push Changes' \u2014 pushes the parsed file(s) from this tab's output\n"
            "   as a single atomic commit"
        ))
        root.addWidget(howto)
        root.addStretch(1)

    # ── Token ────────────────────────────────────────────────────────────────

    def _headers(self):
        token = self.persistence.get_key("github")
        if not token:
            raise ValueError("No GitHub token found. Add one on the Settings tab.")
        return {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def _api(self, method, url, json_body=None):
        headers = self._headers()
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            resp = requests.post(url, headers=headers, json=json_body, timeout=30)
        elif method == "PATCH":
            resp = requests.patch(url, headers=headers, json=json_body, timeout=30)
        elif method == "PUT":
            resp = requests.put(url, headers=headers, json=json_body, timeout=30)
        else:
            raise ValueError(f"Unsupported method {method}")

        if resp.status_code == 401:
            raise ValueError("Invalid or expired GitHub token.")
        if resp.status_code == 404:
            raise ValueError(f"Not found: {url}")
        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            raise ValueError(f"Forbidden/rate-limited (remaining: {remaining}). Check token scopes.")
        if resp.status_code >= 400:
            raise ValueError(f"GitHub API error {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.text else {}


    def _load_recent_repos(self):
        try:
            if os.path.exists(self._recent_repos_file):
                with open(self._recent_repos_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
            else:
                items = []
        except Exception:
            items = []
        self.repo_edit.blockSignals(True)
        self.repo_edit.clear()
        self.repo_edit.addItems(items)
        self.repo_edit.setCurrentText("")
        self.repo_edit.blockSignals(False)

    def _save_recent_repo(self, repo):
        items = [self.repo_edit.itemText(i) for i in range(self.repo_edit.count())]
        items = [repo] + [r for r in items if r != repo]
        items = items[:8]
        try:
            with open(self._recent_repos_file, "w", encoding="utf-8") as f:
                json.dump(items, f, indent=2)
        except Exception:
            pass
        self.repo_edit.blockSignals(True)
        self.repo_edit.clear()
        self.repo_edit.addItems(items)
        self.repo_edit.setCurrentText(repo)
        self.repo_edit.blockSignals(False)

    def _normalize_repo(self, raw):
        raw = raw.strip()
        match = re.search(r"github\.com[:/]+([^/\s]+)/([^/\s#]+)", raw)
        if match:
            owner = match.group(1)
            repo_name = match.group(2)
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            return f"{owner}/{repo_name}"
        return raw

    def _repo_branch(self):
        raw = self.repo_edit.currentText().strip()
        if not raw:
            raise ValueError("Enter the repo (owner/repo or full GitHub URL) first.")
        repo = self._normalize_repo(raw)
        branch = self.branch_edit.text().strip() or "main"
        return repo, branch

    # ── Connect + list files ────────────────────────────────────────────────

    def _connect_and_list_files(self):
        try:
            repo, branch = self._repo_branch()
            self.status.setText(f"Connecting to {repo} ({branch})...")

            branch_data = self._api("GET", f"https://api.github.com/repos/{repo}/branches/{branch}")
            commit_sha = branch_data["commit"]["sha"]
            self.base_commit_sha = commit_sha
            self.base_tree_sha = branch_data["commit"]["commit"]["tree"]["sha"]

            tree_data = self._api(
                "GET",
                f"https://api.github.com/repos/{repo}/git/trees/{commit_sha}?recursive=1",
            )
            entries = tree_data.get("tree", [])
            self.tree_items = [e for e in entries if e.get("type") == "blob"]
            self.tree_items.sort(key=lambda e: e["path"])

            self._populate_file_list()
            self._save_recent_repo(repo)
            self.status.setText(
                f"Connected to {repo} @ {branch}. {len(self.tree_items)} file(s) found."
            )
        except Exception as e:
            self.status.setText("Connection failed.")
            QMessageBox.critical(self, "Connection failed", str(e))

    def _populate_file_list(self):
        self.file_list.clear()
        for entry in self.tree_items:
            item = QListWidgetItem(entry["path"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.file_list.addItem(item)

    def _apply_filter(self, text):
        text = text.strip().lower()
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def _check_all(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if not item.isHidden():
                item.setCheckState(state)

    def _selected_paths(self):
        paths = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                paths.append(item.data(Qt.ItemDataRole.UserRole)["path"])
        return paths

    # ── Prepare input ────────────────────────────────────────────────────────

    def _fetch_file_content(self, repo, branch, path):
        if path in self.file_cache:
            return self.file_cache[path]
        data = self._api(
            "GET", f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
        )
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
        entry = {"content": content, "sha": data.get("sha")}
        self.file_cache[path] = entry
        return entry

    def _prepare_input(self):
        try:
            repo, branch = self._repo_branch()
            paths = self._selected_paths()
            if not paths:
                QMessageBox.information(self, "No files selected", "Check at least one file first.")
                return

            instructions = self.instructions_edit.toPlainText().strip()
            if not instructions:
                QMessageBox.information(self, "No instructions", "Describe what you want changed.")
                return

            self.status.setText(f"Fetching {len(paths)} file(s)...")

            sections = []
            for path in paths:
                entry = self._fetch_file_content(repo, branch, path)
                sections.append(f"--- {path} ---\n```\n{entry['content']}\n```")

            files_blob = "\n\n".join(sections)

            prompt = (
                f"You are a coding assistant working directly on the GitHub repository `{repo}` "
                f"(branch `{branch}`). Below are the current contents of {len(paths)} file(s).\n\n"
                f"## Modification Request\n{instructions}\n\n"
                f"## Current Files\n{files_blob}\n\n"
                f"## Output Format (STRICT — required for automated parsing)\n"
                f"For EVERY file you modify, output a block in exactly this format:\n\n"
                f"{FILE_START} path/to/file.ext\n"
                f"<the ENTIRE new content of the file, not a diff>\n"
                f"{FILE_END}\n\n"
                f"Rules:\n"
                f"- Include a block for every file you changed, using its exact original path.\n"
                f"- Do NOT include files you did not change.\n"
                f"- Do NOT wrap the blocks in markdown code fences.\n"
                f"- Do NOT add explanations, summaries, or text outside the {FILE_START}/{FILE_END} blocks.\n"
                f"- Each file block must contain the complete file content, ready to save as-is.\n"
            )

            self.chain_input_edit.setPlainText(prompt)
            self.status.setText(
                f"Prepared input with {len(paths)} file(s) ({len(prompt):,} chars). Click 'Run Chain'."
            )
        except Exception as e:
            self.status.setText("Prepare failed.")
            QMessageBox.critical(self, "Prepare failed", str(e))

    # ── Self-contained chain execution ──────────────────────────────────────

    def _update_chain_context_meter(self):
        text = self.chain_input_edit.toPlainText()
        est_tokens = max(0, len(text) // 4)
        max_ctx = None
        try:
            steps = self.get_steps_callback() if self.get_steps_callback else []
        except Exception:
            steps = []
        if steps:
            first = steps[0]
            if first.kind in ("local_gguf", "local_transformers"):
                local_params = self.get_local_model_params_callback() if self.get_local_model_params_callback else {}
                params = (local_params or {}).get(first.model_id, {})
                max_ctx = params.get("n_ctx")
        if max_ctx:
            pct = (est_tokens / max_ctx * 100) if max_ctx else 0
            self.chain_context_label.setText(
                f"~{est_tokens:,} tokens (~{pct:.0f}% of {max_ctx:,} local context)"
            )
        else:
            self.chain_context_label.setText(
                f"~{est_tokens:,} tokens (estimate \u2014 remote providers have their own limits)"
            )

    def _run_chain(self):
        if not self.get_steps_callback:
            QMessageBox.warning(self, "Not wired up", "get_steps_callback is not connected.")
            return
        steps = self.get_steps_callback()
        if not steps:
            QMessageBox.information(
                self, "No steps",
                "Add at least one model to the chain first (Model Chain panel, Models tab)."
            )
            return

        initial_input = self.chain_input_edit.toPlainText().strip()
        if not initial_input:
            QMessageBox.information(self, "No input", "Prepare or type input first.")
            return

        self._step_count = len(steps)
        self.output_log.clear()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.output_log.append(
            f"<span style='color:#757c8f'>[{timestamp}]</span> "
            f"<b><span style='color:#4da6ff'>Input:</span></b> "
            f"<span style='color:#4da6ff'>{initial_input}</span>"
        )

        self.run_chain_btn.setEnabled(False)
        self.run_chain_btn.setText("Running...")
        self.stop_btn.setEnabled(True)

        local_params = self.get_local_model_params_callback() if self.get_local_model_params_callback else {}

        self.worker = ChatWorker(steps, initial_input, self.retry_check.isChecked(), local_params)
        self.worker.step_done.connect(self._on_step_done)
        self.worker.step_retry.connect(self._on_step_retry)
        self.worker.step_error.connect(self._on_step_error)
        self.worker.step_stopped.connect(self._on_step_stopped)
        self.worker.step_loading.connect(self._on_step_loading)
        self.worker.step_warning.connect(self._on_step_warning)
        self.worker.token_progress.connect(self._on_token_progress)
        self.worker.chain_finished.connect(self._on_finished)
        self.worker.start()

        self.status.setText("Chain running...")

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
        self._last_tps = tps
        self._last_tokens = count
        self._last_update = time.time()
        self._poll_vram()

    def _stop_chain(self):
        if self.worker is not None:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping...")

    def _reset_run_controls(self):
        self.run_chain_btn.setEnabled(True)
        self.run_chain_btn.setText("\u25b6 Run Chain")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("\u25a0 Stop")

    def _on_step_loading(self, idx, label):
        if self._step_count <= 1:
            return
        self.output_log.append(f"<i style='color:#60a5fa'>Step {idx + 1}: loading local model '{label}' into VRAM...</i>")

    def _on_step_warning(self, idx, msg):
        safe_msg = msg.replace("\n", "<br>")
        self.output_log.append(
            f"<b style='color:#facc15'>\u26a0 Step {idx + 1} warning:</b><br>"
            f"<span style='color:#facc15'>{safe_msg}</span>"
        )

    def _on_step_retry(self, idx, attempt, wait, reason):
        self.output_log.append(
            f"<i style='color:#facc15'>Step {idx + 1}: {reason}, retry {attempt} in {wait:.1f}s...</i>"
        )

    def _on_step_done(self, idx, output):
        if self._step_count > 1:
            self.output_log.append(f"<b>\u2500\u2500 Step {idx + 1} output \u2500\u2500</b>")
        self.output_log.append(output)
        self.output_log.append("")

    def _on_step_error(self, idx, err):
        self.output_log.append(f"<b style='color:#f87171'>Step {idx + 1} failed:</b> {err}")
        self._reset_run_controls()
        self.status.setText("Chain failed.")

    def _on_step_stopped(self, idx):
        self.output_log.append(f"<i style='color:#facc15'>Stopped before/at step {idx + 1} by user request.</i>")
        self.output_log.append("<hr>")
        self._reset_run_controls()
        self.status.setText("Chain stopped.")

    def _on_finished(self):
        if self._step_count > 1:
            self.output_log.append("<b style='color:#4ade80'>Chain complete.</b>")
        self.output_log.append("<hr>")
        final_text = ""
        if self.worker is not None and self.worker.step_results:
            final_text = self.worker.step_results[-1].get("output") or ""
        self._last_output = final_text
        self._reset_run_controls()
        self.status.setText("Chain complete. Ready to push.")

    # ── Parse model output ──────────────────────────────────────────────────

    def _parse_file_blocks(self, text):
        pattern = re.compile(
            re.escape(FILE_START) + r"\s*(.+?)\s*\n(.*?)" + re.escape(FILE_END),
            re.DOTALL,
        )
        results = {}
        for match in pattern.finditer(text):
            path = match.group(1).strip()
            content = match.group(2)
            content = content.strip("\n")
            results[path] = content
        return results

    # ── Push changes (atomic multi-file commit via Git Data API) ───────────

    def _push_changes(self):
        try:
            repo, branch = self._repo_branch()
            output = self._last_output
            if not output:
                QMessageBox.information(self, "No chain output", "Run the chain first (in this tab).")
                return

            files = self._parse_file_blocks(output)

            if not files:
                paths = self._selected_paths()
                if len(paths) == 1:
                    files = {paths[0]: output.strip()}
                    self.status.setText("No file markers found — treating entire output as the single selected file.")
                else:
                    QMessageBox.warning(
                        self, "Could not parse output",
                        f"No {FILE_START} ... {FILE_END} blocks found in the chain output, "
                        f"and more than one file was selected, so I can't tell which content "
                        f"belongs to which file.\n\nCheck that your model followed the output format."
                    )
                    return

            if not self.base_commit_sha or not self.base_tree_sha:
                QMessageBox.warning(self, "Not connected", "Click 'Connect & List Files' first.")
                return

            confirm = QMessageBox.question(
                self, "Confirm push",
                f"Push {len(files)} file(s) to {repo}?\n\n" + "\n".join(files.keys()),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

            self.status.setText("Creating blobs...")

            tree_entries = []
            for path, content in files.items():
                blob_data = self._api(
                    "POST", f"https://api.github.com/repos/{repo}/git/blobs",
                    {"content": content, "encoding": "utf-8"},
                )
                tree_entries.append({
                    "path": path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_data["sha"],
                })

            self.status.setText("Creating tree...")
            tree_data = self._api(
                "POST", f"https://api.github.com/repos/{repo}/git/trees",
                {"base_tree": self.base_tree_sha, "tree": tree_entries},
            )
            new_tree_sha = tree_data["sha"]

            commit_msg = self.commit_msg_edit.text().strip()
            if not commit_msg:
                instructions = self.instructions_edit.toPlainText().strip()
                commit_msg = f"AI: {instructions[:80]}" if instructions else f"AI-modified {len(files)} file(s)"

            self.status.setText("Creating commit...")
            commit_data = self._api(
                "POST", f"https://api.github.com/repos/{repo}/git/commits",
                {
                    "message": commit_msg,
                    "tree": new_tree_sha,
                    "parents": [self.base_commit_sha],
                },
            )
            new_commit_sha = commit_data["sha"]

            if self.new_branch_check.isChecked():
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                target_branch = f"vibe-{ts}"
                self._api(
                    "POST", f"https://api.github.com/repos/{repo}/git/refs",
                    {"ref": f"refs/heads/{target_branch}", "sha": new_commit_sha},
                )
            else:
                target_branch = branch
                self._api(
                    "PATCH", f"https://api.github.com/repos/{repo}/git/refs/heads/{branch}",
                    {"sha": new_commit_sha, "force": False},
                )

            self.base_commit_sha = new_commit_sha
            self.base_tree_sha = new_tree_sha
            for path, content in files.items():
                self.file_cache[path] = {"content": content, "sha": None}

            self.status.setText(f"Pushed {len(files)} file(s) to '{target_branch}'.")

            compare_url = (
                f"https://github.com/{repo}/compare/{branch}...{target_branch}"
                if target_branch != branch else
                f"https://github.com/{repo}/commit/{new_commit_sha}"
            )

            QMessageBox.information(
                self, "Pushed!",
                f"Committed {len(files)} file(s) to branch '{target_branch}':\n\n"
                + "\n".join(files.keys())
                + f"\n\nView: {compare_url}"
            )

        except Exception as e:
            self.status.setText("Push failed.")
            QMessageBox.critical(self, "Push failed", str(e))
