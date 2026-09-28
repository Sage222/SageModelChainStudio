"""Sage Model Chain Studio — Ui Local Models"""
import os
import shutil
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QCheckBox, QGroupBox, QFormLayout, QMessageBox,
    QMenu, QAbstractItemView, QFileDialog, QSpinBox, QDoubleSpinBox,
)
from .constants import (
    LOCAL_GGUF_KIND, LOCAL_TRANSFORMERS_KIND, CUDA_WHEEL_HINT,
    _GGUF_CACHE, _TRANSFORMERS_CACHE,
)
from .models import LocalModel, ChainStep, FavoriteModel
from .utils import (
    get_model_disk_size,
    _cuda_available, llama_cpp_gpu_supported, _is_local_kind,
    unload_all_local_models,
)


class LocalModelsPanel(QGroupBox):
    def __init__(self, persistence, add_to_chain_callback, add_to_favorites_callback):
        super().__init__("\U0001F4BE Local Models \u2014 GGUF / Safetensors")
        self.persistence = persistence
        self.add_to_chain_callback = add_to_chain_callback
        self.add_to_favorites_callback = add_to_favorites_callback
        self.local_models = self.persistence.load_local_models()
        self.on_params_saved = None
        # Optional callback(model_display_name: str), fired right before a
        # model is dropped from the VRAM/RAM cache (Unload All, Save
        # Parameters reload, Remove from List, Delete from Disk). Lets the
        # main window show a transient "Unloading <model>..." footer status.
        self.on_model_unloading = None
        layout = QVBoxLayout(self)

        status_row = QHBoxLayout()
        transformers_gpu = "GPU (CUDA) available" if _cuda_available() else "CPU only"
        self.transformers_gpu_label = QLabel(f"Safetensors/torch device: {transformers_gpu}")
        self.transformers_gpu_label.setObjectName("sectionHint")
        status_row.addWidget(self.transformers_gpu_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        gguf_status_row = QHBoxLayout()
        self.gguf_gpu_label = QLabel()
        self.gguf_gpu_label.setObjectName("sectionHint")
        gguf_status_row.addWidget(self.gguf_gpu_label, 1)
        check_gpu_btn = QPushButton("Check GGUF GPU Support")
        check_gpu_btn.clicked.connect(self._check_gguf_gpu_support)
        gguf_status_row.addWidget(check_gpu_btn)
        layout.addLayout(gguf_status_row)
        self._refresh_gguf_gpu_label()

        btn_row = QHBoxLayout()
        gguf_btn = QPushButton("+ Load GGUF File...")
        gguf_btn.setObjectName("accent")
        gguf_btn.clicked.connect(self._load_gguf)
        btn_row.addWidget(gguf_btn)
        st_btn = QPushButton("+ Load Safetensors Folder...")
        st_btn.setObjectName("accent")
        st_btn.clicked.connect(self._load_safetensors)
        btn_row.addWidget(st_btn)
        unload_btn = QPushButton("\u274C Unload All from VRAM/RAM")
        unload_btn.setObjectName("danger")
        unload_btn.clicked.connect(self._unload_all)
        btn_row.addWidget(unload_btn)
        layout.addLayout(btn_row)

        remove_btn_row = QHBoxLayout()
        self.remove_selected_btn = QPushButton("Remove Selected (keep file on disk)")
        self.remove_selected_btn.clicked.connect(self._remove_selected)
        remove_btn_row.addWidget(self.remove_selected_btn)
        self.delete_selected_btn = QPushButton("\U0001F5D1 Delete Selected from Disk...")
        self.delete_selected_btn.setObjectName("danger")
        self.delete_selected_btn.clicked.connect(self._delete_selected)
        remove_btn_row.addWidget(self.delete_selected_btn)
        layout.addLayout(remove_btn_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.itemDoubleClicked.connect(self._on_double_click)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)

        params_group = QGroupBox("Model Parameters (applies to selected local model)")
        params_layout = QFormLayout(params_group)

        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 131072)
        self.ctx_spin.setValue(16000)
        self.ctx_spin.setToolTip(
            "Maximum context window in tokens.\n"
            "Larger = more text memory but more VRAM.\n"
            "KV cache scales linearly with this value.\n"
            "Typical: 2048-8192. Reduce if you get OOM errors."
        )
        params_layout.addRow("Context Length:", self.ctx_spin)

        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(-1, 999)
        self.gpu_layers_spin.setValue(-1)
        self.gpu_layers_spin.setToolTip(
            "GGUF only: Number of transformer layers to offload to GPU VRAM.\n"
            "-1 = all layers (fastest, needs enough VRAM).\n"
            "0 = CPU only.\n"
            "For 7B Q4 model: ~33 layers fits in 6GB VRAM.\n"
            "This ONLY works if llama-cpp-python was built with CUDA support --\n"
            "use 'Check GGUF GPU Support' above to verify, a plain pip install\n"
            "is CPU-only and silently ignores this setting.\n"
            "If you get CUDA OOM, reduce this number instead of using -1."
        )
        params_layout.addRow("GPU Layers:", self.gpu_layers_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.05)
        self.temp_spin.setValue(0.7)
        self.temp_spin.setToolTip(
            "Controls randomness of output.\n"
            "0.0 = deterministic/greedy decoding.\n"
            "0.7 = balanced creativity (recommended).\n"
            "1.0+ = more random/creative.\n"
            ">1.2 = increasingly chaotic."
        )
        params_layout.addRow("Temperature:", self.temp_spin)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 32768)
        self.max_tokens_spin.setValue(1024)
        self.max_tokens_spin.setToolTip(
            "Maximum number of tokens to generate per response.\n"
            "Higher = longer outputs but slower and more VRAM for KV cache.\n"
            "512 = concise, 1024 = standard, 2048+ = detailed.\n"
            "GGUF: also affects generation time linearly."
        )
        params_layout.addRow("Max Tokens:", self.max_tokens_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.9)
        self.top_p_spin.setToolTip(
            "Nucleus sampling: only consider tokens comprising the top P probability mass.\n"
            "0.9 = consider top 90% of probability mass.\n"
            "1.0 = disabled (all tokens considered).\n"
            "Lower = more focused/deterministic.\n"
            "Use temperature OR top_p, not both aggressively."
        )
        params_layout.addRow("Top P:", self.top_p_spin)

        self.repeat_penalty_spin = QDoubleSpinBox()
        self.repeat_penalty_spin.setRange(0.5, 2.0)
        self.repeat_penalty_spin.setSingleStep(0.05)
        self.repeat_penalty_spin.setValue(1.1)
        self.repeat_penalty_spin.setToolTip(
            "Penalizes repeated tokens to prevent loops.\n"
            "1.0 = no penalty.\n"
            "1.1 = mild (recommended default).\n"
            "1.3+ = strong (may reduce coherence).\n"
            "Increase if outputs are repetitive."
        )
        params_layout.addRow("Repeat Penalty:", self.repeat_penalty_spin)

        save_params_btn = QPushButton("Save Parameters to Selected Model")
        save_params_btn.clicked.connect(self._save_params)
        reset_params_btn = QPushButton("Reset to Defaults")
        reset_params_btn.clicked.connect(self._reset_params)
        params_btn_row = QHBoxLayout()
        params_btn_row.addWidget(save_params_btn)
        params_btn_row.addWidget(reset_params_btn)
        params_layout.addRow(params_btn_row)

        list_params_row = QHBoxLayout()
        list_params_row.addWidget(self.list_widget, 1)
        list_params_row.addWidget(params_group, 1)
        layout.addLayout(list_params_row)

        self._refresh_list()

        self.setStyleSheet(
            "QLineEdit, QComboBox, QPushButton { padding: 3px 8px; min-height: 16px; }"
        )

    def _refresh_gguf_gpu_label(self):
        installed, gpu_ok, detail = llama_cpp_gpu_supported()
        if not installed:
            self.gguf_gpu_label.setText("GGUF GPU offload: llama-cpp-python not installed")
            self.gguf_gpu_label.setStyleSheet("color: #f87171;")
        elif gpu_ok:
            self.gguf_gpu_label.setText("GGUF GPU offload: supported \u2705 (CUDA build detected)")
            self.gguf_gpu_label.setStyleSheet("color: #4ade80;")
        else:
            self.gguf_gpu_label.setText("GGUF GPU offload: NOT supported \u274c (CPU-only build)")
            self.gguf_gpu_label.setStyleSheet("color: #f87171;")

    def _check_gguf_gpu_support(self):
        installed, gpu_ok, detail = llama_cpp_gpu_supported()
        self._refresh_gguf_gpu_label()
        if not installed:
            QMessageBox.warning(
                self, "llama-cpp-python not installed",
                "Install it first:\n\npip install llama-cpp-python\n\n"
                "For GPU support, install the CUDA build instead:\n\n" + CUDA_WHEEL_HINT
            )
        elif gpu_ok:
            QMessageBox.information(
                self, "GGUF GPU Support Detected",
                "Your llama-cpp-python build supports GPU offload.\n\n" + detail +
                "\n\nSet 'GPU Layers' to -1 in the parameters below to load the full "
                "model into VRAM."
            )
        else:
            QMessageBox.warning(
                self, "GGUF GPU Support NOT Detected",
                "Your installed llama-cpp-python is CPU-only.\n" + detail +
                "\n\nThis is why models are loading into system RAM instead of VRAM -- "
                "the n_gpu_layers setting is silently ignored on CPU-only builds.\n\n"
                "To fix it, reinstall with a CUDA-enabled build:\n\n" + CUDA_WHEEL_HINT +
                "\n\nOr compile from source with the CUDA Toolkit installed:\n\n"
                "Windows PowerShell:\n"
                '$env:CMAKE_ARGS="-DGGML_CUDA=on"\n'
                "pip install llama-cpp-python --force-reinstall --no-cache-dir\n\n"
                "Linux/macOS:\n"
                'CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir'
            )

    def _load_gguf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a GGUF Model File", "", "GGUF Models (*.gguf)")
        if not path:
            return
        self._add_local_model(LocalModel(kind=LOCAL_GGUF_KIND, path=path, display_name=os.path.basename(path)))

    def _load_safetensors(self):
        path = QFileDialog.getExistingDirectory(self, "Select a Model Folder")
        if not path:
            return
        try:
            has_st = any(f.endswith(".safetensors") for f in os.listdir(path))
        except OSError:
            has_st = False
        if not has_st:
            QMessageBox.warning(self, "No .safetensors found", "That folder doesn't appear to contain .safetensors files. Added anyway.")
        name = os.path.basename(os.path.normpath(path)) or path
        self._add_local_model(LocalModel(kind=LOCAL_TRANSFORMERS_KIND, path=path, display_name=name))

    def _add_local_model(self, lm):
        for existing in self.local_models:
            if existing.path == lm.path:
                QMessageBox.information(self, "Already added", "This model is already in your local list.")
                return
        self.local_models.append(lm)
        self.persistence.save_local_models(self.local_models)
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        for lm in self.local_models:
            kl = "GGUF" if lm.kind == LOCAL_GGUF_KIND else "Safetensors"
            gpu_info = f" GPU layers={lm.n_gpu_layers}" if lm.kind == LOCAL_GGUF_KIND else " GPU=auto"
            item = QListWidgetItem(f"[{kl}] {lm.display_name} (ctx={lm.n_ctx}, temp={lm.temperature}{gpu_info})")
            item.setData(Qt.ItemDataRole.UserRole, lm)
            item.setToolTip(lm.path)
            self.list_widget.addItem(item)

    def _on_selection_changed(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        lm = item.data(Qt.ItemDataRole.UserRole)
        self.ctx_spin.setValue(lm.n_ctx)
        self.gpu_layers_spin.setValue(lm.n_gpu_layers)
        self.temp_spin.setValue(lm.temperature)
        self.max_tokens_spin.setValue(lm.max_tokens)
        self.top_p_spin.setValue(lm.top_p)
        self.repeat_penalty_spin.setValue(lm.repeat_penalty)

    def _save_params(self):
        item = self.list_widget.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Select a model first.")
            return
        lm = item.data(Qt.ItemDataRole.UserRole)
        lm.n_ctx = self.ctx_spin.value()
        lm.n_gpu_layers = self.gpu_layers_spin.value()
        lm.temperature = self.temp_spin.value()
        lm.max_tokens = self.max_tokens_spin.value()
        lm.top_p = self.top_p_spin.value()
        lm.repeat_penalty = self.repeat_penalty_spin.value()
        self.persistence.save_local_models(self.local_models)
        self._refresh_list()
        self.list_widget.setCurrentRow(self.list_widget.currentRow())
        if lm.path in _GGUF_CACHE or lm.path in _TRANSFORMERS_CACHE:
            if self.on_model_unloading:
                self.on_model_unloading(lm.display_name)
        if lm.path in _GGUF_CACHE:
            del _GGUF_CACHE[lm.path]
        if lm.path in _TRANSFORMERS_CACHE:
            del _TRANSFORMERS_CACHE[lm.path]
        QMessageBox.information(self, "Saved", f"Parameters saved for {lm.display_name}.\nModel will reload with new parameters on next use.")
        if self.on_params_saved:
            self.on_params_saved()

    def _reset_params(self):
        self.ctx_spin.setValue(16000)
        self.gpu_layers_spin.setValue(-1)
        self.temp_spin.setValue(0.7)
        self.max_tokens_spin.setValue(1024)
        self.top_p_spin.setValue(0.9)
        self.repeat_penalty_spin.setValue(1.1)

    def _unload_all(self):
        count = len(_GGUF_CACHE) + len(_TRANSFORMERS_CACHE)
        if count == 0:
            QMessageBox.information(self, "Nothing to unload", "No local models are currently loaded in VRAM/RAM.")
            return
        if self.on_model_unloading:
            self.on_model_unloading(f"{count} model(s)")
        cleared = unload_all_local_models()
        QMessageBox.information(self, "Unloaded", f"Cleared {len(cleared)} model(s) from VRAM/RAM:\n" + "\n".join(cleared))

    def _favorite_label(self, lm):
        return "Local (GGUF)" if lm.kind == LOCAL_GGUF_KIND else "Local (Safetensors)"

    def _build_step(self, lm):
        return ChainStep(
            provider_label=self._favorite_label(lm), kind=lm.kind, base_url="", api_key="",
            model_id=lm.path, prompt_template="{input}", display_name=lm.display_name,
        )

    def _on_double_click(self, item):
        self.add_to_chain_callback(self._build_step(item.data(Qt.ItemDataRole.UserRole)))

    def _unload_from_cache(self, lm):
        was_cached = lm.path in _GGUF_CACHE or lm.path in _TRANSFORMERS_CACHE
        if was_cached and self.on_model_unloading:
            self.on_model_unloading(lm.display_name)
        if lm.path in _GGUF_CACHE:
            del _GGUF_CACHE[lm.path]
        if lm.path in _TRANSFORMERS_CACHE:
            del _TRANSFORMERS_CACHE[lm.path]

    def _selected_model(self):
        item = self.list_widget.currentItem()
        if not item:
            QMessageBox.information(self, "No selection", "Select a model in the list first.")
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _remove_selected(self):
        lm = self._selected_model()
        if lm is None:
            return
        confirm = QMessageBox.question(
            self, "Remove from list",
            f"Remove '{lm.display_name}' from your local models list?\n\n"
            f"The file itself will NOT be deleted:\n{lm.path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._forget_model(lm)

    def _delete_selected(self):
        lm = self._selected_model()
        if lm is None:
            return
        self._delete_model_from_disk(lm)

    def _forget_model(self, lm):
        """Remove from the app's list only. Files on disk are untouched."""
        self._unload_from_cache(lm)
        self.local_models = [m for m in self.local_models if m.path != lm.path]
        self.persistence.save_local_models(self.local_models)
        self._refresh_list()

    def _delete_model_from_disk(self, lm):
        kind_label = "GGUF file" if lm.kind == LOCAL_GGUF_KIND else "safetensors folder (and everything in it)"
        confirm = QMessageBox.warning(
            self, "Delete from disk",
            f"This will PERMANENTLY delete the {kind_label} for '{lm.display_name}' from your hard drive:\n\n"
            f"{lm.path}\n\nThis cannot be undone. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        self._unload_from_cache(lm)
        try:
            if lm.kind == LOCAL_GGUF_KIND:
                if os.path.exists(lm.path):
                    os.remove(lm.path)
            else:
                if os.path.isdir(lm.path):
                    shutil.rmtree(lm.path)
                elif os.path.exists(lm.path):
                    os.remove(lm.path)
        except OSError as e:
            QMessageBox.critical(
                self, "Delete failed",
                f"Couldn't delete {lm.path}:\n{e}\n\n"
                "It may be open in another program, or you may not have permission. "
                "It has been removed from the app's list regardless -- delete the "
                "leftover file(s) manually if needed."
            )

        self.local_models = [m for m in self.local_models if m.path != lm.path]
        self.persistence.save_local_models(self.local_models)
        self._refresh_list()

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        lm = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        aa = menu.addAction("Add to Chain")
        fa = menu.addAction("\u2605 Add to Favorites")
        menu.addSeparator()
        ra = menu.addAction("Remove from List (keep file on disk)")
        da = menu.addAction("\U0001F5D1 Delete from Disk...")
        chosen = menu.exec(self.list_widget.mapToGlobal(pos))
        if chosen == aa:
            self.add_to_chain_callback(self._build_step(lm))
        elif chosen == fa:
            self.add_to_favorites_callback(FavoriteModel(
                provider_label=self._favorite_label(lm), kind=lm.kind, base_url="",
                model_id=lm.path, display_name=lm.display_name,
            ))
        elif chosen == ra:
            self._forget_model(lm)
        elif chosen == da:
            self._delete_model_from_disk(lm)
