"""
Sage LLM Studio — Image tab.

Local-only image generation via diffusers. No Hugging Face browsing/download
here per request -- you point the app at models you already have on disk.

Two ways to add a model:
  - "Browse for Checkpoint File..." -- a single packed .safetensors file
    (most SD1.5/SDXL checkpoints downloaded from Civitai etc. look like this)
  - "Browse for Model Folder..." -- a diffusers-format folder containing
    model_index.json plus unet/vae/text_encoder subfolders (this is what
    you get from `diffusers`' own save_pretrained(), or many Hugging Face
    repos structured for direct from_pretrained() loading)

You pick the pipeline "family" (SD 1.5 / SDXL / SD3 / FLUX) when adding a
model, since diffusers needs to know which pipeline class to instantiate --
there's no fully reliable way to auto-detect this from a bare file/folder.

Mode toggle: Text to Image / Image to Image. Both modes use the SAME loaded
checkpoint -- img2img just re-wires the already-loaded model components into
a different pipeline wrapper (no reload, no separate model needed).

NOT supported in this version (flagged honestly rather than silently failing):
  - GGUF-quantized diffusion checkpoints (a different quant format than
    diffusers' native loaders expect -- this is a v2 feature, not v1)
  - LoRAs, ControlNets, inpainting/outpainting
  - Live per-step progress (diffusers' step-callback API differs across
    versions; showing a busy indicator instead of guessing wrong)
  - Cancelling mid-generation (single blocking call, no clean interrupt point
    without the step callback)
"""
import os
import subprocess
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QGroupBox, QFormLayout,
    QMessageBox, QMenu, QAbstractItemView, QFileDialog, QSpinBox,
    QDoubleSpinBox, QPlainTextEdit, QSplitter, QProgressBar, QInputDialog,
)

from .constants import (
    IMAGE_DIFFUSERS_FOLDER_KIND, IMAGE_DIFFUSERS_SINGLE_FILE_KIND,
    IMAGE_PIPELINE_FAMILIES, IMAGE_OUTPUT_DIR,
)
from .models import ImageModel
from .workers import ImageGenWorker, unload_all_image_models

FAMILY_DEFAULT_SIZE = {
    "sd15": (512, 512),
    "sdxl": (1024, 1024),
    "sd3": (1024, 1024),
    "flux": (1024, 1024),
}

FAMILY_LABELS = {
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "Stable Diffusion XL",
    "sd3": "Stable Diffusion 3",
    "flux": "FLUX",
}


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class ImagePanel(QWidget):
    def __init__(self, persistence):
        super().__init__()
        self.persistence = persistence
        self.image_models = self.persistence.load_image_models()
        self.worker = None
        self.source_image_path = None

        root = QVBoxLayout(self)
        title = QLabel("\U0001F5BC\uFE0F Image Generation")
        title.setObjectName("chatTitle")
        root.addWidget(title)

        gpu_status = "GPU (CUDA) available" if _cuda_available() else "CPU only (will be slow)"
        hint = QLabel(
            f"Local image generation via diffusers. Device: {gpu_status}. "
            "Point the app at models you already have on disk -- no browsing/downloading here. "
            "GGUF-quantized diffusion checkpoints, LoRAs, and inpainting are not supported yet."
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("image_panel_splitter")

        # ---- Left: model management ----
        left_group = QGroupBox("Local Image Models")
        left_layout = QVBoxLayout(left_group)

        add_row = QHBoxLayout()
        add_file_btn = QPushButton("+ Browse for Checkpoint File...")
        add_file_btn.setObjectName("accent")
        add_file_btn.clicked.connect(self._browse_checkpoint_file)
        add_row.addWidget(add_file_btn)
        add_folder_btn = QPushButton("+ Browse for Model Folder...")
        add_folder_btn.setObjectName("accent")
        add_folder_btn.clicked.connect(self._browse_model_folder)
        add_row.addWidget(add_folder_btn)
        left_layout.addLayout(add_row)

        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.model_list.itemSelectionChanged.connect(self._on_model_selected)
        self.model_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.model_list.customContextMenuRequested.connect(self._show_context_menu)
        left_layout.addWidget(self.model_list, 1)

        remove_row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Selected (keep file on disk)")
        self.remove_btn.clicked.connect(self._remove_selected)
        remove_row.addWidget(self.remove_btn)
        left_layout.addLayout(remove_row)

        unload_btn = QPushButton("\u274C Unload All from VRAM/RAM")
        unload_btn.setObjectName("danger")
        unload_btn.clicked.connect(self._unload_all)
        left_layout.addWidget(unload_btn)

        splitter.addWidget(left_group)

        # ---- Right: generation controls + preview ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Text to Image", "Image to Image"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        right_layout.addLayout(mode_row)

        # img2img-only controls, hidden in text-to-image mode
        self.img2img_group = QGroupBox("Source Image")
        img2img_layout = QVBoxLayout(self.img2img_group)
        source_row = QHBoxLayout()
        choose_image_btn = QPushButton("Choose Image...")
        choose_image_btn.clicked.connect(self._choose_source_image)
        source_row.addWidget(choose_image_btn)
        self.source_image_label = QLabel("No image selected.")
        self.source_image_label.setObjectName("sectionHint")
        source_row.addWidget(self.source_image_label, 1)
        img2img_layout.addLayout(source_row)

        self.source_thumb = QLabel()
        self.source_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_thumb.setFixedHeight(140)
        self.source_thumb.setStyleSheet("background-color: #1a1c24; border: 1px solid #262a35; border-radius: 8px;")
        img2img_layout.addWidget(self.source_thumb)

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Strength:"))
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setValue(0.6)
        self.strength_spin.setToolTip(
            "How much the model is allowed to change your source image.\n"
            "Low (0.2-0.4) = subtle changes, stays close to the original.\n"
            "High (0.7-0.9) = much more creative freedom, can deviate a lot.\n"
            "1.0 = effectively ignores the source image."
        )
        strength_row.addWidget(self.strength_spin)
        strength_row.addStretch(1)
        img2img_layout.addLayout(strength_row)

        self.img2img_group.setVisible(False)
        right_layout.addWidget(self.img2img_group)

        gen_group = QGroupBox("Generate")
        form = QFormLayout(gen_group)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the image you want...")
        self.prompt_edit.setFixedHeight(80)
        form.addRow("Prompt:", self.prompt_edit)

        self.negative_prompt_edit = QLineEdit()
        self.negative_prompt_edit.setPlaceholderText("Optional -- things to avoid")
        form.addRow("Negative prompt:", self.negative_prompt_edit)

        dims_row = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 2048)
        self.width_spin.setSingleStep(64)
        self.width_spin.setValue(1024)
        dims_row.addWidget(QLabel("Width:"))
        dims_row.addWidget(self.width_spin)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 2048)
        self.height_spin.setSingleStep(64)
        self.height_spin.setValue(1024)
        dims_row.addWidget(QLabel("Height:"))
        dims_row.addWidget(self.height_spin)
        self.dims_row_label = QLabel("Size:")
        form.addRow(self.dims_row_label, dims_row)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 150)
        self.steps_spin.setValue(30)
        self.steps_spin.setToolTip(
            "Number of denoising steps. More = higher quality but slower.\n"
            "20-30 is typical for SDXL, fewer for distilled/turbo models."
        )
        form.addRow("Steps:", self.steps_spin)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(7.0)
        self.guidance_spin.setToolTip(
            "How strongly the image follows your prompt.\n"
            "7.0 is a typical default. Lower = more creative/looser, higher = more literal."
        )
        form.addRow("Guidance scale:", self.guidance_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2_147_483_647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setToolTip("-1 = random seed each time. Set a fixed value to reproduce an image.")
        form.addRow("Seed:", self.seed_spin)

        save_defaults_btn = QPushButton("Save These as Defaults for Selected Model")
        save_defaults_btn.clicked.connect(self._save_defaults)
        form.addRow(save_defaults_btn)

        right_layout.addWidget(gen_group)

        gen_btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("\U0001F3A8 Generate")
        self.generate_btn.setObjectName("accent")
        self.generate_btn.clicked.connect(self._generate)
        gen_btn_row.addWidget(self.generate_btn, 1)
        right_layout.addLayout(gen_btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready.")
        self.status_label.setObjectName("sectionHint")
        self.status_label.setWordWrap(True)
        right_layout.addWidget(self.status_label)

        self.preview_label = QLabel("No image generated yet.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(320)
        self.preview_label.setStyleSheet("background-color: #1a1c24; border: 1px solid #262a35; border-radius: 8px;")
        right_layout.addWidget(self.preview_label, 1)

        preview_btn_row = QHBoxLayout()
        open_folder_btn = QPushButton("Open Output Folder")
        open_folder_btn.clicked.connect(self._open_output_folder)
        preview_btn_row.addWidget(open_folder_btn)
        self.use_output_as_source_btn = QPushButton("Use This Output as Next Source Image")
        self.use_output_as_source_btn.clicked.connect(self._use_output_as_source)
        self.use_output_as_source_btn.setEnabled(False)
        preview_btn_row.addWidget(self.use_output_as_source_btn)
        right_layout.addLayout(preview_btn_row)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self._refresh_model_list()
        self._last_image_path = None

    # ---- mode ----

    def _on_mode_changed(self, text):
        is_img2img = (text == "Image to Image")
        self.img2img_group.setVisible(is_img2img)
        self.dims_row_label.setText("Output size (resizes source):" if is_img2img else "Size:")

    # ---- model management ----

    def _add_model_with_family_prompt(self, kind, path, default_name):
        family, ok = QInputDialog.getItem(
            self, "Pipeline family",
            "Which pipeline family is this model? (there's no fully reliable way\n"
            "to auto-detect this from the file/folder alone)",
            [FAMILY_LABELS[f] for f in IMAGE_PIPELINE_FAMILIES], 1, False,
        )
        if not ok:
            return
        family_key = next(f for f in IMAGE_PIPELINE_FAMILIES if FAMILY_LABELS[f] == family)
        w, h = FAMILY_DEFAULT_SIZE.get(family_key, (1024, 1024))

        for existing in self.image_models:
            if existing.path == path:
                QMessageBox.information(self, "Already added", "This model is already in your list.")
                return

        model = ImageModel(kind=kind, path=path, display_name=default_name, family=family_key, width=w, height=h)
        self.image_models.append(model)
        self.persistence.save_image_models(self.image_models)
        self._refresh_model_list()

    def _browse_checkpoint_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a Checkpoint File", "", "Safetensors (*.safetensors)")
        if not path:
            return
        self._add_model_with_family_prompt(IMAGE_DIFFUSERS_SINGLE_FILE_KIND, path, os.path.basename(path))

    def _browse_model_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select a Diffusers Model Folder")
        if not path:
            return
        if not os.path.exists(os.path.join(path, "model_index.json")):
            proceed = QMessageBox.warning(
                self, "No model_index.json found",
                "This folder doesn't look like a diffusers-format model (no model_index.json). "
                "Add it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if proceed != QMessageBox.StandardButton.Yes:
                return
        name = os.path.basename(os.path.normpath(path)) or path
        self._add_model_with_family_prompt(IMAGE_DIFFUSERS_FOLDER_KIND, path, name)

    def _refresh_model_list(self):
        self.model_list.clear()
        for m in self.image_models:
            kl = "File" if m.kind == IMAGE_DIFFUSERS_SINGLE_FILE_KIND else "Folder"
            item = QListWidgetItem(f"[{FAMILY_LABELS.get(m.family, m.family)} \u00b7 {kl}] {m.display_name}")
            item.setData(Qt.ItemDataRole.UserRole, m)
            item.setToolTip(m.path)
            self.model_list.addItem(item)

    def _selected_model(self):
        item = self.model_list.currentItem()
        if not item:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_model_selected(self):
        m = self._selected_model()
        if not m:
            return
        self.width_spin.setValue(m.width)
        self.height_spin.setValue(m.height)
        self.steps_spin.setValue(m.steps)
        self.guidance_spin.setValue(m.guidance_scale)
        self.negative_prompt_edit.setText(m.negative_prompt)

    def _save_defaults(self):
        m = self._selected_model()
        if not m:
            QMessageBox.information(self, "No selection", "Select a model first.")
            return
        m.width = self.width_spin.value()
        m.height = self.height_spin.value()
        m.steps = self.steps_spin.value()
        m.guidance_scale = self.guidance_spin.value()
        m.negative_prompt = self.negative_prompt_edit.text()
        self.persistence.save_image_models(self.image_models)
        QMessageBox.information(self, "Saved", f"Defaults saved for {m.display_name}.")

    def _remove_selected(self):
        m = self._selected_model()
        if not m:
            QMessageBox.information(self, "No selection", "Select a model in the list first.")
            return
        confirm = QMessageBox.question(
            self, "Remove from list",
            f"Remove '{m.display_name}' from your image models list?\n\n"
            f"The file itself will NOT be deleted:\n{m.path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.image_models = [x for x in self.image_models if x.path != m.path]
            self.persistence.save_image_models(self.image_models)
            self._refresh_model_list()

    def _show_context_menu(self, pos):
        item = self.model_list.itemAt(pos)
        if not item:
            return
        m = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        ra = menu.addAction("Remove from List (keep file on disk)")
        chosen = menu.exec(self.model_list.mapToGlobal(pos))
        if chosen == ra:
            self.image_models = [x for x in self.image_models if x.path != m.path]
            self.persistence.save_image_models(self.image_models)
            self._refresh_model_list()

    def _unload_all(self):
        cleared = unload_all_image_models()
        if not cleared:
            QMessageBox.information(self, "Nothing to unload", "No image models are currently loaded in VRAM/RAM.")
            return
        QMessageBox.information(self, "Unloaded", f"Cleared {len(cleared)} model(s) from VRAM/RAM:\n" + "\n".join(cleared))

    # ---- source image (img2img) ----

    def _choose_source_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a Source Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        self._set_source_image(path)

    def _set_source_image(self, path):
        self.source_image_path = path
        self.source_image_label.setText(os.path.basename(path))
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                240, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
            self.source_thumb.setPixmap(scaled)

    def _use_output_as_source(self):
        if not self._last_image_path:
            return
        self.mode_combo.setCurrentText("Image to Image")
        self._set_source_image(self._last_image_path)

    # ---- generation ----

    def _generate(self):
        if self.worker is not None and self.worker.isRunning():
            return
        m = self._selected_model()
        if not m:
            QMessageBox.information(self, "No model selected", "Select an image model from the list first.")
            return
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.information(self, "No prompt", "Type a prompt first.")
            return

        is_img2img = (self.mode_combo.currentText() == "Image to Image")
        if is_img2img and not self.source_image_path:
            QMessageBox.information(self, "No source image", "Choose a source image first, or switch to Text to Image mode.")
            return

        seed = self.seed_spin.value()
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        self.progress_bar.setVisible(True)
        self.status_label.setText("Starting generation...")

        self.worker = ImageGenWorker(
            m, prompt, self.negative_prompt_edit.text(),
            self.steps_spin.value(), self.guidance_spin.value(),
            self.width_spin.value(), self.height_spin.value(), seed,
            source_image_path=self.source_image_path if is_img2img else None,
            strength=self.strength_spin.value(),
        )
        self.worker.loading.connect(self._on_loading)
        self.worker.finished_ok.connect(self._on_generate_ok)
        self.worker.finished_err.connect(self._on_generate_err)
        self.worker.start()

    def _on_loading(self, msg):
        self.status_label.setText(msg)

    def _on_generate_ok(self, path):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("\U0001F3A8 Generate")
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Saved to {path}")
        self._last_image_path = path
        self.use_output_as_source_btn.setEnabled(True)
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.preview_label.width() or 512, self.preview_label.height() or 512,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
            self.preview_label.setPixmap(scaled)

    def _on_generate_err(self, msg):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("\U0001F3A8 Generate")
        self.progress_bar.setVisible(False)
        self.status_label.setText("Generation failed.")
        QMessageBox.critical(self, "Generation failed", msg)

    def _open_output_folder(self):
        try:
            if sys.platform.startswith("win"):
                os.startfile(IMAGE_OUTPUT_DIR)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", IMAGE_OUTPUT_DIR])
            else:
                subprocess.Popen(["xdg-open", IMAGE_OUTPUT_DIR])
        except Exception as e:
            QMessageBox.warning(self, "Could not open folder", str(e))
