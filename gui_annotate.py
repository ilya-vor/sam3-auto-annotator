import sys
import os
from pathlib import Path
import yaml
import torch
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget, 
    QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QSlider, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor

# Import external components
from colors import colors
from model_loader import ModelLoaderWorker
from test_predictor import TestPredictorWorker
from predictor_worker import PredictorWorker

# --- Main GUI Window ---
class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 Auto-Annotator GUI")
        self.setMinimumWidth(1200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Side: Original Interface ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        
        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Path to data.yaml")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_file)
        path_layout.addWidget(QLabel("Dataset YAML:"))
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(btn_browse)
        left_layout.addLayout(path_layout)

        left_layout.addWidget(QLabel("\nClasses and Prompts (separate multiple prompts with semicolons ';'):"))
        # Initialize class thresholds dictionary
        self.class_thresholds = {}  # Store threshold for each class ID
        self.class_colors = {}      # Store color for each class ID
        
        # Create a widget to hold the class table and sliders
        self.classes_widget = QWidget()
        self.classes_layout = QVBoxLayout(self.classes_widget)
        
        # We'll populate this dynamically when classes are added
        left_layout.addWidget(self.classes_widget)
        
        btn_add_class = QPushButton("Add Class")
        btn_add_class.clicked.connect(self.add_class_row)
        btn_rem_class = QPushButton("Remove Selected Class")
        btn_rem_class.clicked.connect(self.remove_class_row)
        class_btns_layout = QHBoxLayout()
        class_btns_layout.addWidget(btn_add_class)
        class_btns_layout.addWidget(btn_rem_class)
        left_layout.addLayout(class_btns_layout)

        left_layout.addWidget(QLabel("\nConflict Pairs (Priority Class -> Secondary Class):"))
        self.conflict_table = QTableWidget(0, 3)
        self.conflict_table.setHorizontalHeaderLabels(["Priority ID", "Secondary ID", "Overlap Threshold (%)"])
        self.conflict_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        left_layout.addWidget(self.conflict_table)
        
        left_layout.addWidget(QLabel("\nNMS IoU Threshold:"))
        self.nms_slider = QSlider(Qt.Orientation.Horizontal)
        self.nms_slider.setRange(0, 100)
        self.nms_slider.setValue(25)
        self.nms_label = QLabel("0.25")
        self.nms_slider.valueChanged.connect(lambda v: self.nms_label.setText(f"{v/100:.2f}"))
        self.nms_slider.valueChanged.connect(self.update_test_nms)
        
        nms_layout = QHBoxLayout()
        nms_layout.addWidget(QLabel("NMS IoU:"))
        nms_layout.addWidget(self.nms_slider)
        nms_layout.addWidget(self.nms_label)
        left_layout.addLayout(nms_layout)
        
        left_layout.addWidget(QLabel("\nAnnotation Mode:"))
        self.mode_toggle = QPushButton("Replace Annotation (DELETES existing)")
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setStyleSheet("background-color: #e74c3c; color: white;")
        self.mode_toggle.clicked.connect(self.toggle_annotation_mode)
        left_layout.addWidget(self.mode_toggle)

        btn_add_conflict = QPushButton("Add Conflict Pair")
        btn_add_conflict.clicked.connect(self.add_conflict_row)
        btn_rem_conflict = QPushButton("Remove Selected Pair")
        btn_rem_conflict.clicked.connect(self.remove_conflict_row)
        conflict_btns_layout = QHBoxLayout()
        conflict_btns_layout.addWidget(btn_add_conflict)
        conflict_btns_layout.addWidget(btn_rem_conflict)
        left_layout.addLayout(conflict_btns_layout)

        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        self.overlay_widget = QWidget()
        self.overlay_widget.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        self.overlay_widget.hide()
        
        overlay_layout = QVBoxLayout(self.overlay_widget)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        overlay_progress_label = QLabel("Processing...")
        overlay_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_progress_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        overlay_layout.addWidget(overlay_progress_label)
        
        self.overlay_progress_bar = QProgressBar()
        self.overlay_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #2ecc71; }")
        self.overlay_progress_bar.setTextVisible(False)
        self.overlay_progress_bar.setFixedHeight(20)
        overlay_layout.addWidget(self.overlay_progress_bar)
        
        self.overlay_time_label = QLabel()
        self.overlay_time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay_time_label.setStyleSheet("color: white; font-size: 14px;")
        overlay_layout.addWidget(self.overlay_time_label)
        
        left_layout.addWidget(self.overlay_widget)

        self.btn_run = QPushButton("START PROCESSING")
        self.btn_run.setStyleSheet("background-color: #2ecc71; font-weight: bold; height: 40px;")
        self.btn_run.clicked.connect(self.run_processing)
        left_layout.addWidget(self.btn_run)
        
        main_layout.addWidget(left_container)

        # --- Right Side: Test Panel ---
        self.test_panel = QWidget()
        self.test_main_layout = QVBoxLayout(self.test_panel)
        self.test_main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Initial State: Empty Transparent Square
        self.test_placeholder = QLabel("Select data.yaml to enable testing")
        self.test_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_placeholder.setStyleSheet("border: 1px solid gray; background: transparent;")
        self.test_main_layout.addWidget(self.test_placeholder)
        
        # Hidden Setup Widget (will be shown after yaml selection)
        self.test_setup_widget = QWidget()
        self.test_setup_layout = QVBoxLayout(self.test_setup_widget)
        
        # Resetting layout for setup widget to be centered
        self.test_setup_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 10)
        self.n_spin.setValue(3)
        
        self.btn_gen_test = QPushButton("Generate Test Results")
        self.btn_gen_test.clicked.connect(self.run_test)
        
        self.test_setup_layout.addWidget(QLabel("Number of images (n):"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.test_setup_layout.addWidget(self.n_spin, alignment=Qt.AlignmentFlag.AlignCenter)
        self.test_setup_layout.addWidget(self.btn_gen_test, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Initialize attributes for test functionality
        self.last_selected_files = None
        self.last_n_images = None
        self.test_results = []
        self.current_test_idx = 0
        
        self.test_setup_widget.setVisible(False)
        self.test_main_layout.addWidget(self.test_setup_widget)

        self.test_results_widget = QWidget()
        self.test_results_layout = QVBoxLayout(self.test_results_widget)
        self.test_results_widget.setVisible(False)

        self.test_progress_text = QLabel()
        self.test_progress_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_display = QLabel("Preparing display...")
        self.img_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_display.setMinimumSize(400, 400)
        self.img_display.setStyleSheet("border: 1px solid gray; background: transparent;")
        self.test_results_layout.addWidget(self.img_display, 6)  # Give more space to image
        self.nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<")
        self.btn_update_test = QPushButton("Update Test Results")
        self.btn_update_test.clicked.connect(self.run_test)
        self.btn_update_test.setVisible(False)
        self.btn_next = QPushButton(">")
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_prev.clicked.connect(self.prev_test_img)
        self.btn_next.clicked.connect(self.next_test_img)
        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addWidget(self.btn_update_test)
        self.nav_layout.addWidget(self.btn_next)
        self.test_results_layout.addLayout(self.nav_layout)
        self.test_main_layout.addWidget(self.test_results_widget)
        main_layout.addWidget(self.test_panel, 1)

        # Load model in background
        self.model_worker = ModelLoaderWorker()
        self.model_worker.finished.connect(self.on_model_loaded)
        self.model_worker.error.connect(self.on_model_error)
        self.model_worker.start()

        # Connect path change to enable test panel
        self.path_edit.textChanged.connect(self.toggle_test_panel)
        self.predictor = None
        self.last_selected_files = None

    def on_model_loaded(self, success):
        if not success:
            self.on_model_error("Failed to load model")
            return
        self.predictor = self.model_worker.predictor
    
    def on_model_error(self, msg):
        QMessageBox.critical(self, "Model Load Error", f"Failed to load sam3.pt: {msg}")

    def toggle_test_panel(self):
        has_path = bool(self.path_edit.text().strip())
        self.test_placeholder.setVisible(not has_path)
        self.test_setup_widget.setVisible(has_path)
        self.test_results_widget.setVisible(False)

    def run_test(self):
        if self.predictor is None:
            QMessageBox.warning(self, "Warning", "Model is still loading, please wait...")
            return

        # Extract classes from the new widget structure
        classes = {}
        if hasattr(self, 'class_widgets'):
            for class_info in self.class_widgets:
                cid = class_info['cid']
                prompts = class_info['prompts_input'].text()
                classes[cid] = prompts
        
        conflicts = {}
        for i in range(self.conflict_table.rowCount()):
            try:
                prio = int(self.conflict_table.item(i, 0).text())
                sec = int(self.conflict_table.item(i, 1).text())
                thresh = float(self.conflict_table.item(i, 2).text())
                conflicts[(prio, sec)] = thresh
            except: continue

        config = {
            'dataset_path': self.path_edit.text(),
            'classes': classes,
            'conflicts': conflicts,
            'class_thresholds': self.class_thresholds,
            'nms_threshold': self.nms_slider.value() / 100.0
        }
        
        self.btn_gen_test.setEnabled(False)
        self.btn_gen_test.setText("Processing...")
        
        # Switch UI to results mode immediately (showing placeholder label)
        self.test_setup_widget.setVisible(False)
        self.test_placeholder.setVisible(False)
        self.test_results_widget.setVisible(True)
        self.img_display.setText("Processing SAM3...")

        self.test_worker = TestPredictorWorker(config, self.n_spin.value(), self.predictor, self.last_selected_files)
        self.test_worker.finished.connect(self.on_test_finished)
        self.test_worker.error.connect(self.on_test_error)
        self.test_worker.files_selected.connect(self.on_files_selected)
        self.test_worker.start()

    def on_files_selected(self, selected_files):
        if self.last_selected_files is None:
            self.last_selected_files = selected_files
            self.last_n_images = len(selected_files)
    
    def on_test_finished(self, results):
        self.btn_gen_test.setVisible(True)
        self.btn_gen_test.setEnabled(True)
        self.btn_gen_test.setText("Generate Test Results")
        self.btn_update_test.setVisible(True)
        self.test_results = results
        self.current_test_idx = 0
        self.update_test_image()

    def on_test_error(self, msg):
        self.btn_gen_test.setEnabled(True)
        self.btn_gen_test.setText("Generate Test Results")
        self.btn_update_test.setVisible(False)
        self.test_results_widget.setVisible(False)
        self.test_setup_widget.setVisible(True)
        self.test_placeholder.setVisible(False)
        QMessageBox.critical(self, "Test Error", msg)
    
    def update_test_nms(self):
        if hasattr(self, 'test_results') and self.test_results:
            self.update_test_image()


    def prev_test_img(self):
        if self.current_test_idx > 0:
            self.current_test_idx -= 1
            self.update_test_image()

    def next_test_img(self):
        if self.current_test_idx < len(self.test_results) - 1:
            self.current_test_idx += 1
            self.update_test_image()

    def update_detection_counts(self, cid=None):
        if not hasattr(self, 'test_results'):
            return
        
        if not hasattr(self, 'current_test_idx'):
            return
            
        # Check if we have any test results and if current index is valid
        if not self.test_results or self.current_test_idx >= len(self.test_results):
            return
        
        res = self.test_results[self.current_test_idx]
        dets = res['dets']
        
        for class_info in self.class_widgets:
            class_cid = class_info['cid']
            if cid is not None and class_cid != cid:
                continue
            
            threshold = self.class_thresholds.get(class_cid, 0.5)
            fitting_count = 0
            total_count = 0
            for det_cid, bbox in dets:
                if det_cid == class_cid:
                    total_count += 1
                    conf = res.get('confs', {}).get((det_cid, bbox), 1.0)
                    if conf >= threshold:
                        fitting_count += 1
            
            class_info['detection_count_label'].setText(f"Detections: {fitting_count}/{total_count}")
    
    def update_test_image(self):
        if not hasattr(self, 'test_results') or not self.test_results:
            return
            
        # Check if we have any test results and if current index is valid
        if not self.test_results or self.current_test_idx >= len(self.test_results):
            return
        
        res = self.test_results[self.current_test_idx]
        img = res['img'].copy()
        dets = res['dets']
        confs_dict = res.get('confs', {})
        
        iou_thresh_nms = self.nms_slider.value() / 100.0
        
        from utils import nms_per_class
        detections_nms = nms_per_class(dets, iou_thresh_nms)
        
        filtered_dets = []
        for det in detections_nms:
            if len(det) == 2:
                cid, bbox = det
            else:
                cid, bbox, conf = det
            threshold = self.class_thresholds.get(cid, 0.5)
            conf = confs_dict.get((cid, bbox), 1.0)
            if conf >= threshold:
                filtered_dets.append((cid, bbox))
        
        dets = filtered_dets
        
        # Update detection counts for each class (fitting/total for current image)
        class_counts = {}
        for cid, _ in dets:
            class_counts[cid] = class_counts.get(cid, 0) + 1
        
        class_total_counts = {}
        for det in dets:
            if len(det) == 2:
                cid, _ = det
            else:
                cid, _, _ = det
            class_total_counts[cid] = class_total_counts.get(cid, 0) + 1
        
        for class_info in self.class_widgets:
            class_cid = class_info['cid']
            count = class_counts.get(class_cid, 0)
            total = class_total_counts.get(class_cid, 0)
            class_info['detection_count_label'].setText(f"Detections: {count}/{total}")
        
        # Draw detections with class colors
        names_by_cid = {w['cid']: w['name'] for w in getattr(self, 'class_widgets', [])}
        for cid, (x1, y1, x2, y2) in dets:
            # Get class color (RGB format)
            color_rgb = self.class_colors.get(cid, (0, 255, 0))  # Default green if not found
            # Convert RGB to BGR for OpenCV
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
            try:
                # Add class ID and name to the label
                class_name = names_by_cid.get(cid, f"ID:{cid}")
                cv2.putText(img, class_name, (int(x1), int(max(0, y1-5))), 0, 0.5, color_bgr, 1)
            except Exception as e:
                print(f"Error drawing text: {e}")
                continue
        
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale pixmap to fit display
        scaled_pixmap = pixmap.scaled(self.img_display.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.img_display.setPixmap(scaled_pixmap)
        
        self.btn_prev.setEnabled(self.current_test_idx > 0)
        self.btn_next.setEnabled(self.current_test_idx < len(self.test_results) - 1)

    def browse_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML Files (*.yaml)")
        if file:
            self.path_edit.setText(file)
            self.auto_fill_classes(file)

    def auto_fill_classes(self, yaml_path):
        try:
            # Clear existing class widgets first
            self.clear_class_widgets()
            
            yaml_path = Path(yaml_path)
            with open(yaml_path, 'r') as f:
                data_cfg = yaml.safe_load(f)
            
            # Get names from yaml or scan labels
            names = data_cfg.get('names', [])
            if isinstance(names, dict):
                names = list(names.values())
            
            # Determine labels directory
            train_path = Path(data_cfg.get('train', 'train/images'))
            if not train_path.is_absolute():
                train_path = (yaml_path.parent / train_path).resolve()
            
            labels_dir = train_path.parent / "labels"
            
            found_classes = set()
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, 'r') as lf:
                        for line in lf:
                            parts = line.split()
                            if parts:
                                found_classes.add(int(parts[0]))
            
            if names:
                for i, name in enumerate(names):
                    self.add_class_row(str(i), name, name)
            elif found_classes:
                for cid in sorted(list(found_classes)):
                    self.add_class_row(str(cid), f"class_{cid}", "object")
            else:
                self.add_class_row("0", "class_0", "object")
                
        except Exception as e:
            print(f"Auto-fill failed: {e}")

    def add_class_row(self, cid=None, name=None, prompt=None):
        from colors import colors
        if cid is None:
            cid = "0"
        if name is None:
            name = "class_0"
        if prompt is None:
            prompt = "object"
        
        cid_int = int(cid)
        color_rgb = colors(cid_int)
        color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
        
        # Find next available CID if this one already exists
        existing_cids = [w['cid'] for w in getattr(self, 'class_widgets', [])]
        while cid_int in existing_cids:
            cid_int += 1
            cid = str(cid_int)
        
        # Create a widget for this class entry
        class_widget = QWidget()
        class_layout = QVBoxLayout(class_widget)
        
        # Class header with ID, name, color indicator, and detection count
        header_layout = QHBoxLayout()
        cid_edit = QLineEdit(str(cid))
        cid_edit.setMaximumWidth(50)
        name_edit = QLineEdit(name)
        name_edit.setMaximumWidth(100)
        color_indicator = QLabel("■")
        color_indicator.setStyleSheet(f"font-size: 20px; color: {color_hex};")
        detection_count_label = QLabel("Detections: 0")
        detection_count_label.setStyleSheet("font-size: 12px;")
        
        header_layout.addWidget(cid_edit)
        header_layout.addWidget(name_edit)
        header_layout.addWidget(color_indicator)
        header_layout.addWidget(detection_count_label)
        header_layout.addStretch()
        class_layout.addLayout(header_layout)
        
        # Remove button for this class row
        btn_remove_row = QPushButton("×")
        btn_remove_row.setMaximumWidth(40)
        btn_remove_row.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        btn_remove_row.clicked.connect(lambda: self.remove_specific_class(class_widget))
        header_layout.addWidget(btn_remove_row)
        
        # Prompts input
        prompts_label = QLabel("Prompts:")
        prompts_input = QLineEdit(prompt)
        class_layout.addWidget(prompts_label)
        class_layout.addWidget(prompts_input)
        
        # Confidence threshold slider
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Confidence Threshold:")
        threshold_slider = QSlider(Qt.Orientation.Horizontal)
        threshold_slider.setRange(0, 100)  # 0-100% which represents 0.0-1.0
        threshold_slider.setValue(50)  # Default 0.5
        threshold_value_label = QLabel("0.50")
        
        # Store references for later use
        if not hasattr(self, 'class_widgets'):
            self.class_widgets = []
        self.class_widgets.append({
            'widget': class_widget,
            'cid': cid_int,
            'cid_edit': cid_edit,
            'name': name,
            'name_edit': name_edit,
            'prompts_input': prompts_input,
            'color_indicator': color_indicator,
            'detection_count_label': detection_count_label,
            'threshold_slider': threshold_slider,
            'threshold_value_label': threshold_value_label
        })
        
        # Store initial threshold
        self.class_thresholds[cid_int] = 0.5
        
        # Set random color for this class (use deterministic color based on cid)
        self.class_colors[cid_int] = color_rgb
        
        # Update detection count based on current test results
        if hasattr(self, 'test_results') and self.test_results:
            self.update_detection_counts(cid_int)
        
        # Connect slider to update label and detection counts
        def update_threshold_label(value):
            threshold_value_label.setText(f"{value/100:.2f}")
            self.class_thresholds[cid_int] = value/100.0
            self.update_detection_counts(cid_int)
            self.update_test_image()
            
        threshold_slider.valueChanged.connect(update_threshold_label)
        
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(threshold_slider)
        threshold_layout.addWidget(threshold_value_label)
        class_layout.addLayout(threshold_layout)
        
        # Add to the classes layout
        self.classes_layout.addWidget(class_widget)

    def remove_specific_class(self, widget):
        if not hasattr(self, 'class_widgets') or not self.class_widgets:
            return
        
        # Find the widget info
        for i, info in enumerate(self.class_widgets):
            if info['widget'] == widget:
                # Remove from widgets list
                widget.deleteLater()
                self.class_widgets.pop(i)
                
                # Remove from thresholds and colors
                cid = info['cid']
                if cid in self.class_thresholds:
                    del self.class_thresholds[cid]
                if cid in self.class_colors:
                    del self.class_colors[cid]
                
                # Update detection counts for remaining classes
                self.update_detection_counts()
                self.update_test_image()
                return
    
    def remove_class_row(self):
        if not hasattr(self, 'class_widgets') or not self.class_widgets:
            return
        
        # Remove the last added class
        last_widget_info = self.class_widgets[-1]
        last_widget_info['widget'].deleteLater()
        self.class_widgets.pop()
        
        # Remove from thresholds and colors
        cid = last_widget_info['cid']
        if cid in self.class_thresholds:
            del self.class_thresholds[cid]
        if cid in self.class_colors:
            del self.class_colors[cid]
        
    def clear_class_widgets(self):
        # Remove all class widgets
        while self.classes_layout.count():
            child = self.classes_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        # Clear class_widgets list
        if hasattr(self, 'class_widgets'):
            self.class_widgets.clear()
            
        # Clear thresholds and colors
        self.class_thresholds.clear()
        self.class_colors.clear()

    def toggle_annotation_mode(self):
        if self.mode_toggle.isChecked():
            self.mode_toggle.setText("Merge Annotation (KEEPS non-recognized classes)")
            self.mode_toggle.setStyleSheet("background-color: #f39c12; color: white;")
        else:
            self.mode_toggle.setText("Replace Annotation (DELETES existing)")
            self.mode_toggle.setStyleSheet("background-color: #e74c3c; color: white;")

    def add_conflict_row(self):
        row = self.conflict_table.rowCount()
        self.conflict_table.insertRow(row)
        self.conflict_table.setItem(row, 0, QTableWidgetItem("0"))
        self.conflict_table.setItem(row, 1, QTableWidgetItem("1"))
        # Percent as 0.0 - 1.0 internally
        self.conflict_table.setItem(row, 2, QTableWidgetItem("0.25"))

    def remove_conflict_row(self):
        self.conflict_table.removeRow(self.conflict_table.currentRow())

    def run_processing(self):
        # Gather Classes from the new widget structure
        classes = {}
        if hasattr(self, 'class_widgets'):
            for class_info in self.class_widgets:
                cid = int(class_info['cid_edit'].text())
                name = class_info['name_edit'].text()
                prompts = class_info['prompts_input'].text()
                classes[cid] = prompts
        
        # If no classes from widgets, show error
        if not classes:
            QMessageBox.critical(self, "Error", "Please add at least one class.")
            return

        # Gather Conflicts
        conflicts = {}
        for i in range(self.conflict_table.rowCount()):
            try:
                prio = int(self.conflict_table.item(i, 0).text())
                sec = int(self.conflict_table.item(i, 1).text())
                thresh = float(self.conflict_table.item(i, 2).text())
                conflicts[(prio, sec)] = thresh
            except (AttributeError, ValueError):
                QMessageBox.critical(self, "Error", f"Invalid conflict data at row {i+1}")
                return

        # Get annotation mode
        annotation_mode = "merge" if self.mode_toggle.isChecked() else "replace"

        config = {
            'dataset_path': self.path_edit.text(),
            'classes': classes,
            'conflicts': conflicts,
            'annotation_mode': annotation_mode,
            'class_thresholds': self.class_thresholds,
            'nms_threshold': self.nms_slider.value() / 100.0
        }

        if not config['dataset_path']:
            QMessageBox.warning(self, "Warning", "Please specify dataset path.")
            return

        self.btn_run.setEnabled(False)
        self.overlay_widget.show()
        self.overlay_progress_bar.setRange(0, 100)
        self.overlay_progress_bar.setValue(0)
        self.overlay_time_label.setText("")
        
        config['nms_threshold'] = self.nms_slider.value() / 100.0
        
        self.worker = PredictorWorker(config, self.predictor)
        self.worker.progress.connect(self.on_progress)
        self.worker.progress.connect(self.overlay_progress_bar.setValue)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, value):
        if not hasattr(self, 'start_time'):
            self.start_time = __import__('time').time()
        elapsed = __import__('time').time() - self.start_time
        if value > 0:
            total_time = elapsed / (value / 100.0)
            remaining = total_time - elapsed
            minutes = int(remaining // 60)
            seconds = int(remaining % 60)
            self.overlay_time_label.setText(f"Estimated time remaining: {minutes}m {seconds}s")

    def on_finished(self, msg):
        self.btn_run.setEnabled(True)
        self.overlay_widget.hide()
        if hasattr(self, 'start_time'):
            del self.start_time
        QMessageBox.information(self, "Done", msg)

    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.overlay_widget.hide()
        if hasattr(self, 'start_time'):
            del self.start_time
        QMessageBox.critical(self, "Error", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Check if sam3.pt model exists
    if os.path.exists("sam3.pt"):
        window = AnnotationApp()
        window.show()
    else:
        QMessageBox.critical(None, "Error", "sam3.pt model is required at gui_annotate.py root directory.")
    
    sys.exit(app.exec())
