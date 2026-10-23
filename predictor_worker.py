import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from utils import mask_to_bbox, xyxy_to_yolo, compute_iou, nms_per_class

class PredictorWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config, predictor=None):
        super().__init__()
        self.config = config
        self.predictor = predictor

    def run(self):
        try:
            yaml_path = Path(self.config['dataset_path'])
            yaml_dir = yaml_path.parent
            with open(yaml_path, 'r') as f:
                data_cfg = yaml.safe_load(f)

            train_images_dir = (yaml_dir / data_cfg['train']).resolve()
            labels_dir = train_images_dir.parent / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)

            files = list(train_images_dir.iterdir())
            total_files = len([p for p in files if p.is_file()])
            processed_count = 0

            if self.predictor is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                from ultralytics.models.sam import SAM3SemanticPredictor
                self.predictor = SAM3SemanticPredictor(overrides=dict(
                    conf=0.01, iou=0.7, task="segment", mode="predict",
                    model="sam3.pt", device=device, verbose=False,
                    imgsz=644, save=False, project=None, name=None, max_det=100
                ))

            with torch.inference_mode():
                for img_path in files:
                    if not img_path.is_file():
                        continue

                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        continue
                    img_h, img_w = img_bgr.shape[:2]

                    self.predictor.set_image(img_bgr)

                    detections_px = []

                    for class_id, prompts_str in self.config['classes'].items():
                        prompt_list = [p.strip() for p in prompts_str.split(';') if p.strip()]
                        for prompt in prompt_list:
                            try:
                                res = self.predictor(text=[prompt])
                                masks = res[0].masks
                                if masks is not None:
                                    confs = None
                                    if res[0].boxes is not None and res[0].boxes.conf is not None:
                                        confs = res[0].boxes.conf.cpu().numpy()
                                    
                                    for i, m_tensor in enumerate(masks.data):
                                        if confs is not None and confs[i] < 0.01:
                                            continue
                                        m_np = m_tensor.cpu().numpy().astype(np.uint8) * 255
                                        bbox = mask_to_bbox(m_np)
                                        if bbox: 
                                            conf = confs[i] if confs is not None else 0.5
                                            detections_px.append((int(class_id), bbox, conf))
                            except Exception:
                                continue

                    iou_thresh_nms = self.config.get('nms_threshold', 0.25)
                    detections_px = nms_per_class(detections_px, iou_thresh_nms)

                    # Handle conflict pairs
                    conflicts = self.config.get('conflicts', {})
                    for (prio_id, sec_id), threshold in conflicts.items():
                        # Handle both 2-element and 3-element tuples
                        prio_boxes = []
                        for item in detections_px:
                            if len(item) == 3:
                                cid, b, _ = item
                            else:  # len(item) == 2
                                cid, b = item
                            if cid == prio_id:
                                prio_boxes.append(b)
                        
                        sec_detections = []
                        for item in detections_px:
                            if len(item) == 3:
                                sid, sbox, conf = item
                            else:  # len(item) == 2
                                sid, sbox = item
                                conf = 1.0
                            if sid == sec_id:
                                sec_detections.append((sid, sbox, conf))
                        
                        final_sec = []
                        for sid, sbox, conf in sec_detections:
                            if not any(compute_iou(pbox, sbox) > threshold for pbox in prio_boxes):
                                final_sec.append((sid, sbox, conf))
                        
                        # Remove secondary class detections and add filtered ones
                        filtered_detections_px = []
                        for item in detections_px:
                            if len(item) == 3:
                                cid, bbox, conf = item
                            else:  # len(item) == 2
                                cid, bbox = item
                            if cid != sec_id:
                                filtered_detections_px.append(item)
                        detections_px = filtered_detections_px
                        detections_px.extend(final_sec)

                    filtered_detections = []
                    for i, det in enumerate(detections_px):
                        if len(det) == 3:
                            class_id, bbox, conf = det
                        else:
                            class_id, bbox = det
                            conf = 0.5
                        
                        threshold = self.config['class_thresholds'].get(class_id, 0.5)
                        if isinstance(threshold, float):
                            threshold = threshold
                        elif isinstance(threshold, int):
                            threshold = threshold / 100.0
                        else:
                            threshold = 0.5
                        
                        if conf >= threshold:
                            filtered_detections.append((class_id, bbox))
                    
                    class_id_to_dets = {}
                    for class_id, (x1, y1, x2, y2) in filtered_detections:
                        if class_id not in class_id_to_dets:
                            class_id_to_dets[class_id] = []
                        class_id_to_dets[class_id].append((x1, y1, x2, y2))
                    
                    if class_id_to_dets:
                        label_lines = []
                        for class_id in class_id_to_dets:
                            threshold = self.config['class_thresholds'].get(class_id, 0.5)
                            if isinstance(threshold, float):
                                threshold = threshold
                            elif isinstance(threshold, int):
                                threshold = threshold / 100.0
                            else:
                                threshold = 0.5
                            
                            for (x1, y1, x2, y2) in class_id_to_dets[class_id]:
                                xc, yc, w_norm, h_norm = xyxy_to_yolo((x1, y1, x2, y2), img_w, img_h)
                                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                        
                        label_file = labels_dir / f"{img_path.stem}.txt"
                        
                        annotation_mode = self.config.get('annotation_mode', 'replace')
                        
                        if annotation_mode == 'merge' and label_file.exists():
                            existing_lines = []
                            with open(label_file, 'r') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        parts = line.split()
                                        if len(parts) >= 1:
                                            existing_cid = int(parts[0])
                                            if existing_cid not in self.config['classes'].keys():
                                                existing_lines.append(line)
                            
                            label_lines.extend(existing_lines)
                        
                        with open(label_file, 'w') as f:
                            f.write("\n".join(label_lines))
                            if label_lines:
                                f.write("\n")

                    processed_count += 1
                    self.progress.emit(int(processed_count / total_files * 100))

            self.finished.emit("Processing complete!")
        except Exception as e:
            self.error.emit(str(e))