import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from utils import mask_to_bbox, xyxy_to_yolo, compute_iou, nms_per_class

# --- Worker Thread for Testing (Single Batch) ---
class TestPredictorWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    files_selected = pyqtSignal(list)

    def __init__(self, config, n_images, predictor, selected_files=None):
        super().__init__()
        self.config = config
        self.n_images = n_images
        self.predictor = predictor
        self.selected_files = selected_files

    def run(self):
        try:
            yaml_path = Path(self.config['dataset_path'])
            yaml_dir = yaml_path.parent
            with open(yaml_path, 'r') as f:
                data_cfg = yaml.safe_load(f)
            
            train_images_dir = (yaml_dir / data_cfg['train']).resolve()
            files = [p for p in train_images_dir.iterdir() if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            if not files:
                self.error.emit("No images found in training directory.")
                return
            
            import random
            if self.selected_files is None:
                selected_files = random.sample(files, min(len(files), self.n_images))
                self.files_selected.emit(selected_files)
            else:
                selected_files = self.selected_files
            
            predictor = self.predictor

            results_data = []
            with torch.inference_mode():
                for img_path in selected_files:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None: continue
                    img_h, img_w = img_bgr.shape[:2]
                    predictor.set_image(img_bgr)
                    
                    detections_px = []
                    for class_id, prompts_str in self.config['classes'].items():
                        prompt_list = [p.strip() for p in prompts_str.split(';') if p.strip()]
                        for prompt in prompt_list:
                            try:
                                res = predictor(text=[prompt])
                                masks = res[0].masks
                                if masks is not None and res[0].boxes is not None and res[0].boxes.conf is not None:
                                    confs = res[0].boxes.conf.cpu().numpy()
                                    for i, m_tensor in enumerate(masks.data):
                                        m_np = m_tensor.cpu().numpy().astype(np.uint8) * 255
                                        bbox = mask_to_bbox(m_np)
                                        if bbox: 
                                            threshold = self.config.get('class_thresholds', {}).get(class_id, 0.5)
                                            conf = confs[i] if i < len(confs) else 1.0
                                            detections_px.append((int(class_id), bbox, conf))
                            except: continue
                    
                    confs_dict = {}
                    for item in detections_px:
                        if len(item) == 3:
                            cid, bbox, conf = item
                            confs_dict[(int(cid), tuple(bbox))] = float(conf)
                        else:  # len(item) == 2
                            cid, bbox = item
                            confs_dict[(int(cid), tuple(bbox))] = 1.0
                    
                    for (prio_id, sec_id), threshold in self.config['conflicts'].items():
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
                                final_sec.append((sid, sbox))
                        
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
                    
                    iou_thresh_nms = self.config.get('nms_threshold', 0.25)
                    detections_px = nms_per_class(detections_px, iou_thresh_nms)
                    
                    filtered_dets = []
                    for item in detections_px:
                        if len(item) == 3:
                            cid, bbox, conf = item
                            if conf >= self.config.get('class_thresholds', {}).get(cid, 0.5):
                                filtered_dets.append((cid, bbox))
                        else:  # len(item) == 2
                            cid, bbox = item
                            # For items without confidence, we include them by default
                            # or use a default threshold check
                            if 1.0 >= self.config.get('class_thresholds', {}).get(cid, 0.5):
                                filtered_dets.append((cid, bbox))
                    results_data.append({
                        'path': str(img_path),
                        'img': img_bgr,
                        'dets': filtered_dets,
                        'confs': confs_dict
                    })
            
            self.finished.emit(results_data)
        except Exception as e:
            self.error.emit(str(e))