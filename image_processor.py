import cv2
import numpy as np
import torch
from tqdm import tqdm
from utils import mask_to_bbox, compute_iou, nms_per_class
from class_definitions import CLASS_NAMES_ALL, TARGET_IDS, CONF_THRESHOLDS
from annotation_saver import AnnotationSaver
from memory_manager import MemoryManager

class ImageProcessor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.IOU_THRESHOLD_NMS = 0.25  # for removing duplicates within class
        self.annotation_saver = AnnotationSaver()
        self.memory_manager = MemoryManager()

    def process_images(self, train_images_dir, labels_dir, total_files):
        """Process all images in the training directory"""
        with torch.inference_mode():  # optimization: disable gradients
            for img_path in tqdm(train_images_dir.iterdir(), total=total_files, desc="Обработка"):
                if not img_path.is_file():
                    continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_h, img_w = img_bgr.shape[:2]

                # Optimization: pass already loaded image (numpy BGR) instead of path
                self.predictor.set_image(img_bgr)

                detections_px = []  # (class_id, bbox)

                # Process only three target classes
                for class_id in TARGET_IDS:
                    class_name = CLASS_NAMES_ALL[class_id]
                    threshold = CONF_THRESHOLDS[class_id]

                    try:
                        results = self.predictor(text=[class_name])
                        masks = results[0].masks
                        if masks is None or len(masks.data) == 0:
                            continue

                        # Get prediction confidence, if available
                        confs = None
                        if results[0].boxes is not None and results[0].boxes.conf is not None:
                            confs = results[0].boxes.conf.cpu().numpy()

                        for i, mask_tensor in enumerate(masks.data):
                            # Filter by confidence
                            # if confs is not None and confs[i] < threshold:
                            #     continue

                            mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                            bbox_px = mask_to_bbox(mask_np)
                            if bbox_px:
                                detections_px.append((class_id, bbox_px))

                    except Exception as e:
                        # Ignore errors for individual classes/images
                        continue

                # Postprocessing: NMS within each class (remove duplicates)
                detections_px = nms_per_class(detections_px, self.IOU_THRESHOLD_NMS)

                # Save annotations in YOLO format
                self.annotation_saver.save_annotations(detections_px, img_w, img_h, img_path, labels_dir)

                # Clear memory after each iteration
                self.memory_manager.clear_memory()