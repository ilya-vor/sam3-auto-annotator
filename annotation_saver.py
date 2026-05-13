from pathlib import Path
from utils import xyxy_to_yolo

class AnnotationSaver:
    @staticmethod
    def save_annotations(detections_px, img_w, img_h, img_path, labels_dir):
        """Save annotations in YOLO format"""
        if detections_px:
            label_lines = []
            for class_id, (x1, y1, x2, y2) in detections_px:
                xc, yc, w_norm, h_norm = xyxy_to_yolo((x1, y1, x2, y2), img_w, img_h)
                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
            label_file = labels_dir / f"{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                f.write("\n".join(label_lines))
            return True
        return False