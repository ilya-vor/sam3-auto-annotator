import torch
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics.models.sam import SAM3SemanticPredictor

# --- Worker Thread for Model Loading ---
class ModelLoaderWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def run(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.predictor = SAM3SemanticPredictor(overrides=dict(
                conf=0.5, iou=0.7, task="segment", mode="predict",
                model="sam3.pt", device=device, verbose=False,
                imgsz=644, save=False, project=None, name=None, max_det=100
            ))
            self.finished.emit(self.predictor)
        except Exception as e:
            self.error.emit(str(e))