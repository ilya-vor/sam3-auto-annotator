import torch
from ultralytics.models.sam import SAM3SemanticPredictor

def configure_predictor():
    """Configure and load the SAM3 model with optimized settings"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    GLOBAL_CONF = 0.5   # very low threshold to get all possible candidates

    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=GLOBAL_CONF, iou=0.7, task="segment", mode="predict",
        model="sam3.pt", device=device, verbose=False,
        imgsz=644,          # multiple of 14, removes warning and speeds up
        save=False,         # DO NOT save results to runs/
        project=None,
        name=None,
        max_det=100
    ))
    
    return predictor, device