from config_loader import load_dataset_config, count_files
from model_config import configure_predictor
from image_processor import ImageProcessor

# ----------------------------------------------------------------------
# 1. Load dataset configuration
# ----------------------------------------------------------------------
data_cfg, train_images_dir, labels_dir = load_dataset_config("dataset_videos_unsafety_shoes/data.yaml")
print(f"Dataset configuration loaded successfully")

# ----------------------------------------------------------------------
# 2. Count files
# ----------------------------------------------------------------------
total_files = count_files(train_images_dir)
print(f"Найдено файлов: {total_files}")

# ----------------------------------------------------------------------
# 3. Load SAM3 with low global threshold (filtering will be done later)
#    Added optimizing parameters:
#    imgsz=644, save=False, project=None, name=None, max_det=100
# ----------------------------------------------------------------------
predictor, device = configure_predictor()
print(f"SAM3 загружена на {device} (оптимизированные настройки)")

# ----------------------------------------------------------------------
# 4. Process images
# ----------------------------------------------------------------------
processor = ImageProcessor(predictor)
processor.process_images(train_images_dir, labels_dir, total_files)

print("Готово!")