import cv2
import numpy as np
import torch
import yaml
import gc                     # для принудительной сборки мусора
from pathlib import Path
from tqdm import tqdm
from ultralytics.models.sam import SAM3SemanticPredictor

# ----------------------------------------------------------------------
# 1. Загрузка конфигурации датасета
# ----------------------------------------------------------------------
data_yaml_path = Path("dataset_videos_unsafety_shoes/data.yaml")
if not data_yaml_path.exists():
    raise FileNotFoundError(f"Файл {data_yaml_path} не найден.")

with open(data_yaml_path, 'r') as f:
    data_cfg = yaml.safe_load(f)

train_images_dir = (data_yaml_path / data_cfg['train']).resolve()
labels_dir = train_images_dir.parent / "labels"
labels_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. Подсчёт файлов
# ----------------------------------------------------------------------
total_files = sum(1 for p in train_images_dir.iterdir() if p.is_file())
if total_files == 0:
    raise ValueError(f"В папке {train_images_dir} нет ни одного файла.")
print(f"Найдено файлов: {total_files}")

# ----------------------------------------------------------------------
# 3. Загрузка SAM3 с низким глобальным порогом (фильтрация будет позже)
#    Добавлены оптимизирующие параметры:
#    imgsz=644, save=False, project=None, name=None, max_det=100
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_CONF = 0.5   # очень низкий порог, чтобы получить все возможные кандидаты

predictor = SAM3SemanticPredictor(overrides=dict(
    conf=GLOBAL_CONF, iou=0.7, task="segment", mode="predict",
    model="sam3.pt", device=device, verbose=False,
    imgsz=644,          # кратно 14, убирает warning и ускоряет
    save=False,         # НЕ сохранять результаты в runs/
    project=None,
    name=None,
    max_det=100
))
print(f"SAM3 загружена на {device} (оптимизированные настройки)")

# ----------------------------------------------------------------------
# 4. Только нужные классы и их пороги уверенности
# ----------------------------------------------------------------------
CLASS_NAMES_ALL = [
    "safety gloves",
    "safety shoes",
    "Welding mask",
    "hand",
    "shoes",
    "face"
]

# ID интересующих классов (последние три)
BARE_HAND_ID = 3
FOOT_WITHOUT_SHOES_ID = 4
HUMAN_FACE_ID = 5

TARGET_IDS = {BARE_HAND_ID, FOOT_WITHOUT_SHOES_ID, HUMAN_FACE_ID}

# Пороги уверенности для каждого целевого класса
CONF_THRESHOLDS = {
    BARE_HAND_ID: 0.5,
    HUMAN_FACE_ID: 0.5,
    FOOT_WITHOUT_SHOES_ID: 0.5
}

# ----------------------------------------------------------------------
# 5. Вспомогательные функции
# ----------------------------------------------------------------------
def mask_to_bbox(mask_np):
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x, y, x + w, y + h)

def xyxy_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return x_center, y_center, width, height

def compute_iou(box1, box2):
    """box = (x1,y1,x2,y2)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def nms_per_class(detections, iou_thresh=0.5):
    """Простой NMS внутри каждого класса"""
    by_class = {}
    for cid, box in detections:
        by_class.setdefault(cid, []).append(box)

    filtered = []
    for cid, boxes in by_class.items():
        if len(boxes) <= 1:
            filtered.extend([(cid, b) for b in boxes])
            continue
        keep = []
        boxes_list = list(boxes)
        while boxes_list:
            current = boxes_list.pop(0)
            keep.append(current)
            boxes_list = [b for b in boxes_list if compute_iou(current, b) < iou_thresh]
        filtered.extend([(cid, b) for b in keep])
    return filtered

# ----------------------------------------------------------------------
# 6. Основной цикл обработки (обёрнут в torch.inference_mode для отключения градиентов)
# ----------------------------------------------------------------------
IOU_THRESHOLD_NMS = 0.25   # для удаления дубликатов внутри класса

with torch.inference_mode():                     # оптимизация: отключаем градиенты
    for img_path in tqdm(train_images_dir.iterdir(), total=total_files, desc="Обработка"):
        if not img_path.is_file():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_h, img_w = img_bgr.shape[:2]

        # Оптимизация: передаём уже загруженное изображение (numpy BGR) вместо пути
        predictor.set_image(img_bgr)

        detections_px = []   # (class_id, bbox)

        # Обрабатываем только три целевых класса
        for class_id in TARGET_IDS:
            class_name = CLASS_NAMES_ALL[class_id]
            threshold = CONF_THRESHOLDS[class_id]

            try:
                results = predictor(text=[class_name])
                masks = results[0].masks
                if masks is None or len(masks.data) == 0:
                    continue

                # Получаем уверенности предсказаний, если они есть
                confs = None
                if results[0].boxes is not None and results[0].boxes.conf is not None:
                    confs = results[0].boxes.conf.cpu().numpy()

                for i, mask_tensor in enumerate(masks.data):
                    # Фильтруем по уверенности
                    # if confs is not None and confs[i] < threshold:
                    #     continue

                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                    bbox_px = mask_to_bbox(mask_np)
                    if bbox_px:
                        detections_px.append((class_id, bbox_px))

            except Exception as e:
                # Игнорируем ошибки для отдельных классов/изображений
                continue

        # Постобработка: NMS внутри каждого класса (удаляем дубликаты)
        detections_px = nms_per_class(detections_px, IOU_THRESHOLD_NMS)

        # Сохраняем аннотации в формате YOLO
        if detections_px:
            label_lines = []
            for class_id, (x1, y1, x2, y2) in detections_px:
                xc, yc, w_norm, h_norm = xyxy_to_yolo((x1, y1, x2, y2), img_w, img_h)
                label_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
            label_file = labels_dir / f"{img_path.stem}.txt"
            with open(label_file, 'w') as f:
                f.write("\n".join(label_lines))

        # Очистка памяти после каждой итерации
        torch.cuda.empty_cache()
        gc.collect()

print("Готово!")