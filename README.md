# SAM3 Auto-Annotator GUI

PyQt6 GUI for automated image annotation using SAM3 (Segment Anything Model 3) by Ultralytics. Generates YOLO-format labels from text prompts.

## Requirements

- `sam3.pt` in the project root directory
- Python 3.8+
- PyQt6, PyTorch, OpenCV, Ultralytics, PyYAML

## Quick Start

```bash
pip install PyQt6 torch torchvision opencv-python ultralytics PyYAML
python gui_annotate.py
```

## Usage

1. **Select Dataset** — Browse for a `data.yaml` file (see format below). Classes auto-populate from the YAML.
2. **Configure Classes** — Set text prompts per class (semicolon `;` to separate multiple prompts). Adjust confidence thresholds per class.
3. **Conflict Pairs (optional)** — Define priority rules: when a priority class overlaps a secondary class beyond a threshold, the secondary detection is removed.
4. **NMS IoU Threshold** — Global IoU threshold for Non-Maximum Suppression per class.
5. **Annotation Mode** — *Replace* deletes all existing labels; *Merge* preserves labels for non-configured classes.
6. **Test Panel** — Preview results on a random sample of N images before full processing.
7. **Start Processing** — Runs batch annotation on all images in the dataset.

## Dataset YAML Format

```yaml
train: path/to/train/images
val: path/to/val/images   # optional
names:
  0: class_name_0
  1: class_name_1
```

## YOLO Label Output

Labels saved to `{images_dir}/../labels/{image_name}.txt`:

```
<class_id> <x_center> <y_center> <width> <height>
```

Values normalized to [0, 1].

## Architecture

| File | Role |
|---|---|
| `gui_annotate.py` | Main GUI window and application entry point |
| `model_loader.py` | Loads SAM3 model in background thread |
| `predictor_worker.py` | Batch annotation processing worker |
| `test_predictor.py` | Preview/test worker on a subset of images |
| `config_loader.py` | Dataset YAML parser |
| `utils.py` | Bounding box helpers (mask→bbox, IoU, NMS, YOLO conversion) |
| `colors.py` | Deterministic color palette per class ID |
| `class_definitions.py` | Class definition utilities |
| `annotation_saver.py` | Label file I/O |
| `image_processor.py` | Image loading/validation |
| `memory_manager.py` | Memory/cache management |
| `model_config.py` | SAM3 default parameters |

## Supported Image Formats

JPG, JPEG, PNG, BMP
