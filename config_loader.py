from pathlib import Path
import yaml

def load_dataset_config(data_yaml_path):
    """Load dataset configuration from YAML file"""
    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Файл {data_yaml_path} не найден.")

    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    train_images_dir = (data_yaml_path / data_cfg['train']).resolve()
    labels_dir = train_images_dir.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    return data_cfg, train_images_dir, labels_dir

def count_files(train_images_dir):
    """Count the number of files in the training directory"""
    total_files = sum(1 for p in train_images_dir.iterdir() if p.is_file())
    if total_files == 0:
        raise ValueError(f"В папке {train_images_dir} нет ни одного файла.")
    return total_files