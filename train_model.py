from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "construction_safety_v27"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверь установку PyTorch.")

    # Читаем исходный файл с названиями классов.
    with (DATASET / "data.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Указываем точные пути, сохраняя исходный data.yaml.
    config["path"] = DATASET.as_posix()
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"

    for split in ("train", "valid", "test"):
        folder = DATASET / split / "images"
        if not folder.is_dir():
            raise FileNotFoundError(f"Не найдена папка: {folder}")

    local_config = DATASET / "data_local.yaml"
    with local_config.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

    print("Видеокарта:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(local_config),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        cache=False,
        project=str(ROOT / "runs"),
        name="yolo11n_baseline",
        patience=15,
        plots=True,
        seed=42,
    )


if __name__ == "__main__":
    main()