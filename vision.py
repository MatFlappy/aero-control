"""Inference helpers. No training or tracking logic lives here."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json

import cv2


LABELS = {
    "Hardhat": "Каска",
    "Mask": "Маска",
    "NO-Hardhat": "Без каски",
    "NO-Mask": "Без маски",
    "NO-Safety Vest": "Без жилета",
    "Person": "Человек",
    "Safety Cone": "Конус",
    "Safety Vest": "Жилет",
    "machinery": "Техника",
    "vehicle": "Транспорт",
}

VIOLATION_CLASSES = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}


def detect(model, frame, confidence, device):
    result = model.predict(frame, conf=confidence, device=device, imgsz=640, verbose=False)[0]
    rows = []

    for box in result.boxes:
        name = result.names[int(box.cls.item())]
        score = round(float(box.conf.item()), 3)
        rows.append(
            {
                "class": name,
                "label": LABELS.get(name, name),
                "confidence": score,
                "is_violation": name in VIOLATION_CLASSES,
                "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
            }
        )

    return result.plot(), rows


def find_violations(rows):
    return [row for row in rows if row.get("is_violation")]


def save_snapshot(root, original, annotated, metadata):
    folder = Path(root) / "storage" / "snapshots" / uuid4().hex
    folder.mkdir(parents=True)

    try:
        for name, frame in (("original.jpg", original), ("annotated.jpg", annotated)):
            ok, data = cv2.imencode(".jpg", frame)
            if not ok:
                raise IOError("Не удалось закодировать снимок")
            (folder / name).write_bytes(data.tobytes())

        payload = {
            **metadata,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "requires_review",
        }
        (folder / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        for path in folder.iterdir():
            path.unlink()
        folder.rmdir()
        raise

    return folder
