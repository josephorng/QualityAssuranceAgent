from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    model_path = root / "cua_mcp" / "get_UI" / "model.pt"
    image_path = root / "screenshot.png"
    output_path = root / "screenshot_result.png"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(model_path))

    names = getattr(model, "names", {}) or {}
    print("Model classes:")
    for class_id in sorted(names):
        print(f"{class_id}: {names[class_id]}")

    results = model.predict(str(image_path), conf=0.05, imgsz=640, iou=0.7, verbose=False)
    if not results:
        raise RuntimeError("No prediction results returned.")

    plotted = results[0].plot(labels=False, conf=False)
    ok = cv2.imwrite(str(output_path), plotted)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Saved prediction image to: {output_path}")


if __name__ == "__main__":
    main()
