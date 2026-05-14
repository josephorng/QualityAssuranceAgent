from ultralytics import YOLO

# 1. Load your trained v26 model
model = YOLO("yolo_ui_model.pt") 

# 2. Export to ONNX with end2end enabled
success = model.export(
    format="onnx",
    end2end=True,      # Bakes post-processing into the graph
    imgsz=640,         # Set your input resolution
    simplify=True,     # Highly recommended: removes redundant ONNX nodes
    opset=12           # Standard opset for maximum compatibility
)

print(f"Export successful: {success}")