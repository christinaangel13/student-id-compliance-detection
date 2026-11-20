"""
Step 2: Train a custom YOLOv8 model to detect ID cards and lanyards.

Before running this:
1. Label your images (Roboflow.com is easiest) into YOLO format.
2. Arrange them into dataset/train/images, dataset/train/labels,
   dataset/valid/images, dataset/valid/labels as described in data.yaml.
3. pip install ultralytics

Run:
    python train_model.py
"""

from ultralytics import YOLO

def main():
    # Start from a pretrained nano model (fast, good for laptop webcams)
    model = YOLO("yolov8n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        patience=0,       # stop early if no improvement
        name="id_lanyard_detector",
    )

    # After training, your best weights will be saved at:
    # runs/detect/id_lanyard_detector/weights/best.pt
    # Copy that file into this project folder as "id_lanyard_best.pt"

if __name__ == "__main__":
    main()
