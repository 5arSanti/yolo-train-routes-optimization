import cv2
import os
from pathlib import Path

def draw_bboxes(images_path, labels_path, output_path, max_images=200):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images = []
    for ext in extensions:
        images.extend(list[Path](Path(images_path).glob(ext)))

    images = images[:max_images]

    for img_path in images:
        label_path = Path(labels_path) / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"Sin label => {img_path.name}")
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"ERROR leyendo imagen => {img_path}")
            continue

        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Label inválido => {label_path}")
                continue

            cls, x, y, bw, bh = map(float, parts)

            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, "person", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out_file = output_path / f"{img_path.stem}_preview.jpg"
        cv2.imwrite(str(out_file), img)

    print(f"✔ Previews generados en: {output_path}")

def preview_labels():
    train_images_path = os.path.join("data", "train", "images")
    train_labels_path = os.path.join("data", "train", "labels")
    validation_images_path = os.path.join("data", "validation", "images")
    validation_labels_path = os.path.join("data", "validation", "labels")
    
    draw_bboxes(train_images_path, train_labels_path, "previews/preview_train")

    draw_bboxes(validation_images_path, validation_labels_path, "previews/preview_val")

