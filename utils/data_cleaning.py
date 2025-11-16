from pathlib import Path
from PIL import Image
import hashlib
import os


def clean_missing_labels(images_path, labels_path):
    images = list[Path](Path(images_path).glob("*.*"))
    removed = 0

    for img in images:
        label = Path(labels_path) / (img.stem + ".txt")
        if not label.exists():
            print(f"Eliminada imagen sin etiqueta: {img}")
            img.unlink()
            removed += 1

    print(f"✔ Proceso finalizado. Imágenes eliminadas: {removed}")


def clean_corrupt_labels(labels_path, images_path):
    removed = 0
    for label_file in Path(labels_path).glob("*.txt"):
        valid = True
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    valid = False
                    break
                try:
                    cls, x, y, w, h = map[float](parts)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        valid = False
                        break
                except ValueError:
                    valid = False
                    break
        
        if not valid:
            print(f"Etiqueta corrupta eliminada: {label_file}")
            
            img = Path(images_path) / (label_file.stem + ".jpg")
            if img.exists():
                img.unlink()
            label_file.unlink()
            removed += 1

    print(f"✔ Labels corruptos eliminados: {removed}")


def clean_corrupt_images(images_path, labels_path):
    removed = 0
    for img_path in Path(images_path).glob("*.*"):
        try:
            Image.open(img_path).verify()
        except Exception:
            print(f"Imagen dañada eliminada: {img_path}")
            label = Path(labels_path) / (img_path.stem + ".txt")
            if label.exists():
                label.unlink()
            img_path.unlink()
            removed += 1

    print(f"✔ Imágenes corruptas eliminadas: {removed}")


def get_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def remove_duplicates(images_path):
    seen = {}
    removed = 0

    for img in Path(images_path).glob("*.*"):
        h = get_hash(img)
        if h in seen:
            print(f"Duplicado eliminado: {img}")
            img.unlink()
            removed += 1
        else:
            seen[h] = img
    
    print(f"✔ Imágenes duplicadas eliminadas: {removed}")




def data_cleaning():
    train_images_path = os.path.join("data", "train", "images")
    train_labels_path = os.path.join("data", "train", "labels")
    validation_images_path = os.path.join("data", "validation", "images")
    validation_labels_path = os.path.join("data", "validation", "labels")

    clean_missing_labels(train_images_path, train_labels_path)
    clean_missing_labels(validation_images_path, validation_labels_path)

    clean_corrupt_labels(train_labels_path, train_images_path)
    clean_corrupt_labels(validation_labels_path, validation_images_path)

    clean_corrupt_images(train_images_path, train_labels_path)
    clean_corrupt_images(validation_images_path, validation_labels_path)

    remove_duplicates(train_images_path)
    remove_duplicates(validation_images_path)
