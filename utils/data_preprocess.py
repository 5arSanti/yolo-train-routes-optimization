import cv2
import numpy as np
from pathlib import Path
import shutil

def apply_white_balance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    A = A - np.mean(A) + 128
    B = B - np.mean(B) + 128

    lab_balanced = cv2.merge([L, A, B])
    return cv2.cvtColor(lab_balanced, cv2.COLOR_LAB2BGR)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L_clahe = clahe.apply(L)

    lab_clahe = cv2.merge([L_clahe, A, B])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def apply_gamma(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def letterbox(img, new_size=512):
    h, w = img.shape[:2]
    scale = new_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((new_size, new_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    return canvas

def preprocess_image(img_path, output_path, size=512):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠ Imagen no cargada: {img_path}")
        return

    img = apply_white_balance(img)
    img = apply_clahe(img)
    img = apply_gamma(img, gamma=1.0)
    img = letterbox(img, new_size=size)

    cv2.imwrite(str(output_path), img)

def copy_label(label_path, output_path):
    shutil.copy(label_path, output_path)

def process_dataset(src_root="data", dst_root="data_preprocessed", size=512):
    splits = ["train", "validation"]

    for split in splits:
        print(f"🔧 Procesando {split}…")

        src_img_dir = Path(src_root) / split / "images"
        src_lbl_dir = Path(src_root) / split / "labels"

        dst_img_dir = Path(dst_root) / split / "images"
        dst_lbl_dir = Path(dst_root) / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in src_img_dir.glob("*.*"):
            out_img = dst_img_dir / img_path.name
            preprocess_image(img_path, out_img, size=size)

            label_path = src_lbl_dir / (img_path.stem + ".txt")
            if label_path.exists():
                copy_label(label_path, dst_lbl_dir / label_path.name)

        print(f"✔ {split} completado.\n")

    print("🎉 Dataset preprocesado creado en:", dst_root)

def data_preprocess(datap_path="data"):
    process_dataset(
        src_root="data",
        dst_root=datap_path,
        size=512
    )
