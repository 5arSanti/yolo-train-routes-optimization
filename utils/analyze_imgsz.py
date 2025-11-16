import os
from pathlib import Path
import statistics
import math
from PIL import Image


def gather_bbox_stats(images_dir, labels_dir, sample_limit=None):
    heights = []
    widths = []
    areas = []
    counts = 0

    exts = ['*.jpg', '*.jpeg', '*.png']
    img_paths = []
    for e in exts:
        img_paths.extend(list[Path](Path(images_dir).glob(e)))
    if sample_limit:
        img_paths = img_paths[:sample_limit]

    for img_path in img_paths:
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            continue

        lbl = Path(labels_dir) / (img_path.stem + ".txt")
        if not lbl.exists():
            continue

        with open(lbl, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    _, x, y, w, h = map(float, parts[:5])
                except ValueError:
                    continue
                bw = w * W
                bh = h * H
                ba = bw * bh
                widths.append(bw)
                heights.append(bh)
                areas.append(ba)
                counts += 1

    return {
        'count': counts,
        'widths': widths,
        'heights': heights,
        'areas': areas
    }

def summarize_stats(s):
    if s['count'] == 0:
        return None
    h = s['heights']
    w = s['widths']
    a = s['areas']
    stats = {
        'count': s['count'],
        'height_median': statistics.median(h),
        'height_mean': statistics.mean(h),
        'height_p25': percentile(h,25),
        'height_p75': percentile(h,75),
        'width_median': statistics.median(w),
        'area_median': statistics.median(a)
    }
    return stats

def percentile(data, p):
    if not data:
        return 0
    
    data_sorted = sorted(data)
    k = (len(data_sorted)-1) * (p/100)
    
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c-k)
    d1 = data_sorted[int(c)] * (k-f)
    return d0 + d1

def recommend_imgsz(height_median):
    if height_median < 16:
        return ">=1024 (recomendado 1280)"
    if height_median < 32:
        return "640 - 1024 (recomendado 1024)"
    if height_median < 64:
        return "512 - 640 (recomendado 640)"
    return "416 - 640 (recomendado 512 o 416 si GPU limitada)"

def analyze_imgsz():
    train_imgs = os.path.join("data", "train", "images")
    train_lbls = os.path.join("data", "train", "labels")
    val_imgs = os.path.join("data", "validation", "images")
    val_lbls = os.path.join("data", "validation", "labels")

    print("Analizando TRAIN...")
    s_train = gather_bbox_stats(train_imgs, train_lbls)
    
    print("Analizando VALIDATION...")
    s_val = gather_bbox_stats(val_imgs, val_lbls)

    sum_train = summarize_stats(s_train) if s_train else None
    sum_val = summarize_stats(s_val) if s_val else None

    print("\n--- RESULTADOS TRAIN ---")
    if sum_train:
        for k,v in sum_train.items():
            print(f"{k}: {v}")
        print("Recomendación imgsz (train):", recommend_imgsz(sum_train['height_median']))
    else:
        print("No se encontraron bboxes en TRAIN.")

    print("\n--- RESULTADOS VALIDATION ---")
    if sum_val:
        for k,v in sum_val.items():
            print(f"{k}: {v}")
        print("Recomendación imgsz (val):", recommend_imgsz(sum_val['height_median']))
    else:
        print("No se encontraron bboxes en VALIDATION.")
