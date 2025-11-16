import argparse
import os
import random
import shutil
import uuid
import zipfile
from pathlib import Path

from .create_data_yaml import create_data_yaml_from_set_and_classes


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}


def ensure_dir(path: str) -> None:
  if not os.path.exists(path):
    os.makedirs(path)


def unique_target_name(dst_images_dir: str, stem: str, ext: str) -> str:
  """Return a filename that does not collide in dst_images_dir.

  If "stem.ext" exists, append a short uuid suffix to stem.
  """
  candidate = f"{stem}{ext}"
  candidate_path = os.path.join(dst_images_dir, candidate)
  if not os.path.exists(candidate_path):
    return candidate
  # collision: add suffix
  suffix = uuid.uuid4().hex[:8]
  return f"{stem}_{suffix}{ext}"


def copy_image_and_label(src_img_path: Path, src_labels_dir: str, dst_images_dir: str, dst_labels_dir: str, id_map: dict[int, int] | None = None) -> None:
  stem = src_img_path.stem
  ext = src_img_path.suffix

  new_img_name = unique_target_name(dst_images_dir, stem, ext)
  new_stem, _ = os.path.splitext(new_img_name)

  dst_img_path = os.path.join(dst_images_dir, new_img_name)
  shutil.copy2(str(src_img_path), dst_img_path)

  src_label_path = os.path.join(src_labels_dir, f"{stem}.txt")
  if os.path.exists(src_label_path):
    dst_label_path = os.path.join(dst_labels_dir, f"{new_stem}.txt")
    if id_map is None or len(id_map) == 0:
      shutil.copy2(src_label_path, dst_label_path)
    else:
      # Remap class ids according to provided id_map
      try:
        with open(src_label_path, 'r') as rf, open(dst_label_path, 'w') as wf:
          for line in rf.readlines():
            line = line.strip()
            if not line:
              continue
            parts = line.split()
            if not parts:
              continue
            try:
              old_id = int(parts[0])
            except ValueError:
              # Non-standard line, copy as-is
              wf.write(line + "\n")
              continue
            new_id = id_map.get(old_id, old_id)
            parts[0] = str(new_id)
            wf.write(" ".join(parts) + "\n")
      except Exception:
        # Fallback to copy raw if any issue arises
        shutil.copy2(src_label_path, dst_label_path)


def copy_split_folder(split_root: str, dst_root: str, split_name: str, id_map: dict[int, int] | None = None) -> None:
  """Copy images/labels from a set split (train/valid/test) into dst_root.

  - split_root: e.g., dataset/set/train
  - dst_root: e.g., data
  - split_name: one of 'train', 'validation', or 'test'
  """
  src_images_dir = os.path.join(split_root, 'images')
  src_labels_dir = os.path.join(split_root, 'labels')

  if split_name == 'validation':
    # incoming folder from set is usually named 'valid'
    dst_split_name = 'validation'
  else:
    dst_split_name = split_name

  dst_images_dir = os.path.join(dst_root, dst_split_name, 'images')
  dst_labels_dir = os.path.join(dst_root, dst_split_name, 'labels')

  ensure_dir(dst_images_dir)
  ensure_dir(dst_labels_dir)

  img_files = [p for p in Path(src_images_dir).iterdir() if p.suffix in IMAGE_EXTENSIONS]
  for img_path in img_files:
    copy_image_and_label(img_path, src_labels_dir, dst_images_dir, dst_labels_dir, id_map)


def append_custom_data(custom_root: str, dst_root: str, train_pct: float = 0.8, seed: int | None = 42, id_map: dict[int, int] | None = None) -> None:
  """Split custom_data/images into train/validation and append into dst_root.

  This behaves like our train_val_split but collision-safe and appending-only.
  """
  if seed is not None:
    random.seed(seed)

  src_images_dir = os.path.join(custom_root, 'images')
  src_labels_dir = os.path.join(custom_root, 'labels')

  # Verificar que el directorio de imágenes existe antes de procesar
  if not os.path.isdir(src_images_dir):
    return

  dst_train_images = os.path.join(dst_root, 'train', 'images')
  dst_train_labels = os.path.join(dst_root, 'train', 'labels')
  dst_val_images = os.path.join(dst_root, 'validation', 'images')
  dst_val_labels = os.path.join(dst_root, 'validation', 'labels')
  ensure_dir(dst_train_images)
  ensure_dir(dst_train_labels)
  ensure_dir(dst_val_images)
  ensure_dir(dst_val_labels)

  imgs = [p for p in Path(src_images_dir).iterdir() if p.suffix in IMAGE_EXTENSIONS]
  if not imgs:
    return
  random.shuffle(imgs)
  split_idx = int(len(imgs) * float(train_pct))
  train_imgs = imgs[:split_idx]
  val_imgs = imgs[split_idx:]

  for img_path in train_imgs:
    copy_image_and_label(img_path, src_labels_dir, dst_train_images, dst_train_labels, id_map)
  for img_path in val_imgs:
    copy_image_and_label(img_path, src_labels_dir, dst_val_images, dst_val_labels, id_map)


def read_set_names(path_to_set_yaml: str) -> list[str]:
  import yaml
  if not os.path.exists(path_to_set_yaml):
    return []
  with open(path_to_set_yaml, 'r') as f:
    data = yaml.safe_load(f)
  names = data.get('names', []) or []
  return names if isinstance(names, list) else []


def read_classes_txt(path_to_classes_txt: str) -> list[str]:
  if not os.path.exists(path_to_classes_txt):
    return []
  names: list[str] = []
  with open(path_to_classes_txt, 'r') as f:
    for line in f.readlines():
      t = line.strip()
      if t:
        names.append(t)
  return names


def main():
  parser = argparse.ArgumentParser(description='Merge set and/or custom_data, and prepare data/ folders. Both set and custom_data are optional.')
  parser.add_argument('--set_dir', default='dataset/set', help='Directory where set contents live or will be extracted (optional)')
  parser.add_argument('--custom_dir', default='dataset/custom_data', help='Directory with images/ and labels/ (optional)')
  parser.add_argument('--data_dir', default='data', help='Target data root (data/train, data/validation, data/test)')
  parser.add_argument('--classes_txt', default='dataset/custom_data/classes.txt', help='Path to classes.txt of custom_data (optional)')
  parser.add_argument('--train_pct', default=0.8, type=float, help='Train percentage for custom_data split')
  parser.add_argument('--include_test', action='store_true', help='Copy set/test into data/test as well')

  args = parser.parse_args()

  set_yaml_path = os.path.join(args.set_dir, 'data.yaml')
  set_names = read_set_names(set_yaml_path)
  # Solo leer classes.txt si custom_dir existe
  custom_names = read_classes_txt(args.classes_txt) if os.path.isdir(args.custom_dir) else []

  final_names: list[str] = []
  seen = set()
  for n in set_names:
    if n not in seen:
      final_names.append(n)
      seen.add(n)
  for n in custom_names:
    if n not in seen:
      final_names.append(n)
      seen.add(n)

  set_id_map = {i: final_names.index(n) for i, n in enumerate(set_names) if n in final_names}
  custom_id_map = {i: final_names.index(n) for i, n in enumerate(custom_names) if n in final_names}

  set_train_root = os.path.join(args.set_dir, 'train')
  set_valid_root = os.path.join(args.set_dir, 'valid')
  set_test_root = os.path.join(args.set_dir, 'test')

  if os.path.isdir(set_train_root):
    copy_split_folder(set_train_root, args.data_dir, 'train', set_id_map)
  if os.path.isdir(set_valid_root):
    copy_split_folder(set_valid_root, args.data_dir, 'validation', set_id_map)
  if args.include_test and os.path.isdir(set_test_root):
    copy_split_folder(set_test_root, args.data_dir, 'test', set_id_map)

  if os.path.isdir(args.custom_dir):
    append_custom_data(args.custom_dir, args.data_dir, args.train_pct, id_map=custom_id_map)


if __name__ == '__main__':
  main()


