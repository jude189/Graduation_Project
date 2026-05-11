
import os, zipfile, shutil, gc
from glob import glob

import cv2
import numpy as np

# =============================================================================
# 0)  CONFIG
# =============================================================================

ISIC_ROOT = "/kaggle/input/datasets/mahmudulhasantasin/isic-2017-original-dataset/isic 2017"
ISIC_IMG_DIRS = {
    "train": os.path.join(ISIC_ROOT, "ISIC-2017_Training_Data"),
    "val":   os.path.join(ISIC_ROOT, "ISIC-2017_Validation_Data"),
    "test":  os.path.join(ISIC_ROOT, "ISIC-2017_Test_v2_Data"),
}

FOLDS_BASE_DIR = "/kaggle/input/datasets/engtala/new-five-fold-last"
VAL_PRED_DIR   = "/kaggle/input/datasets/engtala/new-val-pred"
TEST_PRED_DIR  = "/kaggle/input/datasets/engtala/new-test-pred"

N_FOLDS      = 5
FOLD_NAMES   = [f"fold_{i}_predictions" for i in range(N_FOLDS)]
WORKING_DIR  = "/kaggle/working"
JPEG_QUALITY = 92

os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================================================
# 1)  HELPERS
# =============================================================================

def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]

def _free_gb():
    return shutil.disk_usage(WORKING_DIR).free / 1e9

def _print_disk():
    print(f"  [disk] free: {_free_gb():.2f} GB")

def _safe_remove(path):
    try:
        if os.path.exists(path): os.remove(path)
    except OSError:
        pass

# =============================================================================
# 2)  BUILD ISIC-2017 IMAGE LOOKUP  bare_stem → image_path
# =============================================================================

def build_image_lookup():
    lookup = {}
    for split_name, img_dir in ISIC_IMG_DIRS.items():
        imgs = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png")
                      for p in glob(os.path.join(img_dir, ext)))
        for p in imgs:
            lookup[_stem(p)] = p
        print(f"  ISIC-2017 {split_name:5s}: {len(imgs)} images indexed")
    print(f"  Total ISIC-2017 images: {len(lookup)}")
    return lookup

# =============================================================================
# 3)  COLLECT ONLY "17_" MASKS  →  list of (bare_stem, mask_path)
# =============================================================================

def collect_17_masks(directory):
    all_masks = sorted(glob(os.path.join(directory, "17_*.png")) +
                       glob(os.path.join(directory, "17_*.jpg")))
    entries = []
    for m in all_masks:
        bare = _stem(m)[3:]   # strip "17_"
        entries.append((bare, m))
    return entries

def collect_fold_masks():
    entries = []
    for fold_name in FOLD_NAMES:
        fold_dir = os.path.join(FOLDS_BASE_DIR, fold_name)
        if not os.path.isdir(fold_dir):
            print(f"  WARNING: not found → {fold_dir}"); continue
        fold_entries = collect_17_masks(fold_dir)
        entries.extend(fold_entries)
        total_in_fold = len(glob(os.path.join(fold_dir, "*.png")))
        print(f"  {fold_name}: {total_in_fold} total masks  |  17_ kept: {len(fold_entries)}")
    print(f"  Total 17_ fold masks: {len(entries)}")
    return entries

def collect_flat_masks(directory, label):
    entries = collect_17_masks(directory)
    total   = len(glob(os.path.join(directory, "*.png")))
    print(f"  {label}: {total} total masks  |  17_ kept: {len(entries)}")
    return entries

# =============================================================================
# 4)  MASK CLEANING
# =============================================================================

def keep_largest_blob(gray):
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(gray)
    if contours:
        cv2.drawContours(clean, [max(contours, key=cv2.contourArea)],
                         -1, 255, thickness=cv2.FILLED)
    return clean

# =============================================================================
# 5)  MULTIPLY
# =============================================================================

def multiply_image_with_mask(img_path, clean_mask_gray):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None: return None
    h, w = img.shape[:2]
    if clean_mask_gray.shape != (h, w):
        clean_mask_gray = cv2.resize(clean_mask_gray, (w, h),
                                     interpolation=cv2.INTER_NEAREST)
    mask_f = (clean_mask_gray / 255.0).astype(np.float32)
    return (img.astype(np.float32) * mask_f[..., np.newaxis]).astype(np.uint8)

# =============================================================================
# 6)  ZIP WRITER
# =============================================================================

def write_multiplied_zip(split_label, entries, image_lookup, out_path):
    """
    entries      : list of (bare_stem, mask_path)
    image_lookup : bare_stem → original image path
    Output file names inside zip: isic17__<bare_stem>.jpg
    """
    tmp = out_path + ".tmp"
    _safe_remove(tmp)
    _print_disk()
    saved = missing_img = read_err = 0

    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for bare_stem, mask_path in entries:
                img_path = image_lookup.get(bare_stem)
                if img_path is None:
                    missing_img += 1; continue

                raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if raw_mask is None:
                    read_err += 1; continue

                clean_mask = keep_largest_blob(raw_mask)
                multiplied = multiply_image_with_mask(img_path, clean_mask)
                if multiplied is None:
                    read_err += 1; continue

                ok, buf = cv2.imencode(".jpg", multiplied,
                                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    read_err += 1; continue

                zf.writestr(f"isic17__{bare_stem}.jpg", buf.tobytes())
                saved += 1

                if saved % 100 == 0 and _free_gb() < 1.0:
                    print(f"  WARNING: <1 GB free — stopping at {saved}")
                    break

        os.replace(tmp, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  [{split_label}]  saved={saved}  "
              f"missing_img={missing_img}  read_err={read_err}  "
              f"→ {out_path}  ({size_mb:.1f} MB)")

    except OSError as e:
        _safe_remove(tmp)
        raise RuntimeError(f"Failed writing {out_path}: {e}") from e

# =============================================================================
# 7)  MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  POST-PROCESSING  ·  ISIC-2017 ONLY")
    print("=" * 65)
    _print_disk()

    print("\n── Building ISIC-2017 image lookup …")
    image_lookup = build_image_lookup()

    print("\n── Collecting 17_ predicted masks …")
    train_entries = collect_fold_masks()
    val_entries   = collect_flat_masks(VAL_PRED_DIR,  "val predictions")
    test_entries  = collect_flat_masks(TEST_PRED_DIR, "test predictions")

    print(f"\n── Summary of entries to process:")
    print(f"   train (folds): {len(train_entries)}  (expected 2000)")
    print(f"   val           : {len(val_entries)}   (expected 150)")
    print(f"   test          : {len(test_entries)}  (expected 600)")

    print("\n── Writing multiplied zips …\n")

    write_multiplied_zip("TRAIN", train_entries, image_lookup,
                         os.path.join(WORKING_DIR, "train_multiplied.zip"))
    gc.collect()

    write_multiplied_zip("VAL", val_entries, image_lookup,
                         os.path.join(WORKING_DIR, "val_multiplied.zip"))
    gc.collect()

    write_multiplied_zip("TEST", test_entries, image_lookup,
                         os.path.join(WORKING_DIR, "test_multiplied.zip"))
    gc.collect()

    print("\n" + "=" * 65)
    print("  DONE.")
    print("    train_multiplied.zip   val_multiplied.zip   test_multiplied.zip")
    print("  File names inside:  isic17__<stem>.jpg")
    _print_disk()
    print("=" * 65)

if __name__ == "__main__":
    main()
