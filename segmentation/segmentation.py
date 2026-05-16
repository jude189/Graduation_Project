
import os, math, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision import models
from sklearn.model_selection import train_test_split

# =========================
# 0) CONFIG
# =========================

# ── ISIC-2017 root (official splits respected as-is) ────────────────────────
ISIC_ROOT = "/kaggle/input/datasets/mahmudulhasantasin/isic-2017-original-dataset/isic 2017"

ISIC_SPLITS = {
    "train": (
        os.path.join(ISIC_ROOT, "ISIC-2017_Training_Data"),
        os.path.join(ISIC_ROOT, "ISIC-2017_Training_Part1_GroundTruth"),
    ),
    "val": (
        os.path.join(ISIC_ROOT, "ISIC-2017_Validation_Data"),
        os.path.join(ISIC_ROOT, "ISIC-2017_Validation_Part1_GroundTruth"),
    ),
    "test": (
        os.path.join(ISIC_ROOT, "ISIC-2017_Test_v2_Data"),
        os.path.join(ISIC_ROOT, "ISIC-2017_Test_v2_Part1_GroundTruth"),
    ),
}

# ── ISIC-2018 root ──────────────────────────────────────────────────────────
# Only the Training split has GT masks (Task1_Training_GroundTruth).
ISIC2018_ROOT     = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation"
ISIC2018_IMG_DIR  = os.path.join(ISIC2018_ROOT, "ISIC2018_Task1-2_Training_Input")
ISIC2018_MASK_DIR = os.path.join(ISIC2018_ROOT, "ISIC2018_Task1_Training_GroundTruth")

# ── PH2 root ────────────────────────────────────────────────────────────────
# Structure: PH2 Dataset images/<id>/<id>_Dermoscopic_Image/<id>.bmp
#                                    <id>_lesion/<id>_lesion.bmp
PH2_IMG_ROOT = "/kaggle/input/datasets/spacesurfer/ph2-dataset/PH2Dataset/PH2 Dataset images"

OUTPUT_DIR = "/kaggle/working/outputs_v4_merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE        = (384, 384)
BATCH_SIZE      = 8
EPOCHS          = 80
LR              = 3e-4
ENCODER_LR_MULT = 0.1
WEIGHT_DECAY    = 1e-4
SEED            = 42
NUM_WORKERS     = 2
PIN_MEMORY      = True
AUX_LOSS_START  = 0.4
AUX_LOSS_END    = 0.05
WARMUP_EPOCHS   = 5
SWA_START_FRAC  = 0.75
USE_TTA         = True
MIXUP_ALPHA     = 0.3
EARLY_PATIENCE  = 20

# Split ratios applied to the ISIC-2018 + PH2 pool only
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20   # = 1 - TRAIN_RATIO - VAL_RATIO

# CLAHE config
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE  = (8, 8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Source label constants (used for stratification of ISIC-2018 + PH2 pool)
SRC_ISIC2018 = 1
SRC_PH2      = 2

# =========================
# 1) SEED + DEVICE
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True

def get_device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch:", torch.__version__, " | Device:", d)
    if d.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    return d

# =========================
# 2) DATASET COLLECTION
# =========================

def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]


def collect_isic2017_by_split():
    """
    Return three (X, y) lists corresponding to ISIC-2017's OFFICIAL
    train / val / test splits.  No re-splitting is performed.
    """
    result = {}
    for split_name, (img_dir, mask_dir) in ISIC_SPLITS.items():
        img_paths = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in glob(os.path.join(img_dir, ext))
        )
        mask_paths = sorted(
            p for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in glob(os.path.join(mask_dir, ext))
        )
        mask_map = {}
        for m in mask_paths:
            s = _stem(m)
            mask_map[s] = m
            if s.endswith("_segmentation"):
                mask_map[s.replace("_segmentation", "")] = m

        X, y = [], []
        for img in img_paths:
            key = _stem(img)
            if key in mask_map:
                X.append(img)
                y.append(mask_map[key])

        print(f"  ISIC-2017 {split_name:5s}: {len(img_paths)} images → {len(X)} paired")
        result[split_name] = (X, y)

    return result          # {"train": (X,y), "val": (X,y), "test": (X,y)}


def collect_isic2018_pairs():
    """
    Collect ISIC-2018 (img, mask) pairs from the Training split only
    (the only split that has GT masks).
    """
    img_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob(os.path.join(ISIC2018_IMG_DIR, ext))
    )
    mask_paths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg")
        for p in glob(os.path.join(ISIC2018_MASK_DIR, ext))
    )

    mask_map = {}
    for m in mask_paths:
        s = _stem(m)
        mask_map[s] = m
        if s.endswith("_segmentation"):
            mask_map[s.replace("_segmentation", "")] = m

    X, y = [], []
    for img in img_paths:
        key = _stem(img)
        if key in mask_map:
            X.append(img)
            y.append(mask_map[key])

    missing = len(img_paths) - len(X)
    print(f"  ISIC-2018 training: {len(img_paths)} images → {len(X)} paired  (unmatched={missing})")
    return X, y


def collect_ph2_pairs():
    """
    Collect PH2 (img, mask) pairs.

    PH2 directory layout per case:
        <root>/<ID>/
            <ID>_Dermoscopic_Image/<ID>.bmp          ← RGB image
            <ID>_lesion/<ID>_lesion.bmp              ← binary mask
    """
    X, y = [], []
    case_dirs = sorted(
        d for d in glob(os.path.join(PH2_IMG_ROOT, "*"))
        if os.path.isdir(d)
    )
    missing = 0
    for case_dir in case_dirs:
        case_id   = os.path.basename(case_dir)
        img_path  = os.path.join(case_dir, f"{case_id}_Dermoscopic_Image", f"{case_id}.bmp")
        mask_path = os.path.join(case_dir, f"{case_id}_lesion",            f"{case_id}_lesion.bmp")

        if not os.path.exists(img_path):
            candidates  = glob(os.path.join(case_dir, "**", "*.bmp"), recursive=True)
            dermoscopic = [c for c in candidates if "Dermoscopic_Image" in c or "dermoscopic" in c.lower()]
            img_path    = dermoscopic[0] if dermoscopic else ""

        if not os.path.exists(mask_path):
            candidates = glob(os.path.join(case_dir, "**", "*lesion*.bmp"), recursive=True)
            mask_path  = candidates[0] if candidates else ""

        if img_path and mask_path and os.path.exists(img_path) and os.path.exists(mask_path):
            X.append(img_path)
            y.append(mask_path)
        else:
            missing += 1

    print(f"  PH2: {len(case_dirs)} case dirs → {len(X)} paired  (missing={missing})")
    return X, y


def build_18_ph2_splits(X_all, y_all, sources,
                        train_ratio=TRAIN_RATIO,
                        val_ratio=VAL_RATIO,
                        seed=SEED):
    """
    Stratified 70 / 10 / 20 split for the ISIC-2018 + PH2 pool.
    Stratification is on `sources` (SRC_ISIC2018=1 or SRC_PH2=2).

    Returns (train_X, train_y, val_X, val_y, test_X, test_y).
    """
    # First cut: train  vs  (val + test)
    X_tr, X_rest, y_tr, y_rest, s_tr, s_rest = train_test_split(
        X_all, y_all, sources,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=sources,
    )
    # Second cut: val  vs  test
    val_frac_of_rest = val_ratio / (val_ratio + TEST_RATIO)
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest,
        test_size=1.0 - val_frac_of_rest,
        random_state=seed,
        stratify=s_rest,
    )
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# =========================
# 3) CLAHE PREPROCESSING
# =========================
def apply_clahe(img_uint8, clip_limit=CLAHE_CLIP_LIMIT, tile_size=CLAHE_TILE_SIZE):
    """Apply CLAHE to the L channel of a LAB-converted uint8 RGB image."""
    lab  = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    cl   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# =========================
# 4) AUGMENTATION
# =========================
class AugPipeline:
    def __call__(self, img, mask):
        img, mask = img.copy(), mask.copy()

        if random.random() < 0.5:
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        if random.random() < 0.5:
            img, mask = np.flipud(img).copy(), np.flipud(mask).copy()
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()

        if random.random() < 0.5:
            alpha = random.uniform(0.7, 1.3)
            beta  = random.uniform(-0.15, 0.15)
            img   = np.clip(alpha * img + beta, 0.0, 1.0)

        if random.random() < 0.4:
            img_u8  = (img * 255).astype(np.uint8)
            img_hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.int32)
            img_hsv[..., 0] = (img_hsv[..., 0] + random.randint(-18, 18)) % 180
            img_hsv[..., 1] = np.clip(img_hsv[..., 1] + random.randint(-40, 40), 0, 255)
            img = cv2.cvtColor(
                np.clip(img_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB
            ).astype(np.float32) / 255.0

        if random.random() < 0.3:
            ksize  = random.choice([3, 5])
            img_u8 = (img * 255).astype(np.uint8)
            img    = cv2.GaussianBlur(img_u8, (ksize, ksize), 0).astype(np.float32) / 255.0

        if random.random() < 0.4:
            H, W = img.shape[:2]
            for _ in range(random.randint(1, 3)):
                rh = random.randint(H // 16, H // 6)
                rw = random.randint(W // 16, W // 6)
                r0 = random.randint(0, H - rh)
                c0 = random.randint(0, W - rw)
                img[r0:r0+rh, c0:c0+rw] = 0.0

        if random.random() < 0.4:
            img, mask = self._grid_distort(img, mask)

        return img, mask

    @staticmethod
    def _grid_distort(img, mask, num_steps=5, distort_limit=0.2):
        H, W = img.shape[:2]
        xs = np.linspace(0, W, num_steps + 1)
        ys = np.linspace(0, H, num_steps + 1)
        jx = np.random.uniform(
            -distort_limit * W / num_steps, distort_limit * W / num_steps,
            (num_steps + 1, num_steps + 1)).astype(np.float32)
        jy = np.random.uniform(
            -distort_limit * H / num_steps, distort_limit * H / num_steps,
            (num_steps + 1, num_steps + 1)).astype(np.float32)
        map_x = np.zeros((H, W), dtype=np.float32)
        map_y = np.zeros((H, W), dtype=np.float32)
        for i in range(num_steps):
            for j in range(num_steps):
                x0, x1 = int(xs[j]), int(xs[j + 1])
                y0, y1 = int(ys[i]), int(ys[i + 1])
                if x0 >= x1 or y0 >= y1:
                    continue
                py, px = np.mgrid[y0:y1, x0:x1]
                bx = np.interp(px, [x0, x1], [jx[i, j], jx[i, j + 1]])
                by = np.interp(py, [y0, y1], [jy[i, j], jy[i + 1, j]])
                map_x[y0:y1, x0:x1] = px.astype(np.float32) + bx.astype(np.float32)
                map_y[y0:y1, x0:x1] = py.astype(np.float32) + by.astype(np.float32)
        img_u8  = (img  * 255).astype(np.uint8)
        msk_u8  = (mask * 255).astype(np.uint8)
        img_out = cv2.remap(img_u8,  map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
        msk_out = cv2.remap(msk_u8,  map_x, map_y, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_REFLECT_101)
        return (img_out.astype(np.float32) / 255.0,
                (msk_out > 127).astype(np.float32))


_AUG = AugPipeline()

# =========================
# 5) DATASET
# =========================
class ISICDataset(Dataset):
    """Unified dataset for ISIC-2017 (JPG/PNG), ISIC-2018 (JPG/PNG), and PH2 (BMP)."""

    def __init__(self, X, y, size=(384, 384), augment=False):
        self.X       = X
        self.y       = y
        self.size    = size
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # ── Load image (supports JPG, PNG, BMP) ──────────────────────────────
        raw = cv2.imread(self.X[idx], cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"Cannot read image: {self.X[idx]}")
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = apply_clahe(img)
        img = img.astype(np.float32) / 255.0

        # ── Load mask ────────────────────────────────────────────────────────
        raw_mask = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"Cannot read mask: {self.y[idx]}")
        mask = cv2.resize(raw_mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            img, mask = _AUG(img, mask)

        img  = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
        mask = torch.tensor(mask[None], dtype=torch.float32)
        return img, mask


# =========================
# 6) MIXUP
# =========================
def mixup_batch(xb, yb, alpha=0.3):
    if alpha <= 0:
        return xb, yb
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1 - lam) * xb[idx], lam * yb + (1 - lam) * yb[idx]


# =========================
# 7) VISUALISATION
# =========================
def save_batch_preview(loader, out_path, n=3):
    xb, yb = next(iter(loader))
    xb, yb = xb[:n].numpy(), yb[:n].numpy()
    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img  = np.transpose(xb[i], (1, 2, 0))
        mask = yb[i, 0]
        plt.subplot(n, 3, 3*i+1); plt.imshow(img);               plt.title("Image");   plt.axis("off")
        plt.subplot(n, 3, 3*i+2); plt.imshow(mask, cmap="gray"); plt.title("Mask");    plt.axis("off")
        ov = img.copy(); ov[..., 0] = np.clip(ov[..., 0] + 0.5 * mask, 0, 1)
        plt.subplot(n, 3, 3*i+3); plt.imshow(ov);                plt.title("Overlay"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_history_plot(history, out_path):
    plt.figure(figsize=(16, 5))
    for k, (tr, va, title) in enumerate([
        ("loss", "val_loss", "Loss"),
        ("dice", "val_dice", "Dice"),
        ("iou",  "val_iou",  "IoU"),
    ]):
        plt.subplot(1, 3, k + 1)
        plt.plot(history[tr], label="train")
        plt.plot(history[va], label="val")
        plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_prediction_preview(model, loader, device, out_path, n=4, thr=0.5):
    model.eval()
    xb, yb = next(iter(loader))
    xb_dev = xb[:n].to(device)
    with torch.no_grad():
        logits = tta_predict(model, xb_dev) if USE_TTA else model(xb_dev)
        if isinstance(logits, tuple): logits = logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > thr).astype(np.float32)
    xb_np, yb_np = xb.cpu().numpy(), yb[:n].numpy()
    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img = np.transpose(xb_np[i], (1, 2, 0))
        plt.subplot(n, 4, 4*i+1); plt.imshow(img);                      plt.title("Image"); plt.axis("off")
        plt.subplot(n, 4, 4*i+2); plt.imshow(yb_np[i, 0], cmap="gray"); plt.title("GT");    plt.axis("off")
        plt.subplot(n, 4, 4*i+3); plt.imshow(probs[i, 0], cmap="gray"); plt.title("Prob");  plt.axis("off")
        plt.subplot(n, 4, 4*i+4); plt.imshow(preds[i, 0], cmap="gray"); plt.title("Bin");   plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# 8) METRICS + LOSS
# =========================
def dice_coeff(y_true, y_pred, eps=1e-6):
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
    inter  = (y_true * y_pred).sum(dim=1)
    return ((2 * inter + eps) / (y_true.sum(dim=1) + y_pred.sum(dim=1) + eps)).mean()


def iou_score(y_true, y_pred, eps=1e-6):
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
    inter  = (y_true * y_pred).sum(dim=1)
    union  = y_true.sum(dim=1) + y_pred.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()


class TverskyFocalLoss(nn.Module):
    """Tversky (FN-penalising) + Binary Focal loss."""
    def __init__(self, alpha=0.3, beta=0.7, focal_gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha; self.beta = beta
        self.focal_gamma = focal_gamma; self.eps = eps

    def _tversky(self, logits, y):
        p  = torch.sigmoid(logits)
        tp = (p * y).sum(dim=(1, 2, 3))
        fp = (p * (1 - y)).sum(dim=(1, 2, 3))
        fn = ((1 - p) * y).sum(dim=(1, 2, 3))
        return (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)

    def _focal(self, logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return ((1 - torch.exp(-bce)) ** self.focal_gamma * bce).mean()

    def forward(self, logits, y):
        return self._focal(logits, y) + (1.0 - self._tversky(logits, y).mean())


# =========================
# 8b) DOWNSAMPLE GT HELPER
# =========================
def downsample_gt(mask, size):
    down = F.interpolate(mask, size=size, mode="area")
    return (down > 0.5).float()


# =========================
# 9) BUILDING BLOCKS
# =========================
class ImageNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1))
    def forward(self, x): return (x - self.mean) / self.std


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(drop) if drop > 0.0 else nn.Identity(),
        )
    def forward(self, x): return self.net(x)


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class DualDilationASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates_a=(6, 12, 18), rates_b=(3, 6, 9), drop=0.1):
        super().__init__()
        def _br(r):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.branches_a = nn.ModuleList([_br(r) for r in rates_a])
        self.branches_b = nn.ModuleList([_br(r) for r in rates_b])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        n = 1 + len(rates_a) + len(rates_b) + 1
        self.project = nn.Sequential(
            nn.Conv2d(n * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout2d(drop))

    def forward(self, x):
        h, w  = x.shape[-2:]
        parts = ([self.b0(x)]
                 + [br(x) for br in self.branches_a]
                 + [br(x) for br in self.branches_b])
        parts.append(F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=False))
        return self.project(torch.cat(parts, dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(), nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, ch))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1))
        m = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a + m).view(x.size(0), x.size(1), 1, 1)


class SpatialAttentionCBAM(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=(ks - 1) // 2, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))


class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttentionCBAM()

    def forward(self, x): return self.sa(self.ca(x))


class UpBlock(nn.Module):
    """Bilinear upsample + 1×1 projection, CBAM on skip, SE after double-conv."""
    def __init__(self, in_ch, skip_ch, out_ch, drop=0.1):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(skip_ch)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, drop=drop)
        self.se   = SEBlock(out_ch)

    def forward(self, x, skip):
        g = self.up(x)
        if g.shape[-2:] != skip.shape[-2:]:
            g = F.interpolate(g, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.se(self.conv(torch.cat([g, self.cbam(skip)], dim=1)))


# =========================
# 10) FULL MODEL  (v4)
# =========================
class ResNetUNetV4(nn.Module):
    """ResNet-50 encoder + Dual-Dilation ASPP + CBAM/SE decoder + deep supervision."""

    def __init__(self, pretrained=True, drop=0.1):
        super().__init__()
        self.norm = ImageNetNorm()

        enc = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.e1   = enc.layer1   # /4   256 ch
        self.e2   = enc.layer2   # /8   512 ch
        self.e3   = enc.layer3   # /16 1024 ch
        self.e4   = enc.layer4   # /32 2048 ch

        self.bot_reduce = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.aspp     = DualDilationASPP(512, 512, drop=drop)
        self.cbam_bot = CBAM(512)

        self.d1 = UpBlock(512, 1024, 256, drop=drop)
        self.d2 = UpBlock(256,  512, 128, drop=drop)
        self.d3 = UpBlock(128,  256,  64, drop=drop)
        self.d4 = UpBlock( 64,   64,  64, drop=drop)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 32, drop=0.0),
            nn.Conv2d(32, 1, 1),
        )

        # Deep-supervision aux heads
        self.aux0 = nn.Conv2d(512, 1, 1)   # bottleneck  /32
        self.aux1 = nn.Conv2d(256, 1, 1)   # d1          /16
        self.aux2 = nn.Conv2d(128, 1, 1)   # d2          /8

    def encoder_params(self):
        return (list(self.stem.parameters()) + list(self.pool.parameters()) +
                list(self.e1.parameters())   + list(self.e2.parameters()) +
                list(self.e3.parameters())   + list(self.e4.parameters()))

    def decoder_params(self):
        enc_ids = {id(p) for p in self.encoder_params()}
        return [p for p in self.parameters() if id(p) not in enc_ids]

    def forward(self, x):
        x  = self.norm(x)
        s0 = self.stem(x)
        p  = self.pool(s0)
        s1 = self.e1(p)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)

        b  = self.cbam_bot(self.aspp(self.bot_reduce(s4)))
        d1 = self.d1(b,  s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)
        d4 = self.d4(d3, s0)
        out = self.final(d4)

        if self.training:
            return out, self.aux0(b), self.aux1(d1), self.aux2(d2)
        return out


# =========================
# 11) TTA (8-fold)
# =========================
def tta_predict(model, xb):
    model.eval()
    tfms = [
        (lambda x: x,                            lambda x: x),
        (lambda x: torch.flip(x, [3]),            lambda x: torch.flip(x, [3])),
        (lambda x: torch.flip(x, [2]),            lambda x: torch.flip(x, [2])),
        (lambda x: torch.flip(x, [2, 3]),         lambda x: torch.flip(x, [2, 3])),
        (lambda x: torch.rot90(x, 1, [2, 3]),     lambda x: torch.rot90(x, 3, [2, 3])),
        (lambda x: torch.rot90(x, 2, [2, 3]),     lambda x: torch.rot90(x, 2, [2, 3])),
        (lambda x: torch.rot90(x, 3, [2, 3]),     lambda x: torch.rot90(x, 1, [2, 3])),
        (lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]),
         lambda x: torch.rot90(torch.flip(x, [3]), 3, [2, 3])),
    ]
    preds = []
    with torch.no_grad():
        for fwd, inv in tfms:
            out = model(fwd(xb))
            if isinstance(out, tuple): out = out[0]
            preds.append(inv(out))
    return torch.stack(preds).mean(0)


# =========================
# 12) TRAIN / EVAL LOOPS
# =========================
def _aux_w(epoch, total): return AUX_LOSS_START + (AUX_LOSS_END - AUX_LOSS_START) * epoch / total


def run_epoch(model, loader, device, criterion, optimizer=None, train=False,
              epoch=1, total_epochs=EPOCHS):
    model.train() if train else model.eval()
    total_loss = total_dice = total_iou = 0.0
    aw = _aux_w(epoch, total_epochs)

    for xb, yb in tqdm(loader, desc="train" if train else "val", leave=True):
        xb, yb = xb.to(device), yb.to(device)
        if train and MIXUP_ALPHA > 0:
            xb, yb = mixup_batch(xb, yb, MIXUP_ALPHA)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            output = model(xb)
            if isinstance(output, tuple):
                logits   = output[0]
                aux_loss = 0.0
                for aux_logits in output[1:]:
                    aux_h, aux_w = aux_logits.shape[-2:]
                    yb_down  = downsample_gt(yb, (aux_h, aux_w))
                    aux_loss += aw * criterion(aux_logits, yb_down)
                loss = criterion(logits, yb) + aux_loss
            else:
                logits = output
                loss   = criterion(logits, yb)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            gt    = (yb > 0.5).float()
            total_dice += dice_coeff(gt, preds).item()
            total_iou  += iou_score(gt, preds).item()
        total_loss += loss.item()

    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n


def evaluate(model, loader, device, criterion, use_tta=True):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="test", leave=True):
            xb, yb = xb.to(device), yb.to(device)
            logits = tta_predict(model, xb) if use_tta else model(xb)
            if isinstance(logits, tuple): logits = logits[0]
            total_loss += criterion(logits, yb).item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_dice += dice_coeff(yb, preds).item()
            total_iou  += iou_score(yb, preds).item()
    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n


# =========================
# 13) LR SCHEDULER
# =========================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.opt      = optimizer
        self.warmup   = warmup_epochs
        self.total    = total_epochs
        self.eta_min  = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            scale = (epoch + 1) / self.warmup
        else:
            prog  = (epoch - self.warmup) / max(1, self.total - self.warmup)
            scale = (self.eta_min / self.base_lrs[-1] +
                     0.5 * (1 - self.eta_min / self.base_lrs[-1]) *
                     (1 + math.cos(math.pi * prog)))
        for pg, blr in zip(self.opt.param_groups, self.base_lrs):
            pg["lr"] = blr * scale


# =========================
# 14) MAIN
# =========================
def main():
    set_seed(SEED)
    device = get_device()

    # ── 1. Collect ISIC-2017 using its OFFICIAL splits ────────────────────────
    print("\n── Collecting ISIC-2017 pairs (official splits) …")
    isic17_splits = collect_isic2017_by_split()
    isic17_train_X, isic17_train_y = isic17_splits["train"]
    isic17_val_X,   isic17_val_y   = isic17_splits["val"]
    isic17_test_X,  isic17_test_y  = isic17_splits["test"]

    if not isic17_train_X:
        raise RuntimeError("No ISIC-2017 pairs found – check ISIC_ROOT path.")

    # ── 2. Collect ISIC-2018 + PH2 and apply 70/10/20 split ──────────────────
    print("\n── Collecting ISIC-2018 pairs (training split only) …")
    isic18_X, isic18_y = collect_isic2018_pairs()

    print("\n── Collecting PH2 pairs …")
    ph2_X, ph2_y = collect_ph2_pairs()

    if not isic18_X:
        raise RuntimeError("No ISIC-2018 pairs found – check ISIC2018_ROOT path.")
    if not ph2_X:
        raise RuntimeError("No PH2 pairs found – check PH2_IMG_ROOT path.")

    # Pool ISIC-2018 + PH2 and stratify on source label
    pool_X       = isic18_X + ph2_X
    pool_y       = isic18_y + ph2_y
    pool_sources = [SRC_ISIC2018] * len(isic18_X) + [SRC_PH2] * len(ph2_X)

    print(f"\nISIC-2018 + PH2 pool: {len(pool_X)} images"
          f"  (ISIC-2018={len(isic18_X)}, PH2={len(ph2_X)})")

    (p_train_X, p_train_y,
     p_val_X,   p_val_y,
     p_test_X,  p_test_y) = build_18_ph2_splits(pool_X, pool_y, pool_sources)

    # ── 3. Merge ISIC-2017 official splits with ISIC-2018+PH2 splits ──────────
    train_X = isic17_train_X + p_train_X
    train_y = isic17_train_y + p_train_y
    val_X   = isic17_val_X   + p_val_X
    val_y   = isic17_val_y   + p_val_y
    test_X  = isic17_test_X  + p_test_X
    test_y  = isic17_test_y  + p_test_y

    print(f"\nFinal split summary:")
    print(f"  Train : {len(train_X)}"
          f"  (ISIC-2017={len(isic17_train_X)}, ISIC-2018+PH2={len(p_train_X)})")
    print(f"  Val   : {len(val_X)}"
          f"  (ISIC-2017={len(isic17_val_X)}, ISIC-2018+PH2={len(p_val_X)})")
    print(f"  Test  : {len(test_X)}"
          f"  (ISIC-2017={len(isic17_test_X)}, ISIC-2018+PH2={len(p_test_X)})")

    # ── 4. Datasets + loaders ─────────────────────────────────────────────────
    train_ds = ISICDataset(train_X, train_y, size=IMG_SIZE, augment=True)
    val_ds   = ISICDataset(val_X,   val_y,   size=IMG_SIZE, augment=False)
    test_ds  = ISICDataset(test_X,  test_y,  size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,  BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    save_batch_preview(train_loader, os.path.join(OUTPUT_DIR, "train_batch_preview.png"))

    # ── 5. Model ──────────────────────────────────────────────────────────────
    model = ResNetUNetV4(pretrained=True, drop=0.1).to(device)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW([
        {"params": model.encoder_params(), "lr": LR * ENCODER_LR_MULT},
        {"params": model.decoder_params(), "lr": LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler  = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    criterion  = TverskyFocalLoss(alpha=0.3, beta=0.7, focal_gamma=2.0)

    swa_start  = int(EPOCHS * SWA_START_FRAC)
    swa_model  = AveragedModel(model)
    swa_sched  = SWALR(optimizer, swa_lr=1e-5)
    swa_active = False

    best_val_dice  = -1.0
    patience_count = 0
    history = {k: [] for k in ("loss", "val_loss", "dice", "val_dice", "iou", "val_iou")}
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")

    # ── 6. Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}  "
              f"(LR enc={optimizer.param_groups[0]['lr']:.2e} "
              f"dec={optimizer.param_groups[1]['lr']:.2e})")

        tr_loss, tr_dice, tr_iou = run_epoch(
            model, train_loader, device, criterion,
            optimizer, train=True, epoch=epoch, total_epochs=EPOCHS)
        va_loss, va_dice, va_iou = run_epoch(
            model, val_loader, device, criterion,
            train=False, epoch=epoch, total_epochs=EPOCHS)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
            swa_active = True
        else:
            scheduler.step(epoch)

        for k, v in zip(
            ("loss", "dice", "iou", "val_loss", "val_dice", "val_iou"),
            (tr_loss, tr_dice, tr_iou, va_loss, va_dice, va_iou),
        ):
            history[k].append(v)

        print(f"  train  loss={tr_loss:.4f}  dice={tr_dice:.4f}  iou={tr_iou:.4f}")
        print(f"  val    loss={va_loss:.4f}  dice={va_dice:.4f}  iou={va_iou:.4f}")

        if va_dice > best_val_dice:
            best_val_dice  = va_dice
            patience_count = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_dice": va_dice}, ckpt_path)
            print(f"  ✓ Saved best  (val_dice={va_dice:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_PATIENCE:
                print("Early stopping triggered.")
                break

    # ── 7. SWA BN update ──────────────────────────────────────────────────────
    if swa_active:
        print("\nUpdating SWA BatchNorm statistics …")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), os.path.join(OUTPUT_DIR, "swa_model.pt"))

    save_history_plot(history, os.path.join(OUTPUT_DIR, "training_curves.png"))

    # ── 8. Evaluate best checkpoint ───────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\nLoaded best ckpt: epoch {ckpt['epoch']}  val_dice={ckpt['val_dice']:.4f}")

    te_loss, te_dice, te_iou = evaluate(model, test_loader, device, criterion, use_tta=USE_TTA)
    print(f"\nTEST (best ckpt + TTA)  loss={te_loss:.4f}  dice={te_dice:.4f}  iou={te_iou:.4f}")

    # ── 9. Evaluate SWA model ─────────────────────────────────────────────────
    final_model = model
    if swa_active:
        swa_loss, swa_dice, swa_iou = evaluate(
            swa_model, test_loader, device, criterion, use_tta=USE_TTA)
        print(f"TEST (SWA    + TTA)     loss={swa_loss:.4f}  dice={swa_dice:.4f}  iou={swa_iou:.4f}")
        if swa_dice > te_dice:
            print("  → SWA model wins, using as final.")
            final_model = swa_model

    torch.save(final_model.state_dict(),
               os.path.join(OUTPUT_DIR, "final_model_state_dict.pt"))

    save_prediction_preview(model, test_loader, device,
                            os.path.join(OUTPUT_DIR, "test_pred_preview.png"), n=4)
    print(f"\nDONE. Outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
