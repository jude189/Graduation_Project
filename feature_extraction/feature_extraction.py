
import os
import csv
import math
import warnings
import argparse

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy import stats
from scipy.fft import fft as scipy_fft
from skimage.measure import regionprops, label as sklabel
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

warnings.filterwarnings("ignore")

import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)
logging.getLogger("pykwalify").setLevel(logging.ERROR)

try:
    import SimpleITK as sitk
    from radiomics import featureextractor as _rx_extractor
    _RADIOMICS_OK = True
except ImportError:
    _RADIOMICS_OK = False
    print("[WARN] pyradiomics not available — radiomics features will be skipped")


# =============================================================================
#  DEFAULTS & CONSTANTS
# =============================================================================

DEFAULT_MASKED_ROOT = "data/masked_images"
DEFAULT_OUTPUT_CSV  = "outputs/features_v3.csv"
DEFAULT_MAX_DIM     = 512
DEFAULT_WORKERS     = max(1, cpu_count() - 1)

RADIOMICS_CLASSES   = ["firstorder", "shape2D", "glcm", "glrlm", "glszm", "ngtdm"]
N_FOURIER_COEFS     = 6
COLOR6_THRESHOLD    = 0.01
OCTANT_ABRUPT_MULT  = 1.5
MIN_MASK_PIXELS     = 50

# LBP (match deployed model)
LBP_RADIUS   = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD   = "uniform"
LBP_N_BINS   = LBP_N_POINTS + 2

# Gabor (match deployed model)
GABOR_PARAMS = [
    (0.10,   0), (0.10,  45), (0.10,  90), (0.10, 135),
    (0.20,   0), (0.20,  45), (0.20,  90), (0.20, 135),
]
GABOR_KSIZE, GABOR_SIGMA, GABOR_GAMMA, GABOR_PSI = 21, 3.0, 1.0, 0.0

# GLCM (match deployed model)
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES    = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_LEVELS    = 64
GLCM_PROPS     = ["contrast", "correlation", "energy",
                  "homogeneity", "ASM", "dissimilarity"]

# Dots / globules
DOTS_MIN_AREA_PX      = 8
DOTS_MAX_AREA_FRAC    = 0.05
DOTS_MIN_CIRCULARITY  = 0.55

# V3 structure detectors
MILIA_V_MIN        = 0.88   # very bright
MILIA_S_MAX        = 0.20   # low saturation
MILIA_MIN_PX       = 10
MILIA_MAX_FRAC     = 0.03   # < 3% of lesion each
MILIA_MIN_CIRC     = 0.65

COMEDO_V_MAX       = 0.12   # very dark
COMEDO_MIN_PX      = 15
COMEDO_MAX_FRAC    = 0.04
COMEDO_MIN_CIRC    = 0.60

BLOTCH_V_MAX       = 0.25   # dark
BLOTCH_MIN_FRAC    = 0.02   # must be > 2% of lesion

WHITE_V_MIN        = 0.90   # very bright white
WHITE_S_MAX        = 0.10
WHITE_MIN_PX       = 8
WHITE_MAX_FRAC     = 0.05
WHITE_ELONGATED_RATIO = 0.40   # minor/major < this → elongated

PEPPER_V_MIN, PEPPER_V_MAX = 0.25, 0.60
PEPPER_S_MAX       = 0.25

SCAR_V_MIN         = 0.82
SCAR_S_MAX         = 0.18

LACUNAE_H_MAX      = 0.05   # red hue (low end)
LACUNAE_H_MIN_HI   = 0.90   # red hue (high end)
LACUNAE_S_MIN      = 0.40
LACUNAE_V_MIN      = 0.10
LACUNAE_V_MAX      = 0.45
LACUNAE_MIN_CIRC   = 0.50
LACUNAE_MIN_PX     = 15
LACUNAE_MAX_FRAC   = 0.15

DERMOSCOPY_FOV_MM  = 20.0   # standard field of view for diam_mm estimate


# =============================================================================
#  PRE-BUILT GABOR KERNELS
# =============================================================================

def _build_gabor_kernels():
    kernels = []
    for freq, theta_deg in GABOR_PARAMS:
        k = cv2.getGaborKernel(
            (GABOR_KSIZE, GABOR_KSIZE), GABOR_SIGMA,
            np.deg2rad(theta_deg), 1.0 / freq,
            GABOR_GAMMA, GABOR_PSI, ktype=cv2.CV_32F,
        )
        kernels.append((k, freq, theta_deg))
    return kernels

_GABOR_KERNELS = _build_gabor_kernels()


# =============================================================================
#  GENERIC HELPERS
# =============================================================================

def _stem(p):
    return os.path.splitext(os.path.basename(p))[0]

def _safe(v):
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return 0.0

def _resize_img(rgb, mask, max_dim):
    if max_dim <= 0:
        return rgb, mask
    h, w  = rgb.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return rgb, mask
    nh, nw = int(h * scale), int(w * scale)
    return (
        cv2.resize(rgb,  (nw, nh), interpolation=cv2.INTER_AREA),
        cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST),
    )

def derive_mask(roi_rgb: np.ndarray) -> np.ndarray:
    fg = (roi_rgb.max(axis=2) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

def _get_largest_region(mask):
    labeled = sklabel(mask.astype(np.uint8))
    props   = regionprops(labeled)
    if not props:
        return None
    return max(props, key=lambda p: p.area)

def _load_colour_views(bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv = hsv_raw.copy()
    hsv[:, :, 0] /= 180.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0
    lab_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = lab_raw.copy()
    lab[:, :, 0] =  lab_raw[:, :, 0] * (100.0 / 255.0)
    lab[:, :, 1] =  lab_raw[:, :, 1] - 128.0
    lab[:, :, 2] =  lab_raw[:, :, 2] - 128.0
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return rgb, hsv, lab, grey

def _rotate_mask_cv(mask, angle_deg):
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(mask.astype(np.float32), M, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated > 0.5

def _rotate_channel_cv(channel, angle_deg):
    h, w = channel.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(channel.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


# =============================================================================
#  V1 — ASYMMETRY (centre fold)
# =============================================================================

def _shape_asymmetry(mask):
    m  = mask.astype(np.uint8)
    n  = float(m.sum()) + 1e-8
    lr = float(np.logical_xor(m, np.fliplr(m)).sum()) / n
    ud = float(np.logical_xor(m, np.flipud(m)).sum()) / n
    return lr, ud, (lr + ud) / 2.0

def _color_channel_asymmetry(channel, mask):
    ch = channel * mask
    n  = float(mask.sum()) + 1e-8
    lr = float(np.abs(ch - np.fliplr(ch))[mask].sum() / n)
    ud = float(np.abs(ch - np.flipud(ch))[mask].sum() / n)
    return lr, ud

def extract_asymmetry_centre(mask, hsv) -> dict:
    feats = {}
    lr, ud, score = _shape_asymmetry(mask)
    feats["abcde_asym_lr"]    = lr
    feats["abcde_asym_ud"]    = ud
    feats["abcde_asym_score"] = score
    for idx, ch in enumerate(["H", "S", "V"]):
        lr_c, ud_c = _color_channel_asymmetry(hsv[:, :, idx], mask)
        feats[f"abcde_color_asym_{ch}_lr"] = lr_c
        feats[f"abcde_color_asym_{ch}_ud"] = ud_c
        feats[f"abcde_color_asym_{ch}"]    = (lr_c + ud_c) / 2.0
    return feats


# =============================================================================
#  V1 — ASYMMETRY (principal-axis fold)
# =============================================================================

def extract_asymmetry_principal_axis(mask, hsv) -> dict:
    zero_keys = (
        ["abcde_asym_pa_lr", "abcde_asym_pa_ud", "abcde_asym_pa_score",
         "abcde_asym_pa_axis_score"]
        + [f"abcde_asym_pa_color_{ch}_{ax}" for ch in ["H","S","V"] for ax in ["lr","ud"]]
        + [f"abcde_asym_pa_color_{ch}" for ch in ["H","S","V"]]
    )
    feats = {k: 0.0 for k in zero_keys}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    angle_deg = np.degrees(prop.orientation)
    rot_mask  = _rotate_mask_cv(mask, -angle_deg)
    n = float(rot_mask.sum()) + 1e-8
    if n < MIN_MASK_PIXELS:
        return feats
    lr = float(np.logical_xor(rot_mask, np.fliplr(rot_mask)).sum()) / n
    ud = float(np.logical_xor(rot_mask, np.flipud(rot_mask)).sum()) / n
    feats["abcde_asym_pa_lr"]         = lr
    feats["abcde_asym_pa_ud"]         = ud
    feats["abcde_asym_pa_score"]      = (lr + ud) / 2.0
    feats["abcde_asym_pa_axis_score"] = float(int(lr > 0.10) + int(ud > 0.10))
    for idx, ch in enumerate(["H", "S", "V"]):
        ch_rot = _rotate_channel_cv(hsv[:, :, idx], -angle_deg)
        masked = ch_rot * rot_mask
        n_px   = float(rot_mask.sum()) + 1e-8
        lr_c   = float(np.abs(masked - np.fliplr(masked))[rot_mask].sum() / n_px)
        ud_c   = float(np.abs(masked - np.flipud(masked))[rot_mask].sum() / n_px)
        feats[f"abcde_asym_pa_color_{ch}_lr"] = lr_c
        feats[f"abcde_asym_pa_color_{ch}_ud"] = ud_c
        feats[f"abcde_asym_pa_color_{ch}"]    = (lr_c + ud_c) / 2.0
    return feats


# =============================================================================
#  V1 — BORDER (contour)
# =============================================================================

def _largest_contour(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8) * 255,
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea) if cnts else None

def _radial_features(contour):
    pts    = contour[:, 0, :].astype(np.float64)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    dists  = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    mu     = dists.mean() + 1e-8
    f      = np.abs(scipy_fft(dists))
    f_norm = f / (f[0] + 1e-8)
    coefs  = [float(f_norm[k]) for k in range(1, N_FOURIER_COEFS + 1)]
    return float(dists.std() / mu), coefs

def _notch_count(contour, area, thr=0.02):
    hidx = cv2.convexHull(contour, returnPoints=False)
    if hidx is None or len(hidx) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(contour, hidx)
    except Exception:
        return 0
    if defects is None:
        return 0
    return int((defects[:, 0, 3] / 256.0 > thr * np.sqrt(area + 1e-8)).sum())

def extract_border_contour(mask) -> dict:
    zero_keys = (
        ["abcde_compactness", "abcde_border_roughness", "abcde_convexity",
         "abcde_solidity", "abcde_elongation", "abcde_radial_dist_std",
         "abcde_border_notch_cnt"]
        + [f"abcde_fourier_coef{k+1}" for k in range(N_FOURIER_COEFS)]
    )
    feats = {k: 0.0 for k in zero_keys}
    cnt   = _largest_contour(mask)
    if cnt is None or len(cnt) < 5:
        return feats
    area  = float(cv2.contourArea(cnt)) + 1e-8
    peri  = float(cv2.arcLength(cnt, True)) + 1e-8
    hull  = cv2.convexHull(cnt)
    hperi = float(cv2.arcLength(hull, True)) + 1e-8
    harea = float(cv2.contourArea(hull)) + 1e-8
    feats["abcde_compactness"]      = float(4 * np.pi * area / peri ** 2)
    feats["abcde_border_roughness"] = float(peri / hperi) - 1.0
    feats["abcde_convexity"]        = float(peri / hperi)
    feats["abcde_solidity"]         = float(area / harea)
    try:
        _, (ma, mi), _ = cv2.fitEllipse(cnt)
        feats["abcde_elongation"] = float(mi / (ma + 1e-8))
    except Exception:
        feats["abcde_elongation"] = 0.0
    radial_std, coefs = _radial_features(cnt)
    feats["abcde_radial_dist_std"] = radial_std
    for k, v in enumerate(coefs):
        feats[f"abcde_fourier_coef{k+1}"] = v
    feats["abcde_border_notch_cnt"] = float(_notch_count(cnt, area))
    return feats


# =============================================================================
#  V1 — BORDER (8-octant sharpness)
# =============================================================================

def extract_border_octants(mask, grey) -> dict:
    zero_keys = (
        ["abcde_border_b8_score", "abcde_border_mean_gradient",
         "abcde_border_gradient_std", "abcde_border_sharpness_cv"]
        + [f"abcde_border_oct_{o}" for o in range(8)]
    )
    feats = {k: 0.0 for k in zero_keys}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    cy, cx = prop.centroid
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
    border = mask & ~eroded.astype(bool)
    ys, xs = np.where(border)
    if len(ys) < 16:
        return feats
    gx   = cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    angles  = np.arctan2(ys - cy, xs - cx)
    octants = ((angles + np.pi) / (np.pi / 4)).astype(int) % 8
    oct_grad = np.zeros(8)
    for o in range(8):
        idx = octants == o
        if idx.sum() > 0:
            oct_grad[o] = grad[ys[idx], xs[idx]].mean()
    border_grads  = grad[ys, xs]
    global_median = np.median(border_grads)
    abrupt        = oct_grad > (global_median * OCTANT_ABRUPT_MULT)
    mean_g = float(oct_grad.mean())
    std_g  = float(oct_grad.std())
    feats["abcde_border_b8_score"]      = float(int(abrupt.sum()))
    feats["abcde_border_mean_gradient"] = mean_g
    feats["abcde_border_gradient_std"]  = std_g
    feats["abcde_border_sharpness_cv"]  = float(std_g / (mean_g + 1e-8))
    for o in range(8):
        feats[f"abcde_border_oct_{o}"] = float(oct_grad[o])
    return feats


# =============================================================================
#  V1 — COLOR statistics
# =============================================================================

def _ch_stats(vals, prefix) -> dict:
    if not len(vals):
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}
    return {f"{prefix}_mean": float(vals.mean()), f"{prefix}_std": float(vals.std())}

def _color_variegation(h_vals, n_bins=8):
    if len(h_vals) < 10:
        return 0, 0.0
    hist, _ = np.histogram(h_vals, bins=n_bins, range=(0, 1))
    hn      = hist / (hist.sum() + 1e-8)
    return int((hn > 0.05).sum()), float(stats.entropy(hn + 1e-12))

def _blue_white_veil(p) -> float:
    return float(((p[:, 2] > 0.4) & (p[:, 1] < 0.75) &
                  (p[:, 0] < 0.6) & ((p[:, 2] - p[:, 0]) > 0.1)).mean())

def _regression_zone(p) -> float:
    return float(((p.mean(1) > 0.75) & (p.std(1) < 0.08)).mean())

def extract_color_stats(rgb, hsv, lab, grey, mask) -> dict:
    pr = rgb[mask]; ph = hsv[mask]; pl = lab[mask]; pg = grey[mask]
    zero_keys = (
        [f"abcde_col_{ch}_{s}"
         for ch in ["R","G","B","H","S","V","L","A","LAB_B","grey"]
         for s in ["mean","std"]]
        + [f"abcde_col_{k}" for k in [
            "rg_ratio","rb_ratio","bg_diff","std_composite","entropy",
            "dark_fraction","bright_fraction","num_dominant_hues",
            "color_blob_entropy","blue_white_veil_frac","regression_zone_frac"]]
    )
    feats = {k: 0.0 for k in zero_keys}
    if not len(pr):
        return feats
    for i, ch in enumerate(["R","G","B"]):
        feats.update(_ch_stats(pr[:, i], f"abcde_col_{ch}"))
    for i, ch in enumerate(["H","S","V"]):
        feats.update(_ch_stats(ph[:, i], f"abcde_col_{ch}"))
    for i, ch in enumerate(["L","A","LAB_B"]):
        feats.update(_ch_stats(pl[:, i], f"abcde_col_{ch}"))
    feats.update(_ch_stats(pg, "abcde_col_grey"))
    rm = feats["abcde_col_R_mean"]; gm = feats["abcde_col_G_mean"]; bm = feats["abcde_col_B_mean"]
    feats["abcde_col_rg_ratio"]      = float(rm / (gm + 1e-8))
    feats["abcde_col_rb_ratio"]      = float(rm / (bm + 1e-8))
    feats["abcde_col_bg_diff"]       = float(bm - gm)
    feats["abcde_col_std_composite"] = float(
        (feats["abcde_col_R_std"] + feats["abcde_col_G_std"] + feats["abcde_col_B_std"]) / 3.0)
    vh, _ = np.histogram(ph[:, 2], bins=32, range=(0, 1))
    feats["abcde_col_entropy"]         = float(stats.entropy(vh / (vh.sum() + 1e-8) + 1e-12))
    feats["abcde_col_dark_fraction"]   = float((ph[:, 2] < 0.25).mean())
    feats["abcde_col_bright_fraction"] = float((ph[:, 2] > 0.80).mean())
    nd, ce = _color_variegation(ph[:, 0])
    feats["abcde_col_num_dominant_hues"]  = float(nd)
    feats["abcde_col_color_blob_entropy"] = ce
    feats["abcde_col_blue_white_veil_frac"] = _blue_white_veil(pr)
    feats["abcde_col_regression_zone_frac"] = _regression_zone(pr)
    return feats


# =============================================================================
#  V1 — 6-COLOR PALETTE (Argenziano 2003)
# =============================================================================

def extract_six_color_palette(rgb, hsv, mask) -> dict:
    zero_keys = ["col6_white_frac","col6_red_frac","col6_light_brown_frac",
                 "col6_dark_brown_frac","col6_blue_grey_frac","col6_black_frac",
                 "col6_color_count","col6_high_risk_frac"]
    feats = {k: 0.0 for k in zero_keys}
    ph = hsv[mask]
    if not len(ph):
        return feats
    h, s, v = ph[:, 0], ph[:, 1], ph[:, 2]
    fracs = {
        "col6_white_frac":       float(((v > 0.80) & (s < 0.20)).mean()),
        "col6_red_frac":         float((((h < 0.04)|(h > 0.95)) & (s > 0.35) & (v > 0.35)).mean()),
        "col6_light_brown_frac": float(((h>=0.03)&(h<=0.12)&(s>=0.20)&(v>=0.40)&(v<0.80)).mean()),
        "col6_dark_brown_frac":  float(((h>=0.03)&(h<=0.13)&(s>=0.25)&(v>=0.10)&(v<0.40)).mean()),
        "col6_blue_grey_frac":   float(((h>=0.50)&(h<=0.78)&(s>=0.08)&(s<=0.55)&(v>=0.18)).mean()),
        "col6_black_frac":       float((v < 0.12).mean()),
    }
    feats.update(fracs)
    feats["col6_color_count"]    = float(sum(1 for f in fracs.values() if f > COLOR6_THRESHOLD))
    feats["col6_high_risk_frac"] = float(np.clip(
        fracs["col6_dark_brown_frac"] + fracs["col6_blue_grey_frac"] + fracs["col6_black_frac"],
        0.0, 1.0))
    return feats


# =============================================================================
#  V1 — COLOR HETEROGENEITY
# =============================================================================

def extract_color_heterogeneity(hsv, lab, mask) -> dict:
    zero_keys = [
        "abcde_col_het_S_iqr","abcde_col_het_S_mad","abcde_col_het_S_cv",
        "abcde_col_het_S_range","abcde_col_het_S_p10","abcde_col_het_S_p90",
        "abcde_col_het_S_kurtosis","abcde_col_het_S_skewness",
        "abcde_col_het_V_iqr","abcde_col_het_V_mad","abcde_col_het_V_cv","abcde_col_het_V_range",
        "abcde_col_het_H_iqr","abcde_col_het_H_mad","abcde_col_het_H_cv","abcde_col_het_H_range",
        "abcde_col_het_A_iqr","abcde_col_het_A_mad","abcde_col_het_A_cv",
        "abcde_col_het_L_iqr","abcde_col_het_L_cv","abcde_col_het_composite",
    ]
    feats = {k: 0.0 for k in zero_keys}
    if not mask.sum():
        return feats
    ph = hsv[mask]; pl = lab[mask]
    h_v, s_v, v_v = ph[:, 0], ph[:, 1], ph[:, 2]
    l_v, a_v = pl[:, 0], pl[:, 1]
    def _iqr(a): return float(np.percentile(a, 75) - np.percentile(a, 25))
    def _cv(a):  return float(a.std() / (np.abs(a.mean()) + 1e-8))
    def _mad(a): return float(np.mean(np.abs(a - a.mean())))
    feats["abcde_col_het_S_iqr"]      = _iqr(s_v)
    feats["abcde_col_het_S_mad"]      = _mad(s_v)
    feats["abcde_col_het_S_cv"]       = _cv(s_v)
    feats["abcde_col_het_S_range"]    = float(s_v.max() - s_v.min())
    feats["abcde_col_het_S_p10"]      = float(np.percentile(s_v, 10))
    feats["abcde_col_het_S_p90"]      = float(np.percentile(s_v, 90))
    feats["abcde_col_het_S_kurtosis"] = float(stats.kurtosis(s_v))
    feats["abcde_col_het_S_skewness"] = float(stats.skew(s_v))
    feats["abcde_col_het_V_iqr"]   = _iqr(v_v); feats["abcde_col_het_V_mad"]   = _mad(v_v)
    feats["abcde_col_het_V_cv"]    = _cv(v_v);  feats["abcde_col_het_V_range"] = float(v_v.max()-v_v.min())
    feats["abcde_col_het_H_iqr"]   = _iqr(h_v); feats["abcde_col_het_H_mad"]   = _mad(h_v)
    feats["abcde_col_het_H_cv"]    = _cv(h_v);  feats["abcde_col_het_H_range"] = float(h_v.max()-h_v.min())
    feats["abcde_col_het_A_iqr"]   = _iqr(a_v); feats["abcde_col_het_A_mad"]   = _mad(a_v)
    feats["abcde_col_het_A_cv"]    = _cv(a_v)
    feats["abcde_col_het_L_iqr"]   = _iqr(l_v); feats["abcde_col_het_L_cv"]    = _cv(l_v)
    feats["abcde_col_het_composite"] = float((s_v.std() + v_v.std() + h_v.std()) / 3.0)
    return feats


# =============================================================================
#  V1 — RADIAL COLOR (2-zone)
# =============================================================================

def extract_radial_color(hsv, mask) -> dict:
    zero_keys = (
        [f"abcde_radial_{ch}_{z}" for ch in ["H","S","V"]
         for z in ["center_mean","periph_mean","gradient"]]
        + ["abcde_radial_dark_center_frac","abcde_radial_dark_periph_frac",
           "abcde_radial_dark_outer_frac",   # alias used by report.py
           "abcde_radial_dark_gradient",
           "abcde_radial_sat_center_mean","abcde_radial_sat_periph_mean"]
    )
    feats = {k: 0.0 for k in zero_keys}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    cy, cx = prop.centroid
    ys, xs = np.where(mask)
    if len(ys) < 20:
        return feats
    dists  = np.hypot(ys - cy, xs - cx)
    norm_d = dists / (dists.max() + 1e-8)
    central = norm_d <= 0.50
    periph  = norm_d > 0.50
    if central.sum() < 5 or periph.sum() < 5:
        return feats
    for ch_idx, ch_name in [(0,"H"),(1,"S"),(2,"V")]:
        ch_vals = hsv[:, :, ch_idx][mask]
        c_mean  = float(ch_vals[central].mean())
        p_mean  = float(ch_vals[periph].mean())
        feats[f"abcde_radial_{ch_name}_center_mean"] = c_mean
        feats[f"abcde_radial_{ch_name}_periph_mean"] = p_mean
        feats[f"abcde_radial_{ch_name}_gradient"]    = p_mean - c_mean
    v_vals = hsv[:, :, 2][mask]
    dark   = v_vals < 0.30
    dark_c = float(dark[central].mean())
    dark_p = float(dark[periph].mean())
    feats["abcde_radial_dark_center_frac"] = dark_c
    feats["abcde_radial_dark_periph_frac"] = dark_p
    feats["abcde_radial_dark_outer_frac"]  = dark_p   # alias for report.py
    feats["abcde_radial_dark_gradient"]    = dark_p - dark_c
    s_vals = hsv[:, :, 1][mask]
    feats["abcde_radial_sat_center_mean"] = float(s_vals[central].mean())
    feats["abcde_radial_sat_periph_mean"] = float(s_vals[periph].mean())
    return feats


# =============================================================================
#  V1 — DIAMETER (+ diam_mm alias for report.py)
# =============================================================================

def extract_diameter(mask, max_dim: int = DEFAULT_MAX_DIM) -> dict:
    area   = float(mask.sum())
    diam_px = float(np.sqrt(4.0 * area / np.pi))
    return {
        "abcde_area_px": area,
        "abcde_diam_px": diam_px,
        # Estimated mm assuming DERMOSCOPY_FOV_MM field of view at max_dim resolution
        "abcde_diam_mm": float(diam_px * DERMOSCOPY_FOV_MM / (max_dim + 1e-8)),
    }


# =============================================================================
#  V1 — PYRADIOMICS
# =============================================================================

_EXTRACTOR = None

def _get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None and _RADIOMICS_OK:
        params = {
            "setting": {"binWidth": 25, "resampledPixelSpacing": None,
                        "force2D": True, "force2Ddimension": 0},
            "featureClass": {cls: [] for cls in RADIOMICS_CLASSES},
        }
        _EXTRACTOR = _rx_extractor.RadiomicsFeatureExtractor(params)
        _EXTRACTOR.disableAllFeatures()
        for cls in RADIOMICS_CLASSES:
            _EXTRACTOR.enableFeatureClassByName(cls)
    return _EXTRACTOR

def extract_pyradiomics(roi_gray: np.ndarray, bw: np.ndarray) -> dict:
    ext = _get_extractor()
    if ext is None or bw.sum() == 0:
        return {}
    try:
        img_sitk  = sitk.GetImageFromArray(roi_gray.astype(np.int16))
        mask_sitk = sitk.GetImageFromArray(bw.astype(np.uint8))
        img_sitk.SetSpacing([1.0, 1.0]); mask_sitk.SetSpacing([1.0, 1.0])
        result = ext.execute(img_sitk, mask_sitk, label=1)
        return {k.replace("original_", "rx_", 1): _safe(v)
                for k, v in result.items() if not k.startswith("diagnostics_")}
    except Exception:
        return {}


# =============================================================================
#  V2 NEW — RGB FIRST-ORDER STATISTICS
# =============================================================================

def _fo_stats(vals: np.ndarray, prefix: str) -> dict:
    stat_keys = ["mean","std","skewness","kurtosis","median",
                 "p10","p90","iqr","range","energy","entropy","uniformity","mad"]
    if not len(vals):
        return {f"{prefix}_{s}": 0.0 for s in stat_keys}
    v = vals.astype(np.float64)
    h, _ = np.histogram(v, bins=32, range=(0, 1))
    return {
        f"{prefix}_mean":      float(v.mean()),
        f"{prefix}_std":       float(v.std()),
        f"{prefix}_skewness":  float(stats.skew(v)),
        f"{prefix}_kurtosis":  float(stats.kurtosis(v)),
        f"{prefix}_median":    float(np.median(v)),
        f"{prefix}_p10":       float(np.percentile(v, 10)),
        f"{prefix}_p90":       float(np.percentile(v, 90)),
        f"{prefix}_iqr":       float(np.percentile(v, 75) - np.percentile(v, 25)),
        f"{prefix}_range":     float(v.max() - v.min()),
        f"{prefix}_energy":    float((v ** 2).sum()),
        f"{prefix}_entropy":   float(stats.entropy(h + 1e-12)),
        f"{prefix}_uniformity":float(((h / (len(v) + 1e-8)) ** 2).sum()),
        f"{prefix}_mad":       float(np.mean(np.abs(v - v.mean()))),
    }

def extract_fo_rgb(rgb, mask) -> dict:
    feats = {}
    for i, ch in enumerate(["R", "G", "B"]):
        feats.update(_fo_stats(rgb[:, :, i][mask], f"fo_{ch}"))
    return feats


# =============================================================================
#  V2 NEW — LBP ON HSV CHANNELS
# =============================================================================

def _lbp_histogram(ch_float, mask) -> np.ndarray:
    img_u8  = (ch_float * 255).astype(np.uint8)
    lbp     = local_binary_pattern(img_u8, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)
    hist, _ = np.histogram(lbp[mask], bins=LBP_N_BINS, range=(0, LBP_N_BINS))
    hist    = hist.astype(np.float64) / (hist.sum() + 1e-8)
    return hist

def extract_lbp_color(hsv, mask) -> dict:
    feats = {}
    for idx, ch in enumerate(["H", "S", "V"]):
        hist = _lbp_histogram(hsv[:, :, idx], mask)
        for b, v in enumerate(hist):
            feats[f"lbp_{ch}_{b:02d}"] = float(v)
    return feats


# =============================================================================
#  V2 NEW — GABOR TEXTURE
# =============================================================================

def extract_gabor_texture(grey, mask) -> dict:
    feats = {}
    gf    = grey.astype(np.float32)
    for kernel, freq, theta_deg in _GABOR_KERNELS:
        mag  = np.abs(cv2.filter2D(gf, cv2.CV_32F, kernel))
        vals = mag[mask]
        key  = f"gabor_f{int(freq * 100):02d}_t{theta_deg:03.0f}"
        feats[f"{key}_mean"] = float(vals.mean()) if len(vals) else 0.0
        feats[f"{key}_std"]  = float(vals.std())  if len(vals) else 0.0
    return feats


# =============================================================================
#  V2 NEW — GLCM ON COLOR CHANNELS
# =============================================================================

def _glcm_on_channel(ch_float, mask, ch_name) -> dict:
    feats = {}
    if not mask.any():
        return {f"glcm_{ch_name}_{p}": 0.0 for p in GLCM_PROPS}
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]; c0, c1 = np.where(cols)[0][[0, -1]]
    crop = ch_float[r0:r1 + 1, c0:c1 + 1]; mk = mask[r0:r1 + 1, c0:c1 + 1]
    chq = (crop * (GLCM_LEVELS - 1)).astype(np.uint8); chq[~mk] = 0
    try:
        glcm = graycomatrix(chq, GLCM_DISTANCES, GLCM_ANGLES,
                            GLCM_LEVELS, symmetric=True, normed=True)
        for p in GLCM_PROPS:
            feats[f"glcm_{ch_name}_{p}"] = float(graycoprops(glcm, p).mean())
    except Exception:
        feats = {f"glcm_{ch_name}_{p}": 0.0 for p in GLCM_PROPS}
    return feats

def extract_glcm_color(hsv, mask) -> dict:
    feats = {}
    feats.update(_glcm_on_channel(hsv[:, :, 0], mask, "H"))
    feats.update(_glcm_on_channel(hsv[:, :, 1], mask, "S"))
    return feats


# =============================================================================
#  V2 NEW — SHAPE EXTRAS
# =============================================================================

def extract_shape_extras(mask) -> dict:
    feats = {"shape_eccentricity": 0.0, "shape_extent": 0.0, "shape_euler": 0.0}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    feats["shape_eccentricity"] = float(prop.eccentricity)
    feats["shape_extent"]       = float(prop.extent)
    feats["shape_euler"]        = float(prop.euler_number)
    return feats


# =============================================================================
#  V2 NEW — BORDER FRACTAL DIMENSION
# =============================================================================

def _box_count(binary_border, sizes):
    counts = []
    for s in sizes:
        h, w   = binary_border.shape
        h_pad  = int(np.ceil(h / s)) * s
        w_pad  = int(np.ceil(w / s)) * s
        Z      = np.zeros((h_pad, w_pad), dtype=bool)
        Z[:h, :w] = binary_border
        blocks = Z.reshape(h_pad // s, s, w_pad // s, s)
        counts.append(int(blocks.any(axis=(1, 3)).sum()))
    return counts

def extract_border_fractal(mask) -> dict:
    feats = {"abcde_border_fractal_dim": 0.0}
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    border = mask & ~eroded.astype(bool)
    if border.sum() < 16:
        return feats
    max_s = max(border.shape) // 4
    if max_s < 4:
        return feats
    p      = int(np.floor(np.log2(max_s)))
    sizes  = [2 ** k for k in range(1, p + 1)]
    counts = _box_count(border, sizes)
    valid  = [(s, c) for s, c in zip(sizes, counts) if c > 0]
    if len(valid) < 3:
        return feats
    log_s = np.log([v[0] for v in valid])
    log_c = np.log([v[1] for v in valid])
    slope, *_ = stats.linregress(log_s, log_c)
    feats["abcde_border_fractal_dim"] = float(max(0.0, -slope))
    return feats


# =============================================================================
#  V2 NEW — COLOR SPATIAL ENTROPY (4×4 grid)
# =============================================================================

def extract_color_spatial_entropy(hsv, mask, n_grid=4) -> dict:
    feats = {"abcde_color_spatial_S_entropy": 0.0, "abcde_color_spatial_V_entropy": 0.0,
             "abcde_color_spatial_S_std": 0.0,     "abcde_color_spatial_V_std": 0.0}
    ys, xs = np.where(mask)
    if len(ys) < n_grid * n_grid:
        return feats
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    if y1 == y0 or x1 == x0:
        return feats
    S_img = hsv[:, :, 1]; V_img = hsv[:, :, 2]
    cell_S, cell_V = [], []
    for gy in range(n_grid):
        for gx in range(n_grid):
            r0 = y0 + int((y1 - y0) * gy / n_grid)
            r1 = y0 + int((y1 - y0) * (gy + 1) / n_grid)
            c0 = x0 + int((x1 - x0) * gx / n_grid)
            c1 = x0 + int((x1 - x0) * (gx + 1) / n_grid)
            cm = mask[r0:r1, c0:c1]
            if cm.sum() < 4:
                continue
            cell_S.append(float(S_img[r0:r1, c0:c1][cm].mean()))
            cell_V.append(float(V_img[r0:r1, c0:c1][cm].mean()))
    if len(cell_S) < 3:
        return feats
    def _ent(vals):
        h, _ = np.histogram(np.array(vals), bins=min(len(vals), 8), range=(0, 1))
        return float(stats.entropy(h + 1e-12))
    feats["abcde_color_spatial_S_entropy"] = _ent(cell_S)
    feats["abcde_color_spatial_V_entropy"] = _ent(cell_V)
    feats["abcde_color_spatial_S_std"]     = float(np.std(cell_S))
    feats["abcde_color_spatial_V_std"]     = float(np.std(cell_V))
    return feats


# =============================================================================
#  V2 NEW — GABOR DIRECTIONALITY
# =============================================================================

def extract_gabor_directionality(grey, mask) -> dict:
    feats = {"gabor_directionality_entropy": 0.0,
             "gabor_directionality_max_ratio": 0.0,
             "gabor_directionality_angle": 0.0}
    gf            = grey.astype(np.float32)
    orient_angles = [0, 45, 90, 135]
    orient_energy = np.zeros(4)
    for kernel, freq, theta_deg in _GABOR_KERNELS:
        mag  = np.abs(cv2.filter2D(gf, cv2.CV_32F, kernel))
        vals = mag[mask]
        if not len(vals):
            continue
        oidx = orient_angles.index(int(theta_deg))
        orient_energy[oidx] += float(vals.mean())
    total = orient_energy.sum() + 1e-8
    norm  = orient_energy / total
    feats["gabor_directionality_entropy"]   = float(stats.entropy(norm + 1e-12))
    feats["gabor_directionality_max_ratio"] = float(orient_energy.max() / (orient_energy.mean() + 1e-8))
    feats["gabor_directionality_angle"]     = float(orient_angles[int(np.argmax(orient_energy))])
    return feats


# =============================================================================
#  V2 NEW — DOTS / GLOBULES
# =============================================================================

def extract_dots_globules(hsv, mask) -> dict:
    feats = {"abcde_dots_count": 0.0, "abcde_dots_density": 0.0,
             "abcde_dots_size_cv": 0.0, "abcde_dots_frac": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    V = hsv[:, :, 2]; S = hsv[:, :, 1]
    dark_px = mask & ((V < 0.30) | ((V < 0.50) & (S > 0.30)))
    if dark_px.sum() < 4:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(dark_px.astype(np.uint8), cv2.MORPH_OPEN, kern)
    labeled   = sklabel(opened)
    all_props = regionprops(labeled)
    max_dot_area = lesion_area * DOTS_MAX_AREA_FRAC
    dot_areas = []
    for p in all_props:
        a = p.area
        if a < DOTS_MIN_AREA_PX or a > max_dot_area:
            continue
        peri = p.perimeter + 1e-8
        if (4 * np.pi * a / peri ** 2) >= DOTS_MIN_CIRCULARITY:
            dot_areas.append(float(a))
    n = len(dot_areas)
    if n == 0:
        return feats
    feats["abcde_dots_count"]   = float(n)
    feats["abcde_dots_density"] = float(n / lesion_area * 1000.0)
    arr = np.array(dot_areas)
    feats["abcde_dots_size_cv"] = float(arr.std() / (arr.mean() + 1e-8))
    feats["abcde_dots_frac"]    = float(arr.sum() / lesion_area)
    return feats


# =============================================================================
#  V2 NEW — 3-ZONE RADIAL ANALYSIS
# =============================================================================

def extract_radial_3zone(hsv, mask) -> dict:
    channels  = ["H", "S", "V"]
    zero_keys = (
        [f"abcde_radial3_{ch}_{z}" for ch in channels for z in ["inner","mid","outer"]]
        + [f"abcde_radial3_{ch}_{g}" for ch in channels for g in ["mid_grad","out_grad"]]
    )
    feats = {k: 0.0 for k in zero_keys}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    cy, cx = prop.centroid
    ys, xs = np.where(mask)
    if len(ys) < 30:
        return feats
    dists  = np.hypot(ys - cy, xs - cx)
    norm_d = dists / (dists.max() + 1e-8)
    inner  = norm_d <= 1/3
    mid    = (norm_d > 1/3) & (norm_d <= 2/3)
    outer  = norm_d > 2/3
    if inner.sum() < 5 or mid.sum() < 5 or outer.sum() < 5:
        return feats
    for ch_idx, ch_name in enumerate(channels):
        ch_vals = hsv[:, :, ch_idx][mask]
        i_m = float(ch_vals[inner].mean())
        m_m = float(ch_vals[mid].mean())
        o_m = float(ch_vals[outer].mean())
        feats[f"abcde_radial3_{ch_name}_inner"]    = i_m
        feats[f"abcde_radial3_{ch_name}_mid"]      = m_m
        feats[f"abcde_radial3_{ch_name}_outer"]    = o_m
        feats[f"abcde_radial3_{ch_name}_mid_grad"] = m_m - i_m
        feats[f"abcde_radial3_{ch_name}_out_grad"] = o_m - m_m
    return feats


# =============================================================================
#  V3 NEW — MILIA-LIKE CYSTS
#  Clinical: bright, round, white structures inside lesion body.
#  High specificity for seborrhoeic keratosis (Kittler 2016).
#  Reported explicitly in clinical dermoscopy reports as count + presence.
# =============================================================================

def extract_milia_cysts(hsv, mask) -> dict:
    """
    Keys: abcde_milia_count, abcde_milia_density, abcde_milia_frac.
    Detection: V > MILIA_V_MIN, S < MILIA_S_MAX, small, round.
    """
    feats       = {"abcde_milia_count": 0.0, "abcde_milia_density": 0.0, "abcde_milia_frac": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    V = hsv[:, :, 2]; S = hsv[:, :, 1]
    milia_px = mask & (V > MILIA_V_MIN) & (S < MILIA_S_MAX)
    if milia_px.sum() < 4:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(milia_px.astype(np.uint8), cv2.MORPH_OPEN, kern)
    labeled = sklabel(opened)
    max_area = lesion_area * MILIA_MAX_FRAC
    count = 0; total_area = 0
    for p in regionprops(labeled):
        if p.area < MILIA_MIN_PX or p.area > max_area:
            continue
        if (4 * np.pi * p.area / (p.perimeter + 1e-8) ** 2) >= MILIA_MIN_CIRC:
            count += 1; total_area += p.area
    feats["abcde_milia_count"]   = float(count)
    feats["abcde_milia_density"] = float(count / lesion_area * 1000)
    feats["abcde_milia_frac"]    = float(total_area / lesion_area)
    return feats


# =============================================================================
#  V3 NEW — COMEDO-LIKE OPENINGS
#  Clinical: dark, round follicular plugs (keratinous material in ostia).
#  Key indicator of seborrhoeic keratosis; differential from melanocytic.
#  Larger and darker than dots/globules.
# =============================================================================

def extract_comedo_openings(hsv, mask) -> dict:
    """
    Keys: abcde_comedo_count, abcde_comedo_density, abcde_comedo_frac.
    Detection: V < COMEDO_V_MAX (very dark), round, medium-small.
    """
    feats       = {"abcde_comedo_count": 0.0, "abcde_comedo_density": 0.0, "abcde_comedo_frac": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    V = hsv[:, :, 2]
    comedo_px = mask & (V < COMEDO_V_MAX)
    if comedo_px.sum() < 4:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(comedo_px.astype(np.uint8), cv2.MORPH_OPEN, kern)
    labeled  = sklabel(opened)
    max_area = lesion_area * COMEDO_MAX_FRAC
    count = 0; total_area = 0
    for p in regionprops(labeled):
        if p.area < COMEDO_MIN_PX or p.area > max_area:
            continue
        if (4 * np.pi * p.area / (p.perimeter + 1e-8) ** 2) >= COMEDO_MIN_CIRC:
            count += 1; total_area += p.area
    feats["abcde_comedo_count"]   = float(count)
    feats["abcde_comedo_density"] = float(count / lesion_area * 1000)
    feats["abcde_comedo_frac"]    = float(total_area / lesion_area)
    return feats


# =============================================================================
#  V3 NEW — BLOTCH DETECTION
#  Clinical: large, irregular dark areas.  7-point checklist minor criterion
#  (score 1 — "irregular blotches").  Distinct from dots (size) and from
#  regression (colour — blotches are dark pigment, not white/grey scar).
#  In current report.py, blotches are proxied via asymmetry score; this
#  provides a dedicated morphological detector.
# =============================================================================

def extract_blotches(hsv, mask) -> dict:
    """
    Keys: abcde_blotch_count, abcde_blotch_frac, abcde_blotch_irregularity.
    Detection: V < BLOTCH_V_MAX (dark), area > BLOTCH_MIN_FRAC of lesion.
    abcde_blotch_irregularity: 1 − mean_compactness (higher = more irregular).
    """
    feats = {"abcde_blotch_count": 0.0, "abcde_blotch_frac": 0.0, "abcde_blotch_irregularity": 0.0}
    lesion_area   = float(mask.sum())
    min_blotch_px = max(50, lesion_area * BLOTCH_MIN_FRAC)
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    V = hsv[:, :, 2]; S = hsv[:, :, 1]
    dark_px = mask & ((V < BLOTCH_V_MAX) | ((V < 0.40) & (S > 0.45)))
    if dark_px.sum() < min_blotch_px:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(dark_px.astype(np.uint8), cv2.MORPH_CLOSE, kern)
    labeled = sklabel(closed)
    count = 0; blotch_area = 0; circs = []
    for p in regionprops(labeled):
        if p.area < min_blotch_px:
            continue
        count += 1; blotch_area += p.area
        circs.append(4 * np.pi * p.area / (p.perimeter + 1e-8) ** 2)
    feats["abcde_blotch_count"]       = float(count)
    feats["abcde_blotch_frac"]        = float(blotch_area / lesion_area)
    feats["abcde_blotch_irregularity"] = float(1.0 - np.mean(circs)) if circs else 0.0
    return feats


# =============================================================================
#  V3 NEW — LACUNAE
#  Clinical: dark-red to maroon, round structures in vascular lesions
#  (angioma, angiokeratoma).  Dermoscopy textbook key finding (Braun 2008).
#  Colour: dark-red HSV range distinct from brown pigment or blue-grey.
# =============================================================================

def extract_lacunae(hsv, mask) -> dict:
    """
    Keys: abcde_lacunae_count, abcde_lacunae_frac.
    Detection: dark-red colour (H near 0/1, high S, moderate V), round.
    """
    feats       = {"abcde_lacunae_count": 0.0, "abcde_lacunae_frac": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    lac_px = mask & (
        ((H < LACUNAE_H_MAX) | (H > LACUNAE_H_MIN_HI))
        & (S > LACUNAE_S_MIN)
        & (V >= LACUNAE_V_MIN) & (V <= LACUNAE_V_MAX)
    )
    if lac_px.sum() < 8:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(lac_px.astype(np.uint8), cv2.MORPH_CLOSE, kern)
    labeled  = sklabel(closed)
    max_area = lesion_area * LACUNAE_MAX_FRAC
    count = 0; total_area = 0
    for p in regionprops(labeled):
        if p.area < LACUNAE_MIN_PX or p.area > max_area:
            continue
        if (4 * np.pi * p.area / (p.perimeter + 1e-8) ** 2) >= LACUNAE_MIN_CIRC:
            count += 1; total_area += p.area
    feats["abcde_lacunae_count"] = float(count)
    feats["abcde_lacunae_frac"]  = float(total_area / lesion_area)
    return feats


# =============================================================================
#  V3 NEW — PEPPERING AND SCAR-LIKE REGRESSION
#  Clinical: Two distinct regression subtypes:
#   Peppering  — multiple small grey particles = melanophage deposits from
#                involuting melanoma (Soyer 2004; Kittler 2016).
#   Scar-like  — diffuse white area = advanced regression / fibrous tissue.
#  Both are different from the existing abcde_col_regression_zone_frac
#  (which captures only bright uniform areas without distinguishing type).
# =============================================================================

def extract_peppering_regression(hsv, mask) -> dict:
    """
    Keys: abcde_peppering_frac, abcde_scar_regression_frac,
          abcde_regression_type_score (0=none, 1=pepper, 2=scar, 3=both).
    """
    feats       = {"abcde_peppering_frac": 0.0, "abcde_scar_regression_frac": 0.0,
                   "abcde_regression_type_score": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    V = hsv[:, :, 2]; S = hsv[:, :, 1]
    pepper_px = mask & (V >= PEPPER_V_MIN) & (V <= PEPPER_V_MAX) & (S <= PEPPER_S_MAX)
    scar_px   = mask & (V > SCAR_V_MIN)   & (S < SCAR_S_MAX)
    pf = float(pepper_px.sum() / lesion_area)
    sf = float(scar_px.sum()   / lesion_area)
    feats["abcde_peppering_frac"]       = pf
    feats["abcde_scar_regression_frac"] = sf
    feats["abcde_regression_type_score"] = float(int(pf > 0.02) + 2 * int(sf > 0.03))
    return feats


# =============================================================================
#  V3 NEW — WHITE / CRYSTALLINE STRUCTURES
#  Clinical: very bright white structures within lesion.  Under polarised
#  dermoscopy these appear as shiny white "chrysalis" streaks (melanoma,
#  BCC; Balagula 2012).  Without polarisation, extremely bright focal white
#  areas are still clinically notable.
#  Elongated subtype: minor/major axis < 0.40 → shiny white streaks.
# =============================================================================

def extract_white_structures(hsv, mask) -> dict:
    """
    Keys: abcde_white_struct_count, abcde_white_struct_frac,
          abcde_white_struct_elongated_frac.
    Detection: V > WHITE_V_MIN, S < WHITE_S_MAX, small-to-medium, not diffuse.
    """
    feats = {"abcde_white_struct_count": 0.0, "abcde_white_struct_frac": 0.0,
             "abcde_white_struct_elongated_frac": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    V = hsv[:, :, 2]; S = hsv[:, :, 1]
    white_px = mask & (V > WHITE_V_MIN) & (S < WHITE_S_MAX)
    if white_px.sum() < 4:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(white_px.astype(np.uint8), cv2.MORPH_OPEN, kern)
    labeled  = sklabel(opened)
    max_area = lesion_area * WHITE_MAX_FRAC
    count = 0; elongated = 0; total_area = 0
    for p in regionprops(labeled):
        if p.area < WHITE_MIN_PX or p.area > max_area:
            continue
        count += 1; total_area += p.area
        if p.major_axis_length > 0:
            if (p.minor_axis_length / (p.major_axis_length + 1e-8)) < WHITE_ELONGATED_RATIO:
                elongated += 1
    feats["abcde_white_struct_count"]           = float(count)
    feats["abcde_white_struct_frac"]            = float(total_area / lesion_area)
    feats["abcde_white_struct_elongated_frac"]  = float(elongated / (count + 1e-8)) if count else 0.0
    return feats


# =============================================================================
#  V3 NEW — VASCULAR / RED STRUCTURES
#  Clinical: atypical vascular pattern is a major 7-point checklist criterion
#  (score 2).  Current report.py proxies it via shape_eccentricity > 0.80.
#  This detector provides direct evidence: red pixel fraction, number of
#  distinct red clusters, and small round red structures (dotted vessels).
# =============================================================================

def extract_vascular_structures(hsv, mask) -> dict:
    """
    Keys: abcde_vascular_red_frac, abcde_vascular_cluster_count,
          abcde_vascular_dot_score.
    Detection: red hue (H < 0.05 or H > 0.95), S > 0.30, V > 0.30.
    dot_score: fraction of small round red clusters × red fraction
               (proxy for dotted vessel pattern).
    """
    feats = {"abcde_vascular_red_frac": 0.0, "abcde_vascular_cluster_count": 0.0,
             "abcde_vascular_dot_score": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    red_px = mask & (((H < 0.05) | (H > 0.95)) & (S > 0.30) & (V > 0.30))
    red_frac = float(red_px.sum() / lesion_area)
    feats["abcde_vascular_red_frac"] = red_frac
    if red_px.sum() < 4:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(red_px.astype(np.uint8), cv2.MORPH_CLOSE, kern)
    labeled = sklabel(closed)
    sig_props = [p for p in regionprops(labeled) if p.area >= 10]
    feats["abcde_vascular_cluster_count"] = float(len(sig_props))
    dot_max   = lesion_area * 0.02
    dot_props = [p for p in sig_props
                 if p.area <= dot_max
                 and (4 * np.pi * p.area / (p.perimeter + 1e-8) ** 2) > 0.50]
    feats["abcde_vascular_dot_score"] = float(len(dot_props) / (len(sig_props) + 1e-8)) * red_frac
    return feats


# =============================================================================
#  V3 NEW — SATELLITE LESION TOPOLOGY
#  Clinical: satellite lesions (separate fragments outside the main lesion
#  body) suggest in-transit melanoma metastasis.  Simple to derive from
#  connected-component analysis of the binary mask.
# =============================================================================

def extract_satellite_topology(mask) -> dict:
    """
    Keys: abcde_satellite_count, abcde_main_fragment_frac.
    abcde_satellite_count = number of fragments EXCLUDING the main lesion.
    abcde_main_fragment_frac = fraction of total mask pixels in the largest
    fragment (< 1.0 means fragments exist).
    """
    labeled   = sklabel(mask.astype(np.uint8))
    fragments = sorted([p.area for p in regionprops(labeled) if p.area > 50], reverse=True)
    if not fragments:
        return {"abcde_satellite_count": 0.0, "abcde_main_fragment_frac": 1.0}
    total = float(mask.sum())
    return {
        "abcde_satellite_count":    float(max(0, len(fragments) - 1)),
        "abcde_main_fragment_frac": float(fragments[0] / total),
    }


# =============================================================================
#  V3 NEW — SPATIAL COLOR ZONE SEGMENTATION
#  Clinical: col6_color_count counts colours from a histogram (any location).
#  This detector finds spatially DISTINCT colour zones using k-means
#  clustering, revealing how many coherent pigment regions exist.
#  Multiple distinct zones (multicomponent pattern) is a high-risk dermoscopy
#  pattern (Kittler 2016 Revised Pattern Analysis).
# =============================================================================

def extract_color_zones(hsv, mask, n_zones: int = 6) -> dict:
    """
    Keys: abcde_color_zone_count, abcde_color_zone_size_cv,
          abcde_color_zone_entropy.
    Method: k-means (k=6) on subsampled HSV lesion pixels.
    Zones with > 3% of lesion pixels are counted as 'significant'.
    """
    feats = {"abcde_color_zone_count": 0.0, "abcde_color_zone_size_cv": 0.0,
             "abcde_color_zone_entropy": 0.0}
    pixels = hsv[mask].reshape(-1, 3).astype(np.float32)
    n_px   = len(pixels)
    if n_px < n_zones * 50:
        return feats
    sub    = pixels[np.random.choice(n_px, min(n_px, 4000), replace=False)]
    crit   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    try:
        _, labels, _ = cv2.kmeans(sub, n_zones, None, crit, 3, cv2.KMEANS_RANDOM_CENTERS)
    except Exception:
        return feats
    zone_sizes  = np.bincount(labels.flatten(), minlength=n_zones)
    zone_fracs  = zone_sizes / (zone_sizes.sum() + 1e-8)
    significant = zone_fracs > 0.03
    sig_count   = int(significant.sum())
    feats["abcde_color_zone_count"] = float(sig_count)
    sig_sz = zone_sizes[significant]
    feats["abcde_color_zone_size_cv"] = float(sig_sz.std() / (sig_sz.mean() + 1e-8)) if len(sig_sz) > 1 else 0.0
    feats["abcde_color_zone_entropy"] = float(stats.entropy(zone_fracs + 1e-12))
    return feats


# =============================================================================
#  V3 NEW — PERIPHERAL STRUCTURELESS AREA
#  Clinical: peripheral flat, featureless (low-texture) areas are an
#  expanding melanoma sign — they represent the advancing tumour front
#  without organised pigment network (Johr 2004; Braun 2008).
#  Measured as local texture energy ratio: periphery vs centre.
# =============================================================================

def extract_peripheral_structureless(grey, mask) -> dict:
    """
    Keys: abcde_periph_structureless_score (1 − outer/inner texture,
          higher = more structureless periphery),
          abcde_periph_texture_gradient (inner − outer local std).
    """
    feats = {"abcde_periph_structureless_score": 0.0, "abcde_periph_texture_gradient": 0.0}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    cy, cx = prop.centroid
    ys, xs = np.where(mask)
    if len(ys) < 30:
        return feats
    # Local standard deviation via box filter
    gf      = grey.astype(np.float32)
    gf_sq   = gf ** 2
    blur_g  = cv2.blur(gf,    (9, 9))
    blur_sq = cv2.blur(gf_sq, (9, 9))
    local_var = np.clip(blur_sq - blur_g ** 2, 0, None)
    local_std = np.sqrt(local_var)
    norm_d    = np.hypot(ys - cy, xs - cx) / (np.hypot(ys - cy, xs - cx).max() + 1e-8)
    inner = norm_d <= 0.50; outer = norm_d > 0.70
    tex   = local_std[mask]
    t_in  = float(tex[inner].mean()) if inner.any() else 1e-8
    t_out = float(tex[outer].mean()) if outer.any() else 0.0
    feats["abcde_periph_structureless_score"] = float(1.0 - t_out / (t_in + 1e-8))
    feats["abcde_periph_texture_gradient"]    = float(t_in - t_out)
    return feats


# =============================================================================
#  V3 NEW — ECCENTRIC DARK ZONE (Chaos-and-Clues Clue 1)
#  Clinical: an eccentric structureless zone (a dark area displaced from the
#  lesion centre) is Clue 1 of the Chaos-and-Clues dermoscopy algorithm
#  (Braun 2008).  It indicates asymmetric pigment accumulation — a major
#  concern in melanoma differential.
# =============================================================================

def extract_eccentric_dark(hsv, mask) -> dict:
    """
    Keys: abcde_eccentric_dark_score (1 if dark centroid is displaced > 30%
          of lesion radius from lesion centroid, else 0),
          abcde_eccentric_dark_dist (normalised distance, 0–1+).
    """
    feats = {"abcde_eccentric_dark_score": 0.0, "abcde_eccentric_dark_dist": 0.0}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    cy_l, cx_l = prop.centroid
    V       = hsv[:, :, 2]
    dark_mk = mask & (V < 0.30)
    if dark_mk.sum() < 10:
        return feats
    d_ys, d_xs = np.where(dark_mk)
    cy_d, cx_d = d_ys.mean(), d_xs.mean()
    r_lesion   = np.sqrt(float(mask.sum()) / np.pi) + 1e-8
    norm_dist  = float(np.hypot(cy_d - cy_l, cx_d - cx_l) / r_lesion)
    feats["abcde_eccentric_dark_score"] = float(norm_dist > 0.30)
    feats["abcde_eccentric_dark_dist"]  = norm_dist
    return feats


# =============================================================================
#  V3 NEW — QUADRANT COLOR SPREAD
#  Clinical: asymmetric pigment distribution is the ABCDE A criterion, but
#  fold-based asymmetry measures MIRROR symmetry only.  Quadrant spread
#  captures whether any quadrant (NW/NE/SW/SE) is dramatically different
#  in colour from the others — e.g., one dark corner and three light ones
#  indicates focally concentrated pigment, not just global asymmetry.
# =============================================================================

def extract_quadrant_analysis(hsv, mask) -> dict:
    """
    Keys: abcde_quadrant_S_spread, abcde_quadrant_V_spread,
          abcde_quadrant_H_spread.
    Spread = max(quadrant mean) − min(quadrant mean) for each channel.
    """
    feats = {"abcde_quadrant_S_spread": 0.0, "abcde_quadrant_V_spread": 0.0,
             "abcde_quadrant_H_spread": 0.0}
    prop  = _get_largest_region(mask)
    if prop is None:
        return feats
    cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
    H = hsv[:, :, 0]; S = hsv[:, :, 1]; V = hsv[:, :, 2]
    # Quadrant slices: [rows, cols]
    slices = [
        (slice(None, cy), slice(None, cx)),  # NW
        (slice(None, cy), slice(cx, None)),  # NE
        (slice(cy, None), slice(None, cx)),  # SW
        (slice(cy, None), slice(cx, None)),  # SE
    ]
    S_m, V_m, H_m = [], [], []
    for rs, cs in slices:
        qm = mask[rs, cs]
        if qm.sum() < 10:
            continue
        S_m.append(float(S[rs, cs][qm].mean()))
        V_m.append(float(V[rs, cs][qm].mean()))
        H_m.append(float(H[rs, cs][qm].mean()))
    if len(S_m) < 2:
        return feats
    feats["abcde_quadrant_S_spread"] = float(max(S_m) - min(S_m))
    feats["abcde_quadrant_V_spread"] = float(max(V_m) - min(V_m))
    feats["abcde_quadrant_H_spread"] = float(max(H_m) - min(H_m))
    return feats


# =============================================================================
#  V3 NEW — GLOBULE SPATIAL REGULARITY
#  Clinical: regular globules (uniform size and spacing) are a benign
#  pattern; irregular globules (varying size and spacing) are atypical
#  (Argenziano 1998).  This quantifies the SPATIAL DISTRIBUTION of detected
#  dot/globule centroids via nearest-neighbour distance statistics.
# =============================================================================

def extract_globule_regularity(hsv, mask) -> dict:
    """
    Keys: abcde_globule_spacing_cv  (CV of nearest-neighbour distances;
          low = regular, high = irregular),
          abcde_globule_dist_entropy (entropy of distance distribution).
    Returns zeros if < 3 globules detected.
    """
    feats = {"abcde_globule_spacing_cv": 0.0, "abcde_globule_dist_entropy": 0.0}
    lesion_area = float(mask.sum())
    if lesion_area < MIN_MASK_PIXELS:
        return feats
    V = hsv[:, :, 2]; S = hsv[:, :, 1]
    dark_px = mask & ((V < 0.30) | ((V < 0.50) & (S > 0.30)))
    if dark_px.sum() < 4:
        return feats
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(dark_px.astype(np.uint8), cv2.MORPH_OPEN, kern)
    labeled   = sklabel(opened)
    max_area  = lesion_area * DOTS_MAX_AREA_FRAC
    centroids = []
    for p in regionprops(labeled):
        if p.area < DOTS_MIN_AREA_PX or p.area > max_area:
            continue
        if (4 * np.pi * p.area / (p.perimeter + 1e-8) ** 2) >= DOTS_MIN_CIRCULARITY:
            centroids.append(np.array(p.centroid))
    if len(centroids) < 3:
        return feats
    arr    = np.stack(centroids)
    nn_d   = []
    for i, c in enumerate(arr):
        dists = np.linalg.norm(arr - c, axis=1)
        dists[i] = np.inf
        nn_d.append(float(dists.min()))
    nn_arr = np.array(nn_d)
    feats["abcde_globule_spacing_cv"]     = float(nn_arr.std() / (nn_arr.mean() + 1e-8))
    hist, _ = np.histogram(nn_arr, bins=min(len(nn_d), 8))
    feats["abcde_globule_dist_entropy"]   = float(stats.entropy(hist + 1e-12))
    return feats


# =============================================================================
#  MAIN PER-IMAGE EXTRACTION
# =============================================================================

def extract_all(masked_img_path: str, max_dim: int = DEFAULT_MAX_DIM) -> dict:
    """
    Extract all V1 + V2 + V3 features from one masked image.
    See module docstring for feature group descriptions.
    """
    img_bgr = cv2.imread(masked_img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {masked_img_path}")

    roi_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bw_full = derive_mask(roi_rgb)
    roi_rgb_s, bw_s = _resize_img(roi_rgb, bw_full, max_dim)
    img_bgr_s = cv2.cvtColor(roi_rgb_s, cv2.COLOR_RGB2BGR)
    rgb, hsv, lab, grey = _load_colour_views(img_bgr_s)
    bw = bw_s.astype(np.uint8)

    if bw.sum() < MIN_MASK_PIXELS:
        return {"image_id": _stem(masked_img_path), "mask_pixel_count": int(bw.sum())}

    mask = bw.astype(bool)
    record = {"image_id": _stem(masked_img_path), "mask_pixel_count": int(bw.sum())}

    # ── V1 ────────────────────────────────────────────────────────────────
    record.update(extract_pyradiomics(grey, bw))
    record.update(extract_asymmetry_centre(mask, hsv))
    record.update(extract_asymmetry_principal_axis(mask, hsv))
    record.update(extract_border_contour(mask))
    record.update(extract_border_octants(mask, grey))
    record.update(extract_color_stats(rgb, hsv, lab, grey, mask))
    record.update(extract_six_color_palette(rgb, hsv, mask))
    record.update(extract_color_heterogeneity(hsv, lab, mask))
    record.update(extract_radial_color(hsv, mask))
    record.update(extract_diameter(mask, max_dim))

    # ── V2 ────────────────────────────────────────────────────────────────
    record.update(extract_fo_rgb(rgb, mask))
    record.update(extract_lbp_color(hsv, mask))
    record.update(extract_gabor_texture(grey, mask))
    record.update(extract_glcm_color(hsv, mask))
    record.update(extract_shape_extras(mask))
    record.update(extract_border_fractal(mask))
    record.update(extract_color_spatial_entropy(hsv, mask))
    record.update(extract_gabor_directionality(grey, mask))
    record.update(extract_dots_globules(hsv, mask))
    record.update(extract_radial_3zone(hsv, mask))

    # ── V3 ────────────────────────────────────────────────────────────────
    record.update(extract_milia_cysts(hsv, mask))
    record.update(extract_comedo_openings(hsv, mask))
    record.update(extract_blotches(hsv, mask))
    record.update(extract_lacunae(hsv, mask))
    record.update(extract_peppering_regression(hsv, mask))
    record.update(extract_white_structures(hsv, mask))
    record.update(extract_vascular_structures(hsv, mask))
    record.update(extract_satellite_topology(mask))
    record.update(extract_color_zones(hsv, mask))
    record.update(extract_peripheral_structureless(grey, mask))
    record.update(extract_eccentric_dark(hsv, mask))
    record.update(extract_quadrant_analysis(hsv, mask))
    record.update(extract_globule_regularity(hsv, mask))

    return record


# =============================================================================
#  MULTIPROCESSING WORKER
# =============================================================================

_WORKER_MAX_DIM = DEFAULT_MAX_DIM

def _worker(args):
    import logging
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    logging.getLogger("pykwalify").setLevel(logging.ERROR)
    split, img_path = args
    try:
        rec = extract_all(img_path, max_dim=_WORKER_MAX_DIM)
        rec["split"] = split
        return rec, None
    except Exception as e:
        return None, (img_path, str(e))


# =============================================================================
#  IMAGE DISCOVERY
# =============================================================================

def discover_images(masked_root: str) -> list:
    entries = os.listdir(masked_root)
    subdirs = [e for e in entries if os.path.isdir(os.path.join(masked_root, e))]
    splits  = subdirs if subdirs else [""]
    items   = []
    for split in splits:
        img_dir   = os.path.join(masked_root, split) if split else masked_root
        img_files = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg")
                           for p in glob(os.path.join(img_dir, ext)))
        for ip in img_files:
            items.append((split or "unknown", ip))
    print(f"  Found {len(items)} images across splits: {sorted(set(s for s,_ in items))}")
    return items


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feature Extraction V3 — PyRadiomics + ABCDE + "
                    "Texture (V2) + Clinical Structures (V3)"
    )
    parser.add_argument("--masked_root", default=DEFAULT_MASKED_ROOT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--max_dim",     type=int, default=DEFAULT_MAX_DIM)
    parser.add_argument("--workers",     type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    global _WORKER_MAX_DIM
    _WORKER_MAX_DIM = args.max_dim

    print("=" * 65)
    print("  COMBINED FEATURE EXTRACTION — V3")
    print(f"  input   : {args.masked_root}")
    print(f"  output  : {args.output}")
    print(f"  max_dim={args.max_dim}  workers={args.workers}  "
          f"PyRadiomics={'ON' if _RADIOMICS_OK else 'OFF'}")
    print("=" * 65)

    items = discover_images(args.masked_root)
    if not items:
        raise RuntimeError(f"No images found in {args.masked_root}")

    rows, errors = [], []
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for rec, err in tqdm(pool.imap_unordered(_worker, items, chunksize=4),
                                 total=len(items), desc="Extracting"):
                (rows if rec is not None else errors).append(rec or err)
    else:
        for item in tqdm(items, desc="Extracting"):
            rec, err = _worker(item)
            (rows if rec is not None else errors).append(rec or err)

    if errors:
        print(f"\n  WARNING: {len(errors)} images failed:")
        for e in errors[:10]:
            if isinstance(e, tuple):
                print(f"    {os.path.basename(e[0])}: {e[1]}")

    if not rows:
        raise RuntimeError("No features extracted.")

    all_keys = list(rows[0].keys())
    for r in rows:
        for k in all_keys:
            r.setdefault(k, 0.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader(); w.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────────────
    all_cols = [k for k in all_keys if k not in ("image_id","split","mask_pixel_count")]

    def _grp(prefixes, cols=all_cols):
        return [c for c in cols if any(c.startswith(p) for p in prefixes)]

    groups = {
        # V1
        "PyRadiomics (firstorder/shape2D/glcm/glrlm/glszm/ngtdm)": _grp(["rx_"]),
        "A — Asymmetry centre + principal-axis":   _grp(["abcde_asym_"]),
        "B — Border (contour/Fourier/octants)":    _grp(["abcde_compactness","abcde_border","abcde_convexity","abcde_solidity","abcde_elongation","abcde_radial_dist","abcde_fourier"]),
        "C — Color stats (RGB/HSV/LAB/BWV/reg)":   _grp(["abcde_col_R","abcde_col_G","abcde_col_B","abcde_col_H","abcde_col_S","abcde_col_V","abcde_col_L","abcde_col_A","abcde_col_LAB","abcde_col_grey","abcde_col_rg","abcde_col_rb","abcde_col_bg","abcde_col_std","abcde_col_ent","abcde_col_dark","abcde_col_bright","abcde_col_num","abcde_col_color","abcde_col_blue","abcde_col_reg"]),
        "C — 6-color palette (Argenziano 2003)":   _grp(["col6_"]),
        "C — Color heterogeneity (IQR/MAD/CV)":    _grp(["abcde_col_het_"]),
        "C — Radial colour 2-zone":                _grp(["abcde_radial_"]),
        "D — Diameter / area":                     _grp(["abcde_area_","abcde_diam_"]),
        # V2
        "[V2] fo_R/G/B first-order (13 stats×3)":  _grp(["fo_"]),
        "[V2] LBP on H/S/V (26 bins×3)":           _grp(["lbp_"]),
        "[V2] Gabor texture (grey, 8 filters)":    _grp(["gabor_f"]),
        "[V2] GLCM on H+S channels":               _grp(["glcm_"]),
        "[V2] Shape extras (ecc/extent/euler)":    _grp(["shape_"]),
        "[V2] Border fractal dimension":           _grp(["abcde_border_fractal"]),
        "[V2] Color spatial entropy (4×4 grid)":   _grp(["abcde_color_spatial"]),
        "[V2] Gabor directionality (streaks)":     _grp(["gabor_directionality"]),
        "[V2] Dots / globules detector":           _grp(["abcde_dots_"]),
        "[V2] Radial colour 3-zone":               _grp(["abcde_radial3_"]),
        # V3
        "[V3] Milia-like cysts":                   _grp(["abcde_milia_"]),
        "[V3] Comedo-like openings":               _grp(["abcde_comedo_"]),
        "[V3] Blotch detection":                   _grp(["abcde_blotch_"]),
        "[V3] Lacunae (vascular structures)":      _grp(["abcde_lacunae_"]),
        "[V3] Peppering + scar regression":        _grp(["abcde_peppering","abcde_scar","abcde_regression_type"]),
        "[V3] White / crystalline structures":     _grp(["abcde_white_struct"]),
        "[V3] Vascular / red structures":          _grp(["abcde_vascular_"]),
        "[V3] Satellite topology":                 _grp(["abcde_satellite","abcde_main_fragment"]),
        "[V3] Spatial color zone segmentation":    _grp(["abcde_color_zone"]),
        "[V3] Peripheral structureless area":      _grp(["abcde_periph_"]),
        "[V3] Eccentric dark zone (Chaos clue 1)": _grp(["abcde_eccentric_dark"]),
        "[V3] Quadrant color spread":              _grp(["abcde_quadrant_"]),
        "[V3] Globule spatial regularity":         _grp(["abcde_globule_"]),
    }

    total = 0
    print(f"\n  ✓  {len(rows)} rows × {len(all_keys)} columns → {args.output}\n")
    for grp, cols in groups.items():
        if cols:
            print(f"    {grp:<58s}: {len(cols):>4d}")
            total += len(cols)
    print(f"\n    {'── TOTAL (excl. image_id/split/mask_pixel_count)':<58s}: {total:>4d}")


if __name__ == "__main__":
    main()
