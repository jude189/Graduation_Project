# Graduation_Project
# 🔬 Skin Cancer Detection — Automated Lesion Segmentation & Classification

> A deep learning pipeline for the automated detection and classification of skin cancer lesions from dermoscopic images.

---

## 📌 Project Overview

Skin cancer is among the most common and dangerous forms of cancer worldwide. Early and accurate detection is critical for patient survival. This project presents a **complete end-to-end pipeline** that processes raw dermoscopic images through four intelligent stages — from raw image segmentation all the way to final cancer classification.

This work was developed as a graduation project and demonstrates practical application of computer vision and machine learning techniques in the medical imaging domain.

---

## 🧠 Pipeline Architecture

```
Raw Dermoscopic Image
        │
        ▼
┌─────────────────────┐
│   1. Segmentation   │  ← Isolates the lesion from healthy skin
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   2. Cleaning       │  ← Removes artifacts, hair, noise
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│ 3. Feature Extract. │  ← Extracts shape, color & texture features
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  4. Classification  │  ← Predicts benign vs. malignant
└─────────────────────┘
        │
        ▼
   Diagnosis Output
```

---

## 📁 Repository Structure

```
skin-cancer-detection/
│
├── 1_segmentation/
│   └── segmentation.py         # Lesion boundary detection
│
├── 2_cleaning/
│   └── cleaning.py             # Preprocessing & artifact removal
│
├── 3_feature_extraction/
│   └── feature_extraction.py   # ABCD features, texture, color
│
├── 4_classification/
│   └── classification.py       # ML/DL classification model
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ Modules

### 1 — Segmentation
Detects and isolates the skin lesion region from the surrounding healthy tissue using image processing techniques such as thresholding, contour detection, or deep segmentation models (e.g. U-Net). Produces a binary mask of the lesion area.

### 2 — Cleaning
Prepares the segmented image for analysis by removing noise, hair artifacts, reflections, and other dermoscopic-specific distortions. Applies filters and morphological operations to enhance image quality.

### 3 — Feature Extraction
Extracts clinically meaningful features from the cleaned lesion image, inspired by the **ABCD dermatology rule**:
- **A**symmetry — shape irregularity
- **B**order — smoothness vs. jaggedness
- **C**olor — color variation and distribution
- **D**iameter — size estimation

Additional texture features are extracted using methods like LBP (Local Binary Patterns) or GLCM (Gray-Level Co-occurrence Matrix).

### 4 — Classification
Feeds the extracted features into a trained machine learning or deep learning model to predict whether a lesion is **benign** or **malignant**. Includes model evaluation metrics such as accuracy, precision, recall, and F1-score.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Image Processing | OpenCV, Scikit-Image |
| Machine Learning | Scikit-learn, TensorFlow / PyTorch |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |

---

## 👋 Welcome

Welcome to the source code repository for our graduation project! This repo contains the four core code modules of our skin cancer detection pipeline. Each module corresponds to a stage in the processing pipeline and can be explored independently.

If you're reviewing this work — thank you for your time. Every line of code here represents our effort to apply intelligent technology to a real-world medical challenge that affects millions of people.

---

## 👨‍🎓 About

This project was developed as a **graduation project** in the field of medical image analysis and computer vision.

> *"Bringing intelligent tools to the fight against skin cancer — one pixel at a time."*

---

## 📄 License

This project is for academic and educational purposes.
