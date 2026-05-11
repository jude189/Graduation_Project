<h1 align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=28&duration=3000&pause=1000&color=8A2BE2&center=true&vCenter=true&width=600&lines=🔬+Skin+Cancer+Detection;AI-Powered+Lesion+Analysis;From+Image+→+Diagnosis;Computer+Vision+in+Healthcare" />
</h1>

<p align="center">
  <b>🎓 Graduation Project — Automated Lesion Segmentation & Classification using AI</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Type-Graduation%20Project-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Medical%20AI-red?style=for-the-badge"/>
</p>

---

## 🚀 Welcome

✨ **Welcome to our Graduation Project repository!**

This project implements a complete **end-to-end AI pipeline** for skin cancer detection — from raw dermoscopy images all the way to a Benign/Malignant classification result.

> 💡 *Goal: Assist early diagnosis using artificial intelligence and computer vision.*

---

## 🧠 How It Works — The Pipeline

```
🖼️ Input Image
      ↓
  ✂️ Segmentation      → Isolate the lesion from surrounding skin
      ↓
  🧹 Cleaning          → Remove noise & enhance image quality
      ↓
  📊 Feature Extraction → ABCD Rule + Texture Features (LBP, GLCM)
      ↓
  🤖 Classification    → Predict: Benign ✅ or Malignant ❌
      ↓
  📋 Result
```

---

## 📁 Project Structure

```
skin-cancer-detection/
│
├── 📂 segmentation/
│   └── segmentation.py        # Lesion boundary detection
│
├── 📂 cleaning/
│   └── cleaning.py            # Noise removal & image enhancement
│
├── 📂 feature_extraction/
│   └── feature_extraction.py  # ABCD rule + LBP + GLCM features
│
├── 📂 classification/
│   └── classification.py      # ML/DL model — Benign vs Malignant
│
└── 📄 README.md
```

> 📌 **Note:** Each module is self-contained and maps directly to one stage of the pipeline above.

---

## ⚙️ System Modules

<details>
<summary>🟦 Step 1 — Segmentation</summary>

### ✂️ Lesion Segmentation

**Goal:** Detect and isolate the skin lesion from the rest of the image.

- Detects lesion boundaries using image processing / deep learning
- Separates the lesion region from surrounding healthy skin
- Produces a binary mask used by all downstream modules
- Methods: Thresholding, U-Net, or Active Contours

</details>

---

<details>
<summary>🟩 Step 2 — Cleaning</summary>

### 🧹 Image Preprocessing & Noise Removal

**Goal:** Clean the segmented image so features can be extracted accurately.

- Removes artifacts (hair, bubbles, reflections) ⚡
- Applies filters to reduce noise (Gaussian, Median)
- Enhances contrast for better feature visibility
- Ensures consistent image quality across the dataset

</details>

---

<details>
<summary>🟧 Step 3 — Feature Extraction</summary>

### 📊 Feature Extraction

**Goal:** Extract meaningful numerical features from the cleaned lesion.

#### 🔬 ABCD Rule (Dermatology Standard):
| Feature | Description |
|--------|-------------|
| 🔺 **Asymmetry** | Is the lesion symmetric? |
| 🔳 **Border** | Are the edges irregular or smooth? |
| 🎨 **Color** | How many distinct colors are present? |
| 📏 **Diameter** | Is it larger than 6mm? |

#### 🧮 Texture Features:
- **LBP** (Local Binary Patterns) — captures micro-texture
- **GLCM** (Gray-Level Co-occurrence Matrix) — captures spatial relationships

</details>

---

<details>
<summary>🟥 Step 4 — Classification</summary>

### 🤖 Final Classification

**Goal:** Use extracted features to predict whether the lesion is cancerous.

#### 🎯 Predictions:
- ✅ **Benign** — Non-cancerous lesion
- ❌ **Malignant** — Cancerous lesion (requires medical attention)

#### 📈 Evaluation Metrics:
| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correctness |
| **Precision** | How many predicted positives are actually positive |
| **Recall** | How many actual positives were caught |
| **F1-Score** | Harmonic mean of Precision & Recall |

</details>

---

## 🛠️ Technologies Used

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
</p>

---

## 🎯 Why This Project Matters

| 💡 | Benefit |
|----|---------|
| 🧠 | Early detection dramatically increases survival rates |
| ⚡ | AI enables faster screening than manual diagnosis |
| 🌍 | Accessible tool that can support clinics with limited specialists |
| 🤖 | Bridges computer vision and real-world healthcare impact |

---

## 💬 Final Message

> 🧬 *"Fighting skin cancer with the power of pixels and AI."*

---

## 📄 License

Academic use only — Graduation Project 2024/2025.
