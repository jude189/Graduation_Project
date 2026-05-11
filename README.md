# 👋 Hey, Welcome to Our Graduation Project!

We're genuinely excited you're here. This repo is the result of months of late nights, debugging sessions, and a whole lot of coffee — and it's something we're really proud of.

> *"Bringing intelligent tools to the fight against skin cancer — one pixel at a time."*

---

# 🔬 Skin Cancer Detection — Automated Lesion Segmentation & Classification

A deep learning pipeline for the automated detection and classification of skin cancer lesions from dermoscopic images. We built something that actually matters — a system that could one day help doctors catch cancer earlier and save lives. No pressure, right?

---

## 🤔 So... What Does This Thing Actually Do?

Imagine you're a doctor looking at a suspicious mole. You take a close-up dermoscopic photo. Now what? Our pipeline takes that image and runs it through 4 smart stages to give you a diagnosis:

```
📸 Raw Dermoscopic Image
        │
        ▼
🔍 Stage 1 — Segmentation
   "Where exactly is the lesion?"
        │
        ▼
🧹 Stage 2 — Cleaning
   "Let's remove that hair, glare & noise..."
        │
        ▼
📐 Stage 3 — Feature Extraction
   "Hmm, the shape looks irregular. Color's off too."
        │
        ▼
🧠 Stage 4 — Classification
   "Benign or Malignant? Here's our verdict."
        │
        ▼
✅ Diagnosis Output
```

That's it. Raw image in → intelligent diagnosis out.

---

## 📁 What's in the Repo?

```
main/
│
├── 1_segmentation/
│   └── segmentation.py         # Finds the lesion boundary
│
├── 2_cleaning/
│   └── cleaning.py             # Cleans up noise, hair, artifacts
│
├── 3_feature_extraction/
│   └── feature_extraction.py   # Extracts the ABCD clinical features
│
├── 4_classification/
│   └── classification.py       # The brain — makes the final call
│
├── requirements.txt            # Everything you need to get started
└── README.md                   # You are here 📍
```

Each module is **self-contained** — feel free to explore any stage independently!

---

## ⚙️ Breaking Down the Pipeline

### 🔍 Stage 1 — Segmentation
*"Find the lesion, ignore everything else."*

Using techniques like thresholding, contour detection, and deep segmentation models (think U-Net), we isolate the lesion from healthy surrounding skin. The output? A clean binary mask of exactly where the lesion is.

---

### 🧹 Stage 2 — Cleaning
*"Because dermoscopic images are surprisingly messy."*

Hair, reflections, air bubbles, smudges — dermoscopic images have it all. This stage runs the masked lesion through a series of filters and morphological operations to produce a clean, analysis-ready image.

---

### 📐 Stage 3 — Feature Extraction
*"What makes a lesion suspicious? Glad you asked."*

We extract features inspired by the classic **ABCD dermatology rule** that doctors use in real clinics:

| Feature | What We Measure |
|---|---|
| **A** — Asymmetry | Is the shape irregular? |
| **B** — Border | Are the edges jagged or smooth? |
| **C** — Color | Is there unusual color variation? |
| **D** — Diameter | How large is the lesion? |

On top of that, we also pull texture features using **LBP** (Local Binary Patterns) and **GLCM** (Gray-Level Co-occurrence Matrix) for even deeper analysis.

---

### 🧠 Stage 4 — Classification
*"Benign or malignant — the moment of truth."*

All those extracted features get fed into a trained ML/DL model that makes the final prediction. We evaluate everything with accuracy, precision, recall, and F1-score so you know exactly how well it performs — no black boxes here.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Image Processing | OpenCV, Scikit-Image |
| Machine Learning | Scikit-learn, TensorFlow / PyTorch |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/skin-cancer-detection.git
cd skin-cancer-detection

# Install dependencies
pip install -r requirements.txt

# Run any module independently
python 1_segmentation/segmentation.py
python 2_cleaning/cleaning.py
python 3_feature_extraction/feature_extraction.py
python 4_classification/classification.py
```

---

## 👨‍🎓 About This Project

This was built as a **graduation project** in the field of medical image analysis and computer vision.

We wanted to work on something that actually meant something — not just a checkbox project, but real technology applied to a real problem that affects millions of people every year. Skin cancer is one of the most common cancers globally, and early detection dramatically improves survival rates.

Every function, every model, every line of code here was written with that in mind.

If you're reviewing this work — thank you for your time. We hope it shows. 🙏

---

## 📄 License

This project is for academic and educational purposes.
