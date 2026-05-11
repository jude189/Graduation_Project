# 🎓 Graduation Project

<h1 align="center">
  🔬 Skin Cancer Detection
</h1>

<p align="center">
  <b>Automated Lesion Segmentation & Classification using AI</b>
</p>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=00C2FF&center=true&vCenter=true&width=500&lines=Welcome+to+our+Graduation+Project!;AI+for+Skin+Cancer+Detection;From+Image+→+Diagnosis;Built+with+Computer+Vision+%F0%9F%94%AC" />
</p>

---

## 🚀 Welcome

✨ **Welcome to our Graduation Project repository!**

This project presents a **complete AI-powered pipeline** for detecting skin cancer from dermoscopic images.

💡 Our goal:

> Use **Artificial Intelligence** to assist in **early diagnosis** and potentially save lives.

---

## 📌 Project Overview

Skin cancer is one of the most common and dangerous diseases worldwide.

This system processes an image through **4 intelligent stages**:

➡️ From raw image
➡️ To cleaned lesion
➡️ To extracted features
➡️ To final diagnosis

---

## 🧠 Pipeline Architecture

<p align="center">
  <img src="https://img.shields.io/badge/STEP%201-Segmentation-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/STEP%202-Cleaning-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/STEP%203-Feature%20Extraction-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/STEP%204-Classification-red?style=for-the-badge"/>
</p>

```
🖼️ Image → ✂️ Segmentation → 🧹 Cleaning → 📊 Features → 🤖 Classification → 📋 Result
```

---

## 📁 Project Structure

```bash
skin-cancer-detection/
│
├── 1_segmentation/          # Lesion detection
│   └── segmentation.py
│
├── 2_cleaning/              # Noise & artifact removal
│   └── cleaning.py
│
├── 3_feature_extraction/    # ABCD + texture features
│   └── feature_extraction.py
│
├── 4_classification/        # ML/DL model
│   └── classification.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ System Modules

### 🟦 1. Segmentation

* Detects lesion boundaries
* Separates cancer area from healthy skin
* Techniques:

  * Thresholding
  * Contours
  * Deep Learning (U-Net)

---

### 🟩 2. Cleaning

* Removes:

  * Hair 🧵
  * Noise ⚡
  * Reflections ✨
* Improves image quality for analysis

---

### 🟧 3. Feature Extraction

Based on **ABCD Rule**:

* 🔺 Asymmetry
* 🔳 Border
* 🎨 Color
* 📏 Diameter

- Texture features:

* LBP (Local Binary Patterns)
* GLCM (Gray-Level Co-occurrence Matrix)

---

### 🟥 4. Classification

* Predicts:

  * ✅ Benign
  * ❌ Malignant
* Uses ML / DL models
* Outputs:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## 🛠️ Technologies Used

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-green?style=for-the-badge&logo=opencv"/>
  <img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/PyTorch-AI-red?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow?style=for-the-badge"/>
</p>

---

## 🎯 Why This Project Matters

🧠 Early detection = higher survival rates
⚡ Faster diagnosis
🤖 AI-assisted medical support

---

## 👨‍🎓 About

This project was developed as a **graduation project** in:

* Medical Image Processing
* Computer Vision
* Artificial Intelligence

---

## 💬 Final Message

> 🧬 *“Fighting skin cancer with the power of pixels and AI.”*

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 📢 Share it

---

## 📄 License

This project is for academic and educational use only.
