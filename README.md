# 🧠 Brain Tumor Classification using Deep Learning (Custom CNN)

## 📌 Project Overview

This project focuses on **Brain Tumor Classification from MRI images** using a **Custom Convolutional Neural Network (CNN)** built from scratch with **TensorFlow and Keras**.

The model classifies MRI scans into **four categories**:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

The goal of this project is to build an **accurate, well-generalized, and explainable deep learning model** and deploy it using **Streamlit** for real-time predictions.

---

## 🎯 Key Highlights

* ✅ Custom CNN architecture (no transfer learning)
* ✅ Proper train–validation–test split
* ✅ Data augmentation tuned for medical MRI images
* ✅ Learning rate scheduling and early stopping
* ✅ Achieved **85.2% accuracy on unseen test data**
* ✅ Deployed using Streamlit with live predictions

---

## 📂 Dataset Information

* **Source:** Kaggle – Brain Tumor MRI Dataset
* **Image Type:** MRI scans (visually grayscale, stored as RGB)
* **Classes:** 4

### Dataset Structure

```
Dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

### Dataset Split Strategy

* **Training data:** Used for model learning
* **Validation data:** 20% split from training set
* **Test data:** Completely unseen during training

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Google Colab
* Streamlit

---

## 🧠 Model Architecture (Custom CNN)

The model was designed from scratch using convolutional blocks optimized for medical images:

* Convolution + ReLU activation
* Batch Normalization
* MaxPooling layers
* Dropout for regularization
* Dense layers for classification
* Softmax output layer

### Input Shape

```
(224, 224, 3)
```

> MRI images appear grayscale but are stored and processed as RGB images with identical channels.

---

## 🔁 Data Preprocessing & Augmentation

MRI-safe augmentation techniques were applied:

* Image resizing and normalization
* Small rotations
* Zooming
* Width and height shifting
* Horizontal flipping

Aggressive augmentation was avoided to preserve medical image integrity.

---

## ⚙️ Training Strategy

* **Optimizer:** Adam
* **Learning Rate:** 3e-4
* **Loss Function:** Categorical Crossentropy
* **Callbacks:**

  * EarlyStopping (restore best weights)
  * ReduceLROnPlateau

The best model was automatically restored based on validation loss.

---

## 📈 Model Performance

| Metric                   | Value     |
| ------------------------ | --------- |
| Training Accuracy        | ~92%      |
| Best Validation Accuracy | **82.3%** |
| **Final Test Accuracy**  | **85.2%** |
| Test Loss                | 0.6078    |

---

## 🖼️ Training Results Visualization

The following plot shows **training and validation accuracy during model training**:

![Training and Validation Accuracy](assets/Accuracy\&Validation.png)

---

## 🌐 Streamlit Deployment

The trained model is deployed using **Streamlit**, providing an interactive interface for real-time MRI classification.

### Application Features

* Upload MRI images
* Real-time tumor classification
* Clear display of predicted class

---

## 🎥 Streamlit Application Demo Video

Click the link below to watch the Streamlit app running live:

[▶️ Watch Streamlit Demo Video](assets/streamlit_reacording.mp4)

> ℹ️ GitHub does not play videos inline. The link is clickable and downloadable.

---

## 💾 Model Saving

The trained model is saved in both formats:

```
brain_tumor_model.h5
brain_tumor_model.keras
```

* `.h5` → Deployment & compatibility
* `.keras` → Future-proof format

---

## 🧪 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sayeem3051/MRI-Classification1.git
cd MRI-Classification1
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🧠 Learning Outcomes

* Built a CNN from scratch for medical imaging
* Applied correct validation strategies
* Controlled overfitting using callbacks
* Deployed a deep learning model using Streamlit

---

## 🚀 Future Improvements

* Use transfer learning (MobileNetV2 / ResNet)
* Add Grad-CAM for model explainability
* Improve class-wise performance
* Deploy as a public web application

---

## 🏁 Conclusion

This project demonstrates a **complete deep learning pipeline** from data preprocessing and model training to evaluation and deployment. The achieved performance validates the effectiveness of a well-designed **Custom CNN** for brain tumor classification.

---

⭐ If you find this project useful, consider starring the repository!
