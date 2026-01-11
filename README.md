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
* ✅ Model ready for deployment and real-world inference

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
* **Validation data:** 20% split from training set (used for tuning & early stopping)
* **Test data:** Completely unseen during training (used for final evaluation)

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

* Convolution layers with ReLU activation
* Batch Normalization
* MaxPooling layers
* Dropout for regularization
* Dense layers for classification
* Softmax output layer

### Input Shape

```
(224, 224, 3)
```

> Although MRI images appear grayscale, they are stored and processed as RGB images with identical channels to match CNN input requirements.

---

## 🔁 Data Preprocessing & Augmentation

To improve generalization, **MRI-safe augmentation** techniques were applied:

* Image resizing and normalization
* Small rotations
* Zooming
* Width and height shifting
* Horizontal flipping

Aggressive augmentations were avoided to preserve medical image integrity.

---

## ⚙️ Training Strategy

* **Optimizer:** Adam
* **Initial Learning Rate:** 3e-4
* **Loss Function:** Categorical Crossentropy
* **Callbacks Used:**

  * EarlyStopping (monitored validation loss)
  * ReduceLROnPlateau (dynamic learning rate reduction)

Early stopping ensured the model did not overfit and restored the **best-performing weights automatically**.

---

## 📈 Model Performance

### Best Results

| Metric                   | Value     |
| ------------------------ | --------- |
| Training Accuracy        | ~92%      |
| Best Validation Accuracy | **82.3%** |
| **Final Test Accuracy**  | **85.2%** |
| Test Loss                | 0.6078    |

The test accuracy confirms strong **generalization on unseen MRI scans**.

---

## 📊 Result Visualization

The following plots were used to analyze training behavior:

* Training vs Validation Accuracy
* Training vs Validation Loss
* Best epoch marked using validation loss

These plots show smooth convergence and controlled overfitting.

---

## 💾 Model Saving

The trained model is saved in both formats for flexibility:

```
brain_tumor_model.h5
brain_tumor_model.keras
```

* `.h5` → Deployment & compatibility
* `.keras` → Future-proof format

---

## 🌐 Deployment (Streamlit)

The trained model is deployed using **Streamlit**, providing an interactive web interface for real-time brain tumor classification.

### 🔍 Application Features

* Upload MRI images through the UI
* Real-time prediction using the trained CNN model
* Displays predicted tumor class instantly

### 🖼️ Demo Preview

Below are demo assets showcasing the running Streamlit application:

#### 📸 Application Screenshot

Below is a screenshot showing model accuracy & validation performance during training:

````markdown
![Training and Validation Accuracy](assets/Accuracy&Validation.png)
```markdown
![Streamlit App Screenshot](assets/streamlit_app.png)
````

#### 🎥 Application Demo Video

A demo video showing the **Streamlit application running live**:

````markdown
[▶️ Watch Streamlit Demo Video](assets/streamlit_reacording.mp4)
```markdown
[▶️ Watch Streamlit Demo Video](assets/streamlit_demo.mp4)
````

> 💡 Tip: If GitHub does not render the video inline, it will still be downloadable and clickable.

---

## 🧪 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
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

Through this project, I gained hands-on experience in:

* Building CNNs from scratch
* Medical image preprocessing
* Handling overfitting with callbacks
* Proper evaluation using unseen test data
* Deploying deep learning models

---

## 🚀 Future Improvements

* Use transfer learning (MobileNetV2 / ResNet)
* Add Grad-CAM for model explainability
* Improve class-wise performance
* Deploy as a cloud-based web application

---

## 🏁 Conclusion

This project demonstrates a **complete deep learning pipeline** — from dataset handling and model training to evaluation and deployment. The achieved performance validates the effectiveness of a well-designed **Custom CNN** for medical image classification.

---

⭐ If you like this project, don’t forget to star the repository! ⭐
