import streamlit as st
import numpy as np
from PIL import Image
import sys

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import img_to_array
    except (ImportError, AttributeError):
        from keras.models import load_model
        from keras.preprocessing.image import img_to_array
except ImportError as e:
    st.error(f"""
    **TensorFlow is not installed!**
    
    Error: {str(e)}
    
    Please install TensorFlow by running:
    ```
    pip install tensorflow
    ```
    """)
    st.stop()
except Exception as e:
    st.error(f"**Error importing TensorFlow: {str(e)}**")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Classification</h1>', unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #666;'>
            Upload an MRI brain scan image to classify tumor types using deep learning
        </p>
    </div>
""", unsafe_allow_html=True)

# Medical disclaimer
st.warning(
    "⚠️ **Medical Disclaimer:** This application is for educational and research purposes only. "
    "It is NOT a substitute for professional medical diagnosis, advice, or treatment. "
    "Always consult qualified healthcare professionals for medical decisions."
)

# Load model with caching
@st.cache_resource
def load_brain_tumor_model():
    """Load the brain tumor classification model"""
    try:
        model = load_model("brain_tumor-model.h5", compile=False)
        return model
    except FileNotFoundError:
        st.error("Model file 'brain_tumor-model.h5' not found. Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
with st.spinner("Loading model..."):
    model = load_brain_tumor_model()

if model is None:
    st.stop()

# Sidebar - Model Information
st.sidebar.header("📊 Model Information")
try:
    input_shape = model.input_shape
    output_shape = model.output_shape
    st.sidebar.write(f"**Input Shape:** `{input_shape}`")
    st.sidebar.write(f"**Output Shape:** `{output_shape}`")
    
    # Detect expected input channels and size
    if len(input_shape) == 4:  # (batch, height, width, channels)
        expected_channels = input_shape[-1]
        input_height, input_width = input_shape[1:3]
    elif len(input_shape) == 3:  # (height, width, channels)
        expected_channels = input_shape[-1]
        input_height, input_width = input_shape[0:2]
    else:
        expected_channels = 1
        input_height, input_width = 224, 224
        st.sidebar.warning("Could not detect input shape, using defaults")
    
    st.sidebar.write(f"**Expected Channels:** {expected_channels}")
    st.sidebar.write(f"**Input Size:** {input_width}×{input_height}")
except Exception as e:
    expected_channels = 1
    input_height, input_width = 224, 224
    st.sidebar.warning(f"Could not detect input shape: {str(e)}")
    st.sidebar.write("Using defaults: 224×224, 1 channel")

# Class labels
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

st.sidebar.header("📋 Classification Classes")
for i, cls in enumerate(CLASSES, 1):
    st.sidebar.write(f"{i}. {cls.replace('_', ' ').title()}")

# Main content area
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an MRI brain scan image",
    type=["jpg", "jpeg", "png", "bmp", "tiff", "dcm"],
    help="Upload a brain MRI scan image in JPG, PNG, BMP, TIFF, or DICOM format"
)

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Convert image based on expected channels
        if expected_channels == 1:
            if image.mode != "L":
                image = image.convert("L")
        else:
            if image.mode != "RGB":
                image = image.convert("RGB")
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            st.caption(f"Image size: {image.size[0]}×{image.size[1]} | Mode: {image.mode}")
        
        # Preprocess image
        image_resized = image.resize((input_width, input_height))
        
        # Convert to array
        img_array = img_to_array(image_resized)
        
        # Ensure correct channel dimension
        if expected_channels == 1:
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            elif img_array.shape[-1] != 1:
                img_array = np.mean(img_array, axis=-1, keepdims=True)
        else:
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] != 3:
                image_resized = image_resized.convert("RGB")
                img_array = img_to_array(image_resized)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Display preprocessed image
        with col2:
            st.subheader("🔄 Preprocessed Image")
            display_img = np.squeeze(img_array[0])
            if expected_channels == 1:
                st.image(display_img, caption=f"Preprocessed ({input_width}×{input_height}, Grayscale)", 
                        use_container_width=True, clamp=True)
            else:
                st.image(display_img, caption=f"Preprocessed ({input_width}×{input_height}, RGB)", 
                        use_container_width=True, clamp=True)
            st.caption(f"Array shape: {img_array.shape}")
        
        st.markdown("---")
        
        # Prediction button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔍 Predict Tumor Type", type="primary", use_container_width=True)
        
        if predict_button:
            with st.spinner("🤖 Analyzing MRI image... This may take a few seconds."):
                try:
                    # Make prediction
                    predictions = model.predict(img_array, verbose=0)[0]
                    
                    # Get predicted class
                    pred_idx = np.argmax(predictions)
                    confidence = predictions[pred_idx] * 100
                    
                    # Display main prediction result
                    st.markdown("---")
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col_pred1, col_pred2 = st.columns([2, 1])
                    
                    with col_pred1:
                        st.markdown(f"### 🩺 Prediction Result")
                        predicted_class = CLASSES[pred_idx].replace('_', ' ').title()
                        st.markdown(f"**{predicted_class}**")
                    
                    with col_pred2:
                        st.markdown(f"### 📈 Confidence")
                        st.markdown(f"**{confidence:.2f}%**")
                        if confidence >= 80:
                            st.success("High confidence")
                        elif confidence >= 60:
                            st.warning("Moderate confidence")
                        else:
                            st.info("Low confidence")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Visualizations
                    st.subheader("📊 Probability Distribution")
                    
                    # Bar chart
                    prob_data = {
                        "Class": [cls.replace("_", " ").title() for cls in CLASSES],
                        "Probability (%)": [p * 100 for p in predictions]
                    }
                    st.bar_chart(prob_data, x="Class", y="Probability (%)", height=300)
                    
                    # Detailed results table
                    st.subheader("📋 Detailed Class Probabilities")
                    
                    # Create a dataframe for better display
                    import pandas as pd
                    results_df = pd.DataFrame({
                        "Class": [cls.replace("_", " ").title() for cls in CLASSES],
                        "Probability (%)": [f"{p * 100:.2f}%" for p in predictions],
                        "Raw Score": [f"{p:.4f}" for p in predictions]
                    })
                    
                    # Highlight the predicted class
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Top 2 predictions
                    st.subheader("🏆 Top 2 Predictions")
                    top2_indices = np.argsort(predictions)[-2:][::-1]
                    
                    for rank, idx in enumerate(top2_indices, 1):
                        medal = "🥇" if rank == 1 else "🥈"
                        st.markdown(f"{medal} **{rank}. {CLASSES[idx].replace('_', ' ').title()}** - {predictions[idx]*100:.2f}%")
                    
                    # Additional insights
                    st.subheader("💡 Insights")
                    if CLASSES[pred_idx] == "no_tumor":
                        st.success("✅ No tumor detected in the MRI scan.")
                    else:
                        st.info(f"⚠️ Potential {CLASSES[pred_idx].replace('_', ' ')} detected. Please consult with a medical professional for further evaluation.")
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.exception(e)

else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload an MRI brain scan image to get started.")
    
    # Example usage
    with st.expander("ℹ️ How to use this application"):
        st.markdown("""
        1. **Upload an Image**: Click the file uploader above and select a brain MRI scan image
        2. **Review Preprocessing**: The app will automatically resize and preprocess your image
        3. **Get Prediction**: Click the "Predict Tumor Type" button to analyze the image
        4. **View Results**: Review the prediction, confidence scores, and probability distribution
        
        **Supported formats**: JPG, JPEG, PNG, BMP, TIFF, DICOM
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>Brain Tumor Classification Model | Built with Streamlit & TensorFlow</p>
        <p style='font-size: 0.8rem;'>Model: brain_tumor-model.h5</p>
    </div>
    """,
    unsafe_allow_html=True
)
