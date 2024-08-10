import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import time
import plotly.graph_objs as go

#------------------------------------------------page configuration--------------------------------------------
st.set_page_config(page_title="Breast Cancer Classifier",
                   page_icon='📊',
                   layout="wide", 
                   initial_sidebar_state="expanded")

#-------------------------------------------------styling-----------------------------------------------------
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #e0e0e0
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


#-------------------------------------------------Load the model----------------------------------------------
@st.cache_resource
def load_breast_cancer_model():
    return load_model('model_eff.keras')

model = load_breast_cancer_model()

#------------------------------------------------preprocess the image-------------------------------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

#----------------------------------------------predict with unknown image detection---------------------------------
def predict_with_unknown(model, img_array, threshold=0.5, unknown_threshold=0.3):
    prediction = model.predict(img_array)[0][0]
    if prediction >= threshold + unknown_threshold:
        return "Malignant", prediction
    elif prediction <= threshold - unknown_threshold:
        return "Benign", prediction
    else:
        return "Unknown Image", prediction




#-----------------------------------------------------Main app--------------------------------------------------
def main():
    st.title("Breast Cancer Image Classifier")
    st.write("Upload an image to classify it as Malignant, Benign, or Unknown")

    #------------------------------------------------Sidebar for image upload
    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        #----------------------------------------------------Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # ------------------------------------------------reprocess the image
        img_array = preprocess_image(image)

        #---------------------------------------------------Make prediction
        if st.button("Classify Image"):
            with st.spinner("Analyzing image..."):
                time.sleep(2)  # Simulate processing time
                prediction, confidence = predict_with_unknown(model, img_array)

            #-----------------------------------------------Display result with animation
            st.success("Classification Complete!")
            st.balloons()

            #--------------------------------------------------i also added nice visuals to improve the UI
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Prediction: {prediction}"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.7], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.7}}))

            st.plotly_chart(fig)


            #----------------------------------------------Additional information-----------------------------
            st.subheader("Interpretation:")
            if prediction == "Malignant":
                st.warning("The image is classified as potentially malignant. Please consult with a healthcare professional for further evaluation.")
            elif prediction == "Benign":
                st.info("The image is classified as likely benign. However, regular check-ups are still recommended.")
            else:
                st.error("The image could not be confidently classified. It may not be a breast cancer image or may require further analysis.")

    #--------------------------------app Information & Disclaimer-----------------------------------------
    st.sidebar.header("About")
    st.sidebar.info("This app uses a deep learning model based on EfficientNetB0 to classify breast cancer images. It can detect if an image is malignant, benign, or unknown.")
    st.sidebar.warning("Disclaimer: This app is for educational purposes only and should not be used as a substitute for professional medical advice.")


    
    #---------------------------------------------------------FOOTER------------------------------------------------
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600&display=swap');

            .footer-container {
                font-family: 'Raleway', sans-serif;
                margin-top: 50px;
                padding: 30px 0;
                width: 100vw;
                position: absolute;
                left: 50%;
                right: 50%;
                margin-left: -50vw;
                margin-right: -50vw;
                # overflow: hidden;
            }

            .footer-content {
                display: flex;
                justify-content: center;
                align-items: center;
                position: relative;
                z-index: 2;
            }

            .footer-text {
                color: #ffffff;
                font-size: 20px;
                font-weight: 300;
                text-align: center;
                margin: 0;
                padding: 0 20px;
                position: relative;
            }

            .footer-link {
                color: #075E54;  /* WhatsApp dark green */
                font-weight: 600;
                text-decoration: none;
                position: relative;
                transition: all 0.3s ease;
                padding: 5px 10px;
                border-radius: 5px;
            }

            .footer-link:hover {
                background-color: rgba(7, 94, 84, 0.1);  /* Slightly darker on hover */
                box-shadow: 0 0 15px rgba(7, 94, 84, 0.2);
            }

            .footer-heart {
                display: inline-block;
                color: #FF0000;  /* Red heart */
                font-size: 35px;
                animation: pulse 1.5s ease infinite;
            }

            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
        </style>

        <div class="footer-container">
            <div class="footer-content">
                <p class="footer-text">
                    Developed by <a href="https://github.com/Abdulraqib20" target="_blank" class="footer-link">BelloCraft</a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
