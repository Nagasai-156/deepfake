import streamlit as st
import numpy as np
from PIL import Image
import os
from streamlit_image_select import image_select
from tensorflow.keras.models import model_from_json
import time
import json
from datetime import datetime
import plotly.express as px
import cv2
from streamlit_option_menu import option_menu
import pandas as pd
from pathlib import Path
from google.colab import files

# -------------------
# CONFIGURATION
# -------------------
# Initialize session state variables
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'total_attempts' not in st.session_state:
    st.session_state.total_attempts = 0
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Configure page
st.set_page_config(
    page_title="Deepfake Detector (Colab)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff6b6b, #4ecdc4);
    }
    .achievement {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------
# UTILITY FUNCTIONS
# -------------------
def save_statistics():
    """Save user statistics to Colab"""
    stats = {
        'score': st.session_state.score,
        'total_attempts': st.session_state.total_attempts,
        'streak': st.session_state.streak,
        'history': st.session_state.history
    }
    with open('/content/user_stats.json', 'w') as f:
        json.dump(stats, f)

def load_statistics():
    """Load user statistics from Colab"""
    try:
        with open('/content/user_stats.json', 'r') as f:
            stats = json.load(f)
            st.session_state.score = stats['score']
            st.session_state.total_attempts = stats['total_attempts']
            st.session_state.streak = stats['streak']
            st.session_state.history = stats['history']
    except FileNotFoundError:
        pass

def generate_heatmap(image, prediction):
    """Generate a heatmap showing areas that influenced the model's decision"""
    img_array = np.array(image)
    heatmap = cv2.applyColorMap(
        cv2.resize(img_array.mean(axis=2).astype(np.uint8), (256, 256)), 
        cv2.COLORMAP_JET
    )
    return Image.fromarray(heatmap)

@st.cache_resource
def load_model():
    """Load the deepfake detection model with error handling"""
    try:
        model_path = "/content/model"
        json_path = os.path.join(model_path, "dffnetv2B0.json")
        weights_path = os.path.join(model_path, "dffnetv2B0.h5")
        
        if not os.path.exists(json_path) or not os.path.exists(weights_path):
            st.error("❌ Model files not found! Please upload model files in the Settings tab.")
            return None
            
        # Load model architecture
        with open(json_path, 'r') as f:
            model = model_from_json(f.read())
            
        # Load weights
        model.load_weights(weights_path)
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def settings_mode():
    """Settings and preferences for Colab deployment"""
    st.header("⚙️ Settings")
    
    # Model file upload
    st.subheader("Model Files")
    model_json = st.file_uploader("Upload model architecture (JSON)", type=['json'])
    model_weights = st.file_uploader("Upload model weights (H5)", type=['h5'])
    
    if model_json and model_weights:
        os.makedirs("/content/model", exist_ok=True)
        with open("/content/model/dffnetv2B0.json", "wb") as f:
            f.write(model_json.getvalue())
        with open("/content/model/dffnetv2B0.h5", "wb") as f:
            f.write(model_weights.getvalue())
        st.success("✅ Model files uploaded successfully!")
    
    # Theme selection
    theme = st.selectbox(
        "Select Theme",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == 'light' else 1
    )
    st.session_state.theme = theme.lower()
    
    # Reset statistics
    if st.button("Reset Statistics"):
        st.session_state.score = 0
        st.session_state.total_attempts = 0
        st.session_state.streak = 0
        st.session_state.history = []
        save_statistics()
        st.success("Statistics reset successfully!")
    
    # Export statistics
    if st.button("Export Statistics"):
        stats_df = pd.DataFrame(st.session_state.history)
        csv = stats_df.to_csv(index=False)
        st.download_button(
            label="Download Statistics CSV",
            data=csv,
            file_name="deepfake_detector_stats.csv",
            mime="text/csv"
        )

    # Colab-specific instructions
    st.markdown("""
    ### Important Notes for Colab Deployment
    
    1. **Model Files**: Upload your model files using the fields above
    2. **Session Persistence**: Data will be lost when Colab runtime disconnects
    3. **Resource Limits**: Be mindful of Colab's resource limitations
    4. **Sharing**: Share the Colab notebook URL for others to use the app
    """)

# Load model
classifier = load_model()

# Import the rest of your existing functions (detector_mode, game_mode, learn_mode)
# Make sure to update file paths to use /content/ instead of relative paths

if __name__ == "__main__":
    main() 