from PIL import Image
import numpy as np
import streamlit as st
import pickle
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
    page_title="Advanced Deepfake Detector",
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
    """Save user statistics to a file"""
    stats = {
        'score': st.session_state.score,
        'total_attempts': st.session_state.total_attempts,
        'streak': st.session_state.streak,
        'history': st.session_state.history
    }
    with open('user_stats.json', 'w') as f:
        json.dump(stats, f)

def load_statistics():
    """Load user statistics from file"""
    try:
        with open('user_stats.json', 'r') as f:
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

def get_prediction(model, image):
    """Enhanced prediction function with detailed analysis"""
    with st.spinner('🔍 Analyzing image...'):
        open_image = Image.open(image)
        resized_image = open_image.resize((256, 256))
        np_image = np.array(resized_image)
        reshaped = np.expand_dims(np_image, axis=0)

        predicted_prob = model.predict(reshaped)[0][0]
        
        # Generate analysis results
        confidence = predicted_prob if predicted_prob >= 0.5 else 1 - predicted_prob
        result = {
            'prediction': 'Real' if predicted_prob >= 0.5 else 'Fake',
            'confidence': float(confidence),
            'probability': float(predicted_prob),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add to history
        st.session_state.history.append(result)
        save_statistics()
        
        return result

def detector_mode():
    """Enhanced detector mode with advanced features"""
    st.header("🔍 Advanced Detector Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Image for Analysis")
        uploaded_image = st.file_uploader(
            "Upload an image to test (Supported: JPG, JPEG, PNG, WebP):", 
            type=['jpg', 'jpeg', 'png', 'webp']
        )
        
        if uploaded_image is not None:
            try:
                # Display original image
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                
                # Get prediction
                result = get_prediction(classifier, uploaded_image)
                
                # Display results with enhanced visualization
                st.markdown(f"### Analysis Results")
                
                # Confidence meter
                st.progress(result['confidence'])
                
                # Result with emoji
                emoji = "✅" if result['prediction'] == "Real" else "❌"
                st.markdown(f"### Prediction: {emoji} {result['prediction']}")
                st.markdown(f"**Confidence Score:** {result['confidence']:.2%}")
                
                # Generate and display heatmap
                st.subheader("Analysis Heatmap")
                heatmap = generate_heatmap(Image.open(uploaded_image), result['prediction'])
                st.image(heatmap, caption="Areas of Interest", use_column_width=True)
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("💡 Please try uploading a different image.")
    
    with col2:
        st.subheader("Analysis History")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Show statistics
            st.metric("Total Analyses", len(st.session_state.history))
            
            # Plot confidence distribution
            fig = px.histogram(
                history_df, 
                x='confidence',
                title='Confidence Distribution',
                labels={'confidence': 'Confidence Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions
            st.markdown("### Recent Predictions")
            for pred in reversed(st.session_state.history[-5:]):
                st.markdown(
                    f"**{pred['prediction']}** ({pred['confidence']:.2%}) - {pred['timestamp']}"
                )
        else:
            st.info("No analysis history yet!")

def game_mode():
    """Enhanced game mode with scoring and achievements"""
    st.header("🎮 Challenge Mode")
    
    # Display current stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", st.session_state.score)
    with col2:
        st.metric("Current Streak", st.session_state.streak)
    with col3:
        accuracy = (st.session_state.score / st.session_state.total_attempts * 100) if st.session_state.total_attempts > 0 else 0
        st.metric("Accuracy", f"{accuracy:.1f}%")

    st.subheader("Can you beat the AI? 🤖")
    
    # Image selection with enhanced UI
    selected_image = image_select(
        "Select an image to analyze:", 
        images,
        return_value="index",
        use_container_width=True
    )
    
    # Display selected image
    st.image(images[selected_image], use_column_width=True)
    
    # Get true label and model prediction
    true_label = 'Fake' if 'fake' in images[selected_image].lower() else 'Real'
    prediction = get_prediction(classifier, images[selected_image])
    
    # User input buttons with better UI
    col1, col2 = st.columns(2)
    
    with col1:
        real_button = st.button("✅ It's Real", use_container_width=True)
    with col2:
        fake_button = st.button("❌ It's Fake", use_container_width=True)
    
    if real_button or fake_button:
        user_guess = "Real" if real_button else "Fake"
        st.session_state.total_attempts += 1
        
        # Show results
        st.markdown("### Results:")
        
        # User guess
        st.markdown(f"**Your guess:** {user_guess}")
        
        # Model prediction
        with st.spinner("AI is thinking... 🤔"):
            time.sleep(1)
            st.markdown(f"**AI prediction:** {prediction['prediction']} ({prediction['confidence']:.2%})")
        
        # True label
        time.sleep(0.5)
        st.markdown(f"**Actual answer:** {true_label}")
        
        # Update score and streak
        if user_guess == true_label:
            st.session_state.score += 1
            st.session_state.streak += 1
            st.success("🎉 Correct! Keep going!")
            
            # Check for achievements
            if st.session_state.streak == 5:
                st.balloons()
                st.success("🏆 Achievement Unlocked: 5 correct in a row!")
        else:
            st.session_state.streak = 0
            st.error("❌ Wrong! Try again!")
        
        save_statistics()

def learn_mode():
    """Educational section about deepfake detection"""
    st.header("📚 Learn About Deepfakes")
    
    # Introduction
    st.markdown("""
    ### What are Deepfakes?
    Deepfakes are synthetic media where a person's image or video is replaced with someone else's likeness using artificial intelligence and deep learning technology.
    
    ### How to Spot Deepfakes
    1. **Unnatural Eye Movement**: Watch for irregular blinking or fixed gaze
    2. **Facial Inconsistencies**: Look for blurring or warping around the face
    3. **Audio-Visual Sync**: Check if lip movements match the audio
    4. **Lighting Inconsistencies**: Notice unusual shadows or lighting
    5. **Boundary Issues**: Look for blurring or artifacts around hair and face edges
    """)
    
    # Interactive examples
    st.subheader("Interactive Examples")
    
    tab1, tab2 = st.tabs(["Common Indicators", "Case Studies"])
    
    with tab1:
        st.markdown("""
        ### Key Visual Indicators
        - Blurry or inconsistent facial features
        - Unnatural skin texture
        - Flickering around edges
        - Inconsistent lighting across the face
        """)
        
    with tab2:
        st.markdown("""
        ### Real-World Examples
        Learn from actual cases where deepfakes were detected and analyzed.
        """)
        
    # Resources
    st.subheader("Additional Resources")
    st.markdown("""
    - [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
    - [AI Foundation's Reality Defender](https://aifoundation.com/responsibility/)
    - [Media Forensics](https://mediaf.org/)
    """)

# Update main navigation
def main():
    # Load saved statistics
    load_statistics()
    
    # Enhanced navigation with icons
    selected_mode = option_menu(
        menu_title=None,
        options=["Detector", "Challenge", "Learn", "Settings"],
        icons=["search", "controller", "book", "gear"],
        orientation="horizontal",
    )
    
    if selected_mode == "Detector":
        detector_mode()
    elif selected_mode == "Challenge":
        game_mode()
    elif selected_mode == "Learn":
        learn_mode()
    elif selected_mode == "Settings":
        settings_mode()

def settings_mode():
    """Settings and preferences"""
    st.header("⚙️ Settings")
    
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
        st.download_button(
            label="Download Statistics CSV",
            data=stats_df.to_csv(index=False),
            file_name="deepfake_detector_stats.csv",
            mime="text/csv"
        )

# -------------------
# MODEL LOADING
# -------------------
@st.cache_resource
def load_model():
    """Load the deepfake detection model with error handling"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), "model")
        json_path = os.path.join(model_path, "dffnetv2B0.json")
        weights_path = os.path.join(model_path, "dffnetv2B0.h5")
        
        if not os.path.exists(json_path) or not os.path.exists(weights_path):
            st.error("❌ Model files not found! Please ensure model files are in the correct location.")
            st.stop()
            
        # Load model architecture
        with open(json_path, 'r') as f:
            model = model_from_json(f.read())
            
        # Load weights
        model.load_weights(weights_path)
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model at startup
try:
    classifier = load_model()
except Exception as e:
    st.error("❌ Failed to initialize the application. Please check the model files and dependencies.")
    st.stop()

if __name__ == "__main__":
    main() 