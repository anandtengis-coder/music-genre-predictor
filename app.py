import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------
# 1. Page Configuration & Styling
# ----------------------------------------------------
st.set_page_config(
    page_title="Music Genre Predictor",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, clean look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DB954; /* Spotify Green */
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1aa34a;
        color: white;
    }
    .prediction-holder {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Load the Pre-trained KNN Model (Using Joblib)
# ----------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # joblib natively opens the file path directly
        model = joblib.load('genre_knn.pkl')
        return model
    except FileNotFoundError:
        st.error("Error: 'genre_knn.pkl' not found. Please place the model file in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("If you are still seeing a STACK_GLOBAL/Unpickling error, ensure the model was actually saved using joblib.dump() in Python 3.14, or downgrade your Python version to 3.11/3.12.")
        return None

model = load_model()

# ----------------------------------------------------
# 3. UI Layout
# ----------------------------------------------------
st.title("🎵 Music Genre Predictor")
st.markdown("Adjust the musical attributes below to find out the most likely genre.")
st.write("---")

# Feature Inputs
st.subheader("🎛️ Audio Features")

# Creating layout columns for sliders to look organized
col1, col2 = st.columns(2)

with col1:
    tempo = st.slider("Tempo (BPM)", min_value=50.0, max_value=220.0, value=120.0, step=1.0, help="Speed of the track")
    energy = st.slider("Energy", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="Perceptual measure of intensity and activity")

with col2:
    danceability = st.slider("Danceability", min_value=0.0, max_value=1.0, value=0.5, step=0.01, help="How suitable a track is for dancing")
    acousticness = st.slider("Acousticness", min_value=0.0, max_value=1.0, value=0.2, step=0.01, help="Confidence measure of whether the track is acoustic")

st.write("---")

# ----------------------------------------------------
# 4. Prediction Logic
# ----------------------------------------------------
if st.button("Predict Genre") and model is not None:
    # Prepare input data matching the order expected by the model
    features = np.array([[tempo, energy, danceability, acousticness]])
    
    # Perform prediction
    prediction = model.predict(features)[0]
    
    # Get probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = model.classes_
    else:
        probabilities = None

    # Display Result
    st.markdown(f"""
        <div class="prediction-holder">
            <h3 style='margin:0; color:#555;'>Predicted Genre</h3>
            <h1 style='margin:10px 0; color:#1DB954; font-size: 3rem;'>{str(prediction).upper()}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.write(" ")
    
    # Display Probabilities if available
    if probabilities is not None:
        st.subheader("📊 Probability Breakdown")
        
        # Create a clean dataframe for the chart
        prob_df = pd.DataFrame({
            'Genre': classes,
            'Probability': probabilities
        }).sort_values(by='Probability', ascending=True)
        
        # Streamlit Native Horizontal Bar Chart
        st.bar_chart(
            data=prob_df,
            x='Probability',
            y='Genre',
            color='#1DB954',
            horizontal=True
        )