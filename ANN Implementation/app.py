import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# PAGE CONFIG
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)


# GLOBAL CSS (NO WHITE BAR, DARK/LIGHT SAFE)
st.markdown("""
<style>
header[data-testid="stHeader"] {display: none;}
footer {visibility: hidden;}

div.block-container {
    padding-top: 1.5rem;
}

.main-card {
    background: rgba(20, 20, 20, 0.85);
    padding: 2.2rem;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #eaeaea;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #b0b0b0;
    margin-bottom: 25px;
}

.prediction {
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    color: #00e676;
    margin-top: 20px;
}

.confidence {
    text-align: center;
    font-size: 16px;
    color: #90caf9;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL & SCALER
@st.cache_resource
def load_artifacts():
    model = load_model("iris_model.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load model or scaler")
    st.exception(e)
    st.stop()


# UI CARD
st.markdown("<div class='main-card'>", unsafe_allow_html=True)

st.markdown(
    "<div class='title'>🌸 Iris Flower Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Artificial Neural Network based classification</div>",
    unsafe_allow_html=True
)


# INPUT FEATURES
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)


# PREDICTION
if st.button("Predict Species"):
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    classes = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]

    st.markdown(
        f"<div class='prediction'>Predicted Species: {classes[predicted_class]}</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div class='confidence'>Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)


# FOOTER
st.markdown(
    "<p style='text-align:center; color:#9e9e9e; font-size:13px;'>"
    "ANN based Iris Flower Classification | Streamlit"
    "</p>",
    unsafe_allow_html=True
)
