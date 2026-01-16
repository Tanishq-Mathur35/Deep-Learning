import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST CNN Digit Predictor", layout="wide")

st.title("✍️ MNIST Digit Recognition (CNN)")
st.write("Draw a digit (0-9) on the left and see the analysis on the right.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_model()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Drawing Canvas")
    canvas_size = 280
    canvas = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        width=canvas_size,
        height=canvas_size,
        drawing_mode="freedraw",
        key="canvas",
    )
    predict_button = st.button("🔮 Predict Digit")

def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert("L")
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

if predict_button:
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0]
        img = Image.fromarray(img.astype("uint8"))
        processed_img = preprocess_image(img)
        
        prediction = model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)

        with left_col:
            st.success(f"### ✅ Predicted Digit: **{predicted_digit}**")
            st.info(f"Confidence: **{confidence:.2%}**")

        with right_col:
            st.subheader("📊 Prediction Probabilities")
            st.bar_chart(prediction[0])
            
    else:
        st.warning("Please draw a digit first!")

st.markdown("---")
st.caption("Built with ❤️ using CNN + Streamlit")
