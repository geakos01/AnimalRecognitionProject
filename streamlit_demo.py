import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
import mapper as m


# Load the model only once using st.cache
@st.cache_resource  # (allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("animal_classifier_model")
    return model


# Define a function to load and process the image
def load_image_from_path(path):
    img = cv2.imread(path)
    return img


def load_image(file):
    # Read the file object as a np.uint8 array
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
    return img


def process_image(img):
    resized_img = cv2.resize(img, (224, 224))  # This is needed to match the model input shape
    processed_img = preprocess_input(resized_img)
    processed_img = processed_img[np.newaxis, ...]
    return processed_img


# Define a function to make a prediction
def make_prediction(model, image):
    probs = model.predict(image)
    predicted_class = np.argmax(probs)
    return predicted_class


# Create a plus function that writes the probability of each class
def show_prediction_probs(model, image):
    # Call the original function to make a prediction
    predicted_class = make_prediction(model, image)
    # Get the probability array for each class
    probs = model.predict(image)
    # Get the probability for the predicted class
    prob = probs[0][predicted_class]
    # Print the predicted class and probability
    st.write(f"Predicted Class: {m.REVERSE_ANIMAL_DICT[predicted_class]}")
    st.write(f"Predicted probability for {m.REVERSE_ANIMAL_DICT[predicted_class]}: {prob * 100:.1f}%")

    return predicted_class, prob


# Add a title and a header to the app
st.title("Animal Classification")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload an animal image for image classification", type=["jpg", "jpeg", "png"])

# If a file is uploaded, display the image and make a prediction
if uploaded_file is not None:
    image = load_image(uploaded_file)
    processed_image = process_image(image)

    model = load_model()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_class, prob = show_prediction_probs(model, processed_image)
