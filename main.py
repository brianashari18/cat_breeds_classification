import cv2
import streamlit as st
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern

model_data = joblib.load('cat_breed_classifier_v2.pkl')

ensemble = model_data['model']
scaler = model_data['scaler']
selector = model_data['selector']
le = model_data['label_encoder']
selected_breeds = model_data['classes']

st.title("Cat Breed Classifier")
st.write("Upload an image of a cat, and I will predict its breed.")

def extract_features(img_array):
    img = cv2.resize(img_array, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray, orientations=12, pixels_per_cell=(32, 32),
                       cells_per_block=(3, 3), feature_vector=True)
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    color_features = []
    for i in range(3):
        channel = lab[:, :, i]
        color_features.extend([
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            np.min(channel),
            np.max(channel)
        ])
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    texture_features = [
        np.mean(sobelx),
        np.std(sobelx),
        np.mean(sobely),
        np.std(sobely)
    ]
    return np.concatenate([hog_features, lbp_hist, color_features, texture_features])

def manual_augmentation(img):
    augmented = [cv2.flip(img, 1)]
    rows, cols = img.shape[:2]
    for angle in [10, -10, 15, -15]:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        augmented.append(cv2.warpAffine(img, M, (cols, rows)))
    for beta in [30, -30]:
        augmented.append(cv2.convertScaleAbs(img, beta=beta))
    return augmented

uploaded_file = st.file_uploader("Choose a cat image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    X = []
    features = extract_features(img_array)
    X.append(features)

    X_scaled = scaler.transform(X)

    X_selected = selector.transform(X_scaled)

    X_final = X_selected

    prediction = ensemble.predict(X_final)

    predicted_breed = le.inverse_transform(prediction)

    st.write(f"Predicted Breed: {predicted_breed[0]}")
