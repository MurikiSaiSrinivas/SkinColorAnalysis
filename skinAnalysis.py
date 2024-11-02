from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.cluster import KMeans
import requests 


app = Flask(__name__)

# Define constants for landmark indices
LIP_INDICES = [61, 91, 181, 84, 17, 314, 405, 321, 375, 409, 270, 269, 267, 0, 37, 39, 40]
LEFT_EYE_INDICES = [469, 470, 471, 472]
RIGHT_EYE_INDICES = [374, 474, 475, 476]
CHEEK_INDICES = [234, 454, 361, 132]


# Step 1: Create the FaceLandmarker detector
def create_face_landmarker():
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)

# Step 4: Extract pixels inside a polygon defined by landmarks
def get_pixels_in_polygon(image, landmarks):
    height, width, _ = image.shape
    points = np.array([(int(lm.x * width), int(lm.y * height)) for lm in landmarks], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    return image[mask == 255]

# Step 5: Find the dominant color using KMeans clustering
def find_dominant_color_kmeans(pixels, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    dominant_cluster = np.bincount(kmeans.labels_).argmax()
    return kmeans.cluster_centers_[dominant_cluster].astype(int)

# Convert RGB to a single integer color representation
def rgb_to_int(rgb):
    return (rgb[2] << 16) + (rgb[1] << 8) + rgb[0]

# Step 6: Extract the dominant color for a given region
def extract_dominant_color(image, landmarks, indices):
    if landmarks:
        region_landmarks = [landmarks[0][i] for i in indices]
        pixels = get_pixels_in_polygon(image, region_landmarks)
        dominant_color = find_dominant_color_kmeans(pixels)
        return rgb_to_int(dominant_color)
        # return find_dominant_color_kmeans(pixels)
    print("No face detected.")
    return None

def download_model():
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    model_path = "face_landmarker_v2_with_blendshapes.task"

    # Check if the model file already exists
    if not os.path.isfile(model_path):
        print("Model file not found. Downloading...")
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully.")
        else:
            print("Failed to download model. Status code:", response.status_code)

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_uri = data.get('imageUri')

    if image_uri is None:
        return jsonify({'error': 'No image URI provided'}), 400

    response = requests.get(image_uri)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download image'}), 500

    image_path = 'temp_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(response.content)

    download_model()
    detector = create_face_landmarker()

    image = cv2.imread(image_path)
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)

    landmarks = detection_result.face_landmarks

    dominant_colors = {
        'cheek': extract_dominant_color(image, landmarks, CHEEK_INDICES),
        'lip': extract_dominant_color(image, landmarks, LIP_INDICES),
        'left_eye': extract_dominant_color(image, landmarks, LEFT_EYE_INDICES),
        'right_eye': extract_dominant_color(image, landmarks, RIGHT_EYE_INDICES)
    }

    os.remove(image_path)

    return jsonify({
        'dominant_cheek_color': dominant_colors['cheek'].tolist() if dominant_colors['cheek'] is not None else None,
        'dominant_lip_color': dominant_colors['lip'].tolist() if dominant_colors['lip'] is not None else None,
        'dominant_left_eye_color': dominant_colors['left_eye'].tolist() if dominant_colors['left_eye'] is not None else None,
        'dominant_right_eye_color': dominant_colors['right_eye'].tolist() if dominant_colors['right_eye'] is not None else None,
    })

if __name__ == '__main__':
    app.run(debug=True)