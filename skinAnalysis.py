from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import requests 


app = Flask(__name__)

# Define constants for landmark indices
LIP_INDICES = [61, 91, 181, 84, 17, 314, 405, 321, 375, 409, 270, 269, 267, 0, 37, 39, 40]
LEFT_EYE_INDICES = [469, 470, 471, 472]
RIGHT_EYE_INDICES = [374, 474, 475, 476]
CHEEK_INDICES = [234, 454, 361, 132]


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

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_uri = data.get('imageUri')
    landmarks = data.get('landmarks')

    if image_uri is None:
        return jsonify({'error': 'No image URI provided'}), 400

    response = requests.get(image_uri)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to download image'}), 500

    image_path = 'temp_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(response.content)

    image = cv2.imread(image_path)

    dominant_colors = {
        'cheek': extract_dominant_color(image, landmarks, CHEEK_INDICES),
        'lip': extract_dominant_color(image, landmarks, LIP_INDICES),
        'left_eye': extract_dominant_color(image, landmarks, LEFT_EYE_INDICES),
        'right_eye': extract_dominant_color(image, landmarks, RIGHT_EYE_INDICES)
    }

    os.remove(image_path)

    return jsonify({
        'cheek_color': dominant_colors['cheek'].tolist() if dominant_colors['cheek'] is not None else None,
        'lip_color': dominant_colors['lip'].tolist() if dominant_colors['lip'] is not None else None,
        'left_eye_color': dominant_colors['left_eye'].tolist() if dominant_colors['left_eye'] is not None else None,
        'right_eye_color': dominant_colors['right_eye'].tolist() if dominant_colors['right_eye'] is not None else None,
    })

if __name__ == '__main__':
    app.run(debug=True)