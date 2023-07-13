from flask import Flask, request, render_template
import base64
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# # Load the trained model
model = load_model('model/master/emotion_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.form['image']
    emotion_label = request.form['emotion']
    
    # Save an image to a dataset
    save_image_to_dataset(image_data, emotion_label)
    
    return 'Upload berhasil'

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    with open('dataset/predict/image.jpg', 'wb') as file:
        file.write(image_data)

    # Make predictions with models
    predict_emotion()

def predict_emotion():
    img_path = 'dataset/predict/image.jpg'
    # Load image
    img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')

    # Convert image to array and expand dimension
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Standardize (Rescale pixel values)
    test_data = img_array / 255.0

    # Make prediction
    preds = model.predict(test_data)
    print(preds[0])
     
def save_image_to_dataset(image_data, emotion_label):
    # Decode image data from base64
    image_data = base64.b64decode(image_data.split(',')[1])
    
    # Create dataset folder if not exist 
    dataset_folder = 'dataset/web'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Specify a file name for a new image
    emotion_folder = os.path.join(dataset_folder, emotion_label)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)
    image_count = len(os.listdir(emotion_folder))
    image_path = os.path.join(emotion_folder, f'image-{image_count+1}.jpg')

    # Save an image to a file
    with open(image_path, 'wb') as file:
        file.write(image_data)
    
    print(f'Gambar berhasil disimpan: {image_path}')

if __name__ == '__main__':
    app.run()
