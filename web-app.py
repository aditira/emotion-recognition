from flask import Flask, request, render_template, jsonify
import os
import io
from PIL import Image
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.environ['OPENAI_API_KEY']

app = Flask(__name__)

model = load_model('model/fine-tune/emotion_model_kaggle.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.form['image']
    emotion_label = request.form['emotion']

    save_image_to_dataset(image_data, emotion_label)
    return 'Upload berhasil'

@app.route('/finetune', methods=['POST'])
def fine_tune():
    image_data = request.form['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    
    image = Image.open(io.BytesIO(image_data))
    emotion_label = request.form['emotion']

    image = image.convert('L')
    image = image.resize((224, 224))

    image_path = f'dataset/fine-tune/{emotion_label}/image.jpg'
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image.save(image_path)

    fine_tune_emotion = fine_tune()
    return fine_tune_emotion

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.form['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    
    image = Image.open(io.BytesIO(image_data))

    image = image.convert('L')
    image = image.resize((224, 224))

    image.save('dataset/predict/image.jpg')
    predicted_emotion = predict_emotion()
    return predicted_emotion

@app.route('/respond', methods=['POST'])
def respond():
    emotion = request.form['emotion']

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "The person seems to be " + emotion}
        ]
    )

    output = response['choices'][0]['message']['content']
    return jsonify({'response': output})

def predict_emotion():
    img_path = 'dataset/predict/image.jpg'
    img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    test_data = img_array / 255.0
    preds = model.predict(test_data)

    emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']
    predicted_emotion = emotion_labels[np.argmax(preds)]

    return predicted_emotion

def fine_tune():
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

    train_data = datagen.flow_from_directory('dataset/fine-tune', 
                                             target_size=(48, 48), 
                                             batch_size=1, 
                                             color_mode='grayscale', 
                                             class_mode='categorical')


    model.fit(train_data, epochs=5)
    model.save('model/fine-tune/emotion_model_kaggle.h5')

    return "Fine tuning completed!"
     
def save_image_to_dataset(image_data, emotion_label):
    image_data = base64.b64decode(image_data.split(',')[1])

    dataset_folder = 'dataset/web'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    emotion_folder = os.path.join(dataset_folder, emotion_label)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)
    image_count = len(os.listdir(emotion_folder))
    image_path = os.path.join(emotion_folder, f'image-{image_count+1}.jpg')

    with open(image_path, 'wb') as file:
        file.write(image_data)
    
    print(f'Gambar berhasil disimpan: {image_path}')

if __name__ == '__main__':
    app.run()
