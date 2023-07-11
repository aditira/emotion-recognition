from flask import Flask, request, render_template
import base64
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load the trained model
model_path = 'model'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)

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
    text = request.form['text']
    
    # Make predictions with models
    input_ids = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )['input_ids'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = torch.argmax(outputs.logits, dim=1)
        predicted_label = predictions.item()
    
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    predicted_emotion = labels[predicted_label]
    
    return f'Emosi yang diprediksi: {predicted_emotion}'

def save_image_to_dataset(image_data, emotion_label):
    # Decode image data from base64
    image_data = base64.b64decode(image_data.split(',')[1])
    
    # Create dataset folder if not exist 
    dataset_folder = 'dataset/train'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Specify a file name for a new image
    emotion_folder = os.path.join(dataset_folder, emotion_label)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)
    image_count = len(os.listdir(emotion_folder))
    image_path = os.path.join(emotion_folder, f'{image_count}.jpg')

    # Save an image to a file
    with open(image_path, 'wb') as file:
        file.write(image_data)
    
    print(f'Gambar berhasil disimpan: {image_path}')

if __name__ == '__main__':
    app.run()
