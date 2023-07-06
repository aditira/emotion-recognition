from flask import Flask, request, render_template
import base64
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load model yang telah dilatih
model_path = 'model'  # Ganti dengan path model yang sesuai
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.form['image']
    emotion_label = request.form['emotion']
    
    # Menyimpan gambar ke dataset
    save_image_to_dataset(image_data, emotion_label)
    
    return 'Upload berhasil'

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    
    # Melakukan prediksi dengan model
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
    
    labels = ['angry', 'happy', 'neutral', 'sad', 'stress', 'surprised'] # Ganti dengan label yang sesuai dengan model Anda
    predicted_emotion = labels[predicted_label]
    
    return f'Emosi yang diprediksi: {predicted_emotion}'

def save_image_to_dataset(image_data, emotion_label):
    # Mendekode data gambar dari base64
    image_data = base64.b64decode(image_data.split(',')[1])
    
    # Membuat folder dataset jika belum ada
    dataset_folder = 'dataset/train'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Menentukan nama file untuk gambar baru
    emotion_folder = os.path.join(dataset_folder, emotion_label)
    if not os.path.exists(emotion_folder):
        os.makedirs(emotion_folder)
    image_count = len(os.listdir(emotion_folder))
    image_path = os.path.join(emotion_folder, f'{image_count}.jpg')

    # Menyimpan gambar ke file
    with open(image_path, 'wb') as file:
        file.write(image_data)
    
    print(f'Gambar berhasil disimpan: {image_path}')

if __name__ == '__main__':
    app.run()
