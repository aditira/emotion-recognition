import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

# Definisikan kelas dataset untuk data latih
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Ambil data dari dataset yang telah dikumpulkan
# Gantilah dengan path dan nama file sesuai dengan dataset Anda
dataset_folder = 'dataset/train'
labels = ['angry', 'happy', 'neutral', 'sad', 'stress', 'surprised']  # Ganti dengan label yang sesuai dengan dataset Anda
texts = []
all_labels = []

for label_idx, label in enumerate(labels):
    label_folder = os.path.join(dataset_folder, label)
    for file_name in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file_name)
        with open(file_path, 'rb') as file:  # Baca file sebagai file biner
            text = file.read().decode('utf-8', errors='ignore')  # Ubah file biner menjadi teks dengan encoding UTF-8
            texts.append(text)
            all_labels.append(label_idx)

# Split dataset menjadi data latih dan data validasi
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, all_labels, test_size=0.2, random_state=42
)

# Inisialisasi tokenizer dan model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(labels))

# Buat dataset dan dataloader untuk data latih dan validasi
train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, max_length=512)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, max_length=512)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Tentukan perangkat keras yang akan digunakan (misalnya, 'cuda' untuk GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pindahkan model ke perangkat keras yang sesuai
model.to(device)

# Definisikan optimizer dan jumlah epoch
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3

# Loop melalui epoch
for epoch in range(num_epochs):
    # Mode pelatihan
    model.train()

    # Loop melalui dataloader latih
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Reset gradien
        model.zero_grad()

        # Hitung output model
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Hitung loss dan lakukan backpropagation
        loss = outputs.loss
        loss.backward()

        # Update parameter dengan optimizer
        optimizer.step()

    # Mode evaluasi
    model.eval()

    # Inisialisasi variabel untuk menghitung akurasi validasi
    correct_predictions = 0
    total_predictions = 0

    # Loop melalui dataloader validasi
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Tidak perlu perhitungan gradien saat evaluasi
        with torch.no_grad():
            # Hitung output model
            outputs = model(input_ids, attention_mask=attention_mask)

        # Ambil prediksi dari output
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

        # Hitung jumlah prediksi yang benar
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    # Hitung akurasi validasi
    accuracy = correct_predictions / total_predictions
    print(f'Epoch {epoch + 1} - Validation Accuracy: {accuracy}')

# Simpan model
model_folder = 'model'
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

model.save_pretrained(model_folder)
tokenizer.save_pretrained(model_folder)