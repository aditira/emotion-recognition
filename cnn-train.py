from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Membuat arsitektur model CNN yang lebih kompleks
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu')) # tambahan layer konvolusional
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu')) # tambahan Dense layer dan neuron lebih banyak
model.add(Dropout(0.5))

dataset_path = 'dataset/kaggle'
# Mendapatkan list dari semua sub-direktori
sub_dirs = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
num_class = len(sub_dirs)
print(f"Number of classes: {num_class}")
model.add(Dense(num_class, activation='softmax')) # jumlah kelas emosi yang akan di prediksi

# Mengkompilasi model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Membuat generator gambar dengan augmentasi data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255) # test data tetap hanya di rescale

training_set = train_datagen.flow_from_directory(dataset_path, 
                                                 target_size=(48, 48), 
                                                 batch_size=32, 
                                                 color_mode='grayscale', 
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(dataset_path, 
                                            target_size=(48, 48), 
                                            batch_size=32, 
                                            color_mode='grayscale', 
                                            class_mode='categorical')

# Mengatur early stopping
earlystop = EarlyStopping(patience=5) # Jumlah epoch dengan tidak ada peningkatan setelah mana pelatihan akan dihentikan.

# Mengatur model checkpoint
checkpoint = ModelCheckpoint('model/cnn/best_model.h5',  # model filename
                             monitor='val_loss', # property to monitor
                             verbose=1, # verbosity - 0 or 1
                             save_best_only=True, # The latest best model will not be overwritten
                             mode='auto') # The decision to overwrite model is made 
                                          # automatically depending on the quantity to monitor 

# Membuat list dari kedua callbacks
callbacks_list = [earlystop, checkpoint]

# Melakukan training model dengan data yang sudah di augmentasi
model.fit(training_set, 
          steps_per_epoch=training_set.samples//32, 
          validation_data=test_set, 
          validation_steps=test_set.samples//32, 
          epochs=50, # Bisa menambahkan lebih banyak epoch karena kita menggunakan callback
          callbacks=callbacks_list)