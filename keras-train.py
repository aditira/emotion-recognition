from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

dataset_path = 'dataset/web'
sub_dirs = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
num_class = len(sub_dirs)

model.add(Dense(num_class, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

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

earlystop = EarlyStopping(patience=5)

checkpoint = ModelCheckpoint('model/keras/best_model.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

callbacks_list = [earlystop, checkpoint]

model.fit(training_set, 
          steps_per_epoch=training_set.samples//32, 
          validation_data=test_set, 
          validation_steps=test_set.samples//32, 
          epochs=50,
          callbacks=callbacks_list)