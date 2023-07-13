from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from IPython.display import Image, display
import numpy as np

num_classes = 8
my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=None))
my_new_model.add(Dense(num_classes, activation='softmax'))
my_new_model.layers[0].trainable = False

my_new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory(
        'dataset/kaggle',
        target_size=(image_size, image_size),
        batch_size=24,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        'dataset/kaggle',
        target_size=(image_size, image_size),
        class_mode='categorical')

my_new_model.fit_generator(
        train_generator,
        steps_per_epoch=3,
        validation_data=validation_generator,
        validation_steps=1)