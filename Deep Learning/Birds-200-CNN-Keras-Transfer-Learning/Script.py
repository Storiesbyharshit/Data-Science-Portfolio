import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename));
from keras.preprocessing.image import ImageDataGenerator
train_batch = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/kaggle/input/100-bird-species/train',
                                                 target_size = (224, 224),
test_set = test_datagen.flow_from_directory('/kaggle/input/100-bird-species/test',
                                            target_size = (224,224),
                                            batch_size = train_batch,
                                            class_mode = 'categorical')
validation_generator = train_datagen.flow_from_directory('/kaggle/input/100-bird-species/valid',target_size=(224, 224),class_mode='categorical'),batch_size = train_batch,class_mode = 'categorical')
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras import applications
import tensorflow as tf
from keras.models import Model

base_model = applications.ResNet50V2(weights='imagenet', include_top=False, input_shape= (224,224,3))
add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(512, activation='relu'))
add_model.add(Dense(200, activation='softmax'))
model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

epoch = 10
train_steps_per_epoch = 1000
history= model.fit_generator( training_set, steps_per_epoch=train_steps_per_epoch, epochs=epoch, validation_data=validation_generator) 
import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()