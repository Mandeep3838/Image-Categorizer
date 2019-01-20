# -*- coding: utf-8 -*-


# Importing the Keras Packages
from keras.models import Sequential
from keras.layers import Convolution2D as Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Dense
    
model = Sequential()

model.add(Conv2D(filters=4, kernel_size=2, padding='same',
                 activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv2D(filters=12, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))


# Step 4 - Full Connention

model.add(Dense(units = 6 , activation = "softmax"))
model.summary()
# Compiling the CNN
model.compile(optimizer = "rmsprop" , loss = "categorical_crossentropy" , metrics = ["accuracy"])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                            rescale=1./255,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                'training',
                                                target_size=(100, 100),
                                                color_mode='grayscale',
                                                batch_size=32, # Taking set of 32 images for training
                                                class_mode='sparse')
                                        
test_set = test_datagen.flow_from_directory(
                                            'test',
                                            target_size=(100, 100),
                                            batch_size = 32,
                                            color_mode='grayscale',
                                            class_mode='sparse')

model.fit_generator(
                        training_set,
                        steps_per_epoch=4632,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=368)




