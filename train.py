# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
import operator
import numpy as np

categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=32, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model

# Code copied from - https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='rgb',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='rgb',
                                            class_mode='categorical') 
classifier.fit_generator(
        training_set,
        steps_per_epoch=1757, # No of images in training set
        epochs=10,
        validation_data=test_set,
        validation_steps=478)# No of images in test set


# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')

image_predict = image.load_img('data/test/3/hand4_3_bot_seg_5_cropped.jpeg',target_size=(64, 64))

image_predict = image.img_to_array(image_predict)
image_predict = np.expand_dims(image_predict, axis=0)

result = classifier.predict(image_predict)
prediction = {    'ZERO': result[0][0],
                  'ONE': result[0][1],
                  'TWO': result[0][2],
                  'THREE': result[0][3],
                  'FOUR': result[0][4],
                  'FIVE': result[0][5],
                  'a': result[0][6],
                  'b': result[0][7],
                  'c': result[0][8],
                  'd': result[0][9],
                  'e': result[0][10],
                  'f': result[0][11],
                  'g': result[0][12],
                  'h': result[0][13],
                  'i': result[0][14],
                  'j': result[0][15],
                  'k': result[0][16],
                  'l': result[0][17],
                  'm': result[0][18],
                  'n': result[0][19],
                  'o': result[0][20],
                  'p': result[0][21],
                  'q': result[0][22],
                  'r': result[0][23],
                  's': result[0][24],
                  't': result[0][25],
                  'u': result[0][26],
                  'v': result[0][27],
                  'w': result[0][28],
                  'x': result[0][29],
                  'y': result[0][30],
                  'z': result[0][31],
              }
# Sorting based on top prediction
prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
print(prediction)

