# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import operator
import numpy as np


# layers : Convolution Layer, max pooling layer, convolution, max pooling, 2 Dense layers
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer
converted_image_shape=(64,64,3)
# 32 features in the O/P layer
filter = 32
# 3*3 kernel or filter matrix to calculate O/P features
kernel_size=(3,3)
# relu activation as it produces the best output
activation_function='relu'
# padding = 'same' to make sure that the output image is same as the input image
conv_layer = Convolution2D(filter, kernel_size, input_shape=converted_image_shape, activation=activation_function, padding='same')
classifier.add(conv_layer)

# max pooling layer
# max pool layer to descrease the size of the matrix of 32*32 to 16*16
# (2,2) means it finds the maximum value in each of the 2*2 section
pool_size=(2,2)
max_pool_layer = MaxPooling2D(pool_size=pool_size)
classifier.add(max_pool_layer)

# Second convolution layer
conv_layer_2 = Convolution2D(filter, kernel_size, activation=activation_function)
classifier.add(conv_layer_2)

# max pooling layer
# input_shape is going to be the pooled feature maps from the previous convolution layer
max_pool_layer_2 = MaxPooling2D(pool_size=pool_size)
classifier.add(max_pool_layer_2)

# Flattening the layers - converts the matrix to 1d array
classifier.add(Flatten())

# First dense layer to create the actual prediction network
# Adding a fully connected layer
# higher number of units = higher accuracy = higher
dense1_units = 512
dense1_activation = 'relu'

dense1_layer = Dense(units=dense1_units, activation=dense1_activation)
classifier.add(dense1_layer)

# Dropout layer - to ignore some neurons to improve reliability
dropout_perc = 0.5 # dropping half of the neurons
dropout_layer = Dropout(rate=dropout_perc)

# final dense layer to produce the o/p for 32 classes
# softmax because we are calculating probabilities for all of the cateogories
# 2nd dense layer
# 32 because we have 32 classes
dense2_units = 32
dense2_activation = 'softmax'
dense2_layer = Dense(units=dense2_units, activation=dense2_activation)
classifier.add(dense2_layer) # softmax for more than 2




# Compiling the CNN
optimizer_used = 'adam'
loss_measure = 'categorical_crossentropy'
metrics_values = ['accuracy']
classifier.compile(optimizer=optimizer_used, loss=loss_measure, metrics=metrics_values) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=10,
                                                 color_mode='rgb',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=10,
                                            color_mode='rgb',
                                            class_mode='categorical') 
history = classifier.fit_generator(
        training_set,
        steps_per_epoch=1757, # No of images in training set
        epochs=20,
        validation_data=test_set,
        validation_steps=478)# No of images in test set

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

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

