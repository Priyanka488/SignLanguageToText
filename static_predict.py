from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import operator

# Loading the model
json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from memory")

image_predict = image.load_img('data/test/w/hand5_w_dif_seg_5_cropped.jpeg',target_size=(64, 64))

image_predict = image.img_to_array(image_predict)
image_predict = np.expand_dims(image_predict, axis=0)

result = loaded_model.predict(image_predict)
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
