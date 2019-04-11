from mnist import MNIST
import random
import numpy as np
import cv2
import tensorflow.python.keras
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import cv2

mndata = MNIST('../data')

#load data
imagesTest_, labelsTest_ = mndata.load_testing()

imagesTest, labelsTest = np.array(imagesTest_), np.array(labelsTest_)

# reshape the data 
imagesTest = imagesTest.reshape(imagesTest.shape[0],784)
labelsTest = labelsTest.reshape(labelsTest.shape[0],1)


# show one image randomly
def preprocess(img):
    img = img/255
    return img

imagesTest = np.array(list(map(preprocess, imagesTest)))

#labelsTrain = to_categorical(labelsTest, 10)
index = random.randrange(0, len(imagesTest)) 
#cv2.imshow("Image",imagesTest[index])
#cv2.waitKey(0)

print(imagesTest[index].shape)

mp = "../output/model_MLP.h5"
model = load_model(mp)

print("predicted sign: "+ str(model.predict_classes(imagesTest[index].reshape(1,784))))
print("actual sign: ",labelsTest[index])