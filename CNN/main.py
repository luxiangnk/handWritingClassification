from mnist import MNIST
import random
import numpy as np
import cv2
import tensorflow.python.keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten, Dropout
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

mndata = MNIST('../data')

#load data
imagesTrain_, labelsTrain_ = mndata.load_training()
imagesTest_, labelsTest_ = mndata.load_testing()

imagesTrain, labelsTrain = np.array(imagesTrain_), np.array(labelsTrain_)
imagesTest, labelsTest = np.array(imagesTest_), np.array(labelsTest_)

print("Train data size:", imagesTrain.shape,imagesTrain.shape[0])
print("Train label size:", labelsTrain.shape,labelsTrain.shape[0])

# reshape the data 
imagesTrain = imagesTrain.reshape(imagesTrain.shape[0],28,28,1)
imagesTest = imagesTest.reshape(imagesTest.shape[0],28,28,1)
labelsTrain = labelsTrain.reshape(labelsTrain.shape[0],1)
labelsTest = labelsTest.reshape(labelsTest.shape[0],1)

print("Data size after reshape:",imagesTrain.shape,imagesTrain.shape[0])

# show one image randomly
index = random.randrange(0, len(imagesTrain)) 
img = imagesTrain[index]
img = img/255

cv2.imshow("Image",img)
print(labelsTrain_[index])

# Preprocess the images
def preprocess(img):
    img = img/255
    return img

imagesTrain = np.array(list(map(preprocess, imagesTrain)))
imagesTest = np.array(list(map(preprocess, imagesTest)))

labelsTrain = to_categorical(labelsTrain, 10)

cv2.imshow("Image2",imagesTrain[index])
#cv2.waitKey(0)

# define model
def model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(Adam(lr = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save the model
model = model()
print(model.summary())
'''
history = model.fit(imagesTrain, labelsTrain_, batch_size=400,
                            epochs=10,
                            validation_data=(xVals, yVals), shuffle = 1, verbose = 1)
'''
history = model.fit(imagesTrain, labelsTrain, batch_size=400,
                            epochs=10, shuffle = 1, verbose = 1)


mp = "../output/model.h5"
model.save(mp)


plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
 
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.legend(['training','test'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()