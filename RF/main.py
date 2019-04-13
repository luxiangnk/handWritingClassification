from mnist import MNIST
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mndata = MNIST('../data')

#load data
imagesTrain_, labelsTrain_ = mndata.load_training()
imagesTest_, labelsTest_ = mndata.load_testing()

imagesTrain, labelsTrain = np.array(imagesTrain_), np.array(labelsTrain_)
imagesTest, labelsTest = np.array(imagesTest_), np.array(labelsTest_)

print("Train data size:", imagesTrain.shape,imagesTrain.shape[0])
print("Train label size:", labelsTrain.shape,labelsTrain.shape[0])

# reshape the data 
imagesTrain = imagesTrain.reshape(imagesTrain.shape[0],784)
imagesTest = imagesTest.reshape(imagesTest.shape[0],784)
labelsTrain = labelsTrain.reshape(labelsTrain.shape[0])
labelsTest = labelsTest.reshape(labelsTest.shape[0])



# show one image randomly
index = random.randrange(0, len(imagesTrain)) 
img = imagesTrain[index]
img = img/255


# Preprocess the images
def preprocess(img):
    img = img/255
    return img

imagesTrain = np.array(list(map(preprocess, imagesTrain)))
imagesTest = np.array(list(map(preprocess, imagesTest)))


imagesTrain_batch = imagesTrain#[1:-1:10,:]
labelsTrain_batch = labelsTrain#[1:-1:10]

imagesTest_batch = imagesTest#[1:-1:20,:]
labelsTest_batch = labelsTest#[1:-1:20]

print("Data size after shink:",imagesTrain_batch.shape,imagesTest_batch.shape)


for i in range(10,200,10):
    clf_rf = RandomForestClassifier(n_estimators=i)
    clf_rf.fit(imagesTrain_batch,labelsTrain_batch)
 
    y_pred_rf = clf_rf.predict(imagesTest_batch)
    acc_rf = accuracy_score(labelsTest_batch,y_pred_rf)
    print("n_estimators = %d, random forest accuracy:%f" %(i,acc_rf))

