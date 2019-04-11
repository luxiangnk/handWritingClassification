from mnist import MNIST
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import operator


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

print(imagesTrain[0].shape)

imagesTrain_batch = imagesTrain[1:-1:10,:]
labelsTrain_batch = labelsTrain[1:-1:10]

imagesTest_batch = imagesTest[1:-1:20,:]
labelsTest_batch = labelsTest[1:-1:20]

imagesTrain_batch.astype(int)
imagesTest_batch.astype(int)
labelsTrain_batch.astype(int)
labelsTest_batch.astype(int)

print(imagesTrain_batch.shape)
print(imagesTest_batch.shape)
print(len(imagesTest_batch))

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[int(sortedDistIndicies[i])]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == "__main__":
    
    errorCount = 0.0
    for i in range(len(imagesTest_batch)):
        classifierResult = classify(imagesTest_batch[i], imagesTrain_batch, labelsTrain_batch, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, labelsTest_batch[i]))
        if (classifierResult != labelsTest_batch[i]): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount) 
    print("\nthe total error rate is: %f" % (errorCount/float(len(imagesTest_batch)))) 
    