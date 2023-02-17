# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters

    # Perceptron Learning Rule
    #   x is a feature vector, y is the correct class label, yHat is label; computed using current weights
    #       If y == yHat do nothing
    #       Otherwise w_i = w_i + (alpha * y * x_i).... alpha is learning rate constant
    #
    #       Variant: w_i = w_i + alpha * (y - yHat) * x_i
    #  
    #       Procedure will converge if alpha is proportional to 1000/(1000 + t)   where t is current iteration
    #
    #   There will be 32x32*3 = 3072 weights... a weight for each pixel value
    #   Calculate the sum of w*pixel value for all pixels + bias 
    #   After each image update the weights

    # Initialize weights and biases to 0
    # use numpy features
    weightArray = np.zeros(len(train_set[0]))
    b = 0

    curIt = 0
    while curIt < max_iter:
        imageNumber = 0
        for featureVect in train_set:
            sum = np.dot(weightArray, featureVect) + b
            
            if train_labels[imageNumber] == 0 and sum > 0:
                weightArray += learning_rate * -1 * featureVect
                b += learning_rate * -1
            elif train_labels[imageNumber] == 1 and sum <= 0:
                weightArray += learning_rate * featureVect
                b += learning_rate
            imageNumber += 1
        curIt += 1

    # # Iterate over training data
    # imageNumber = 0
    # for image in train_set:
    #     # print(imageNumber)
    #     sum = 0
    #     for idx in range(len(image)):
    #         sum += image[idx] * weightArray[idx]
        
    #     # Add bias term
    #     sum += b

    #     # Tweak weights
    #     yHat = 1
    #     labelCheck = 1
    #     if sum <= 0:
    #         yHat = -1
    #         labelCheck = 0
    #     # else:
    #     #     yHat = 1
    #     #     labelCheck = 1
    #     y = 1
    #     if 0 == train_labels[imageNumber]:
    #         y = -1

    #     if labelCheck != train_labels[imageNumber]:
    #         for i in range(3072):
    #             weightArray[i] += learning_rate * y * image[i]
    #         b += learning_rate * y

    #     # if labelCheck != train_labels[imageNumber]:
    #     #     y = 1
    #     #     if train_labels[imageNumber] == 0:
    #     #         y = -1
    #     #     for i in range(3072):
    #     #         weightArray[i] += learning_rate * (y - yHat) * image[i]
    #     #     b += learning_rate * (y - yHat)
    #     imageNumber += 1

    return weightArray, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    weightArray, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    # print('Training Finished')
    dev_labels = []

    for image in dev_set:
        # sum = 0
        # for idx in range(len(image)):
        #     sum += image[idx] * weightArray[idx]
        # sum += b

        # image is a np array
        sum = np.dot(weightArray, image) + b
        if sum > 0:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    #print(dev_labels)
    return dev_labels

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    # Use numpy features!!! don't try to work around like in perceptron
    # np.sqrt(), np.sum(), np.square,
    # np.argpartition(arr, k)... I think this sorts up to and including the kth index of an array

    dev_labels = []

    for image in dev_set:
        # Have to get euclidian distances from the dev_image to the training set images for all training set
        eDist = np.sqrt(np.sum(np.square(train_set - image), axis = 1))

        # Use argpartition to sort k of the closest distances... THESE ARE THEIR INDICIES
        eDistIdx = np.argpartition(eDist, k)

        # Get k closest distences to the image
        minDist = eDist[eDistIdx[:k]]
        # print(minDist)
        # print()
        # print()
        numCorrect = 0
        for d in minDist:
            # get index of minDist in the distance array, this index corresponds to train_set
            idx = np.where(eDist == d)
            # print(np.where(eDist == d))
            # print(np.where(eDist == d)[0])
            # print(d)
            # print(eDist[idx[0]][0])

            if train_labels[idx[0]][0] == 1:
                numCorrect += 1
        if numCorrect > len(minDist)/2:
            dev_labels.append(1)
        else:
            dev_labels.append(0)


    return dev_labels
