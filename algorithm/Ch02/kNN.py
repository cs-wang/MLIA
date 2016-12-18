'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir
import logging

def classify0(inX, dataSet, labels, k):
    logging.info(inX)
    logging.info(dataSet)
    logging.info(labels)
    logging.info(k)
    dataSetSize = dataSet.shape[0]
    logging.info(dataSetSize)
    tileMat = tile(inX, (dataSetSize,1))
    logging.info(tileMat)
    diffMat = tileMat - dataSet
    logging.info(diffMat)
    sqDiffMat = diffMat**2
    logging.info(sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    logging.info(sqDistances)
    distances = sqDistances**0.5
    logging.info(distances)
    sortedDistIndicies = distances.argsort()     
    logging.info(sortedDistIndicies)
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        logging.info("voteIlabel: %s" % voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        logging.info("classCount: %s" % classCount)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    logging.info("sortedClassCount: %s" % sortedClassCount)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
# set_printoptions(threshold='nan')
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    # logging.info(trainingFileList)
    m = len(trainingFileList)
    # logging.info(m)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        # logging.info(fileNameStr)
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        # logging.info(fileStr)
        classNumStr = int(fileStr.split('_')[0])
        # logging.info(classNumStr)
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

def main():
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt
    import kNN

    logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',  
                    datefmt='%a, %d %b %Y %H:%M:%S',  
                    filename='./test.log',  
                    filemode='w')  
    # group,labels = kNN.createDataSet()
    # print kNN.classify0([0, 0], group, labels, 3)
    # datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*numpy.array(datingLabels),15.0*numpy.array(datingLabels))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*numpy.array(datingLabels),15.0*numpy.array(datingLabels))
    # plt.show()
    #kNN.classifyPerson()
    # testVector = kNN.img2vector('testDigits/0_13.txt')
    # print testVector[0, 0:31]
    # print testVector[0, 32:63]

    kNN.handwritingClassTest()


if __name__ == '__main__':
    main()
