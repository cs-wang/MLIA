'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        # print featVec
        currentLabel = featVec[-1]
        # print currentLabel
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # print labelCounts
    # print labelCounts.keys()
    # print labelCounts.values()
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # print prob
        shannonEnt -= prob * log(prob,2) #log base 2
        # print shannonEnt
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        # print("featVec:%s" % featVec)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            # print(reducedFeatVec)
            reducedFeatVec.extend(featVec[axis+1:])
            # print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
            # print(retDataSet)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    # print dataSet
    for i in range(numFeatures):        #iterate over all the features
        # print("numFeatures:%s" % numFeatures)
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # print(featList)
        uniqueVals = set(featList)       #get a set of unique values
        # print uniqueVals
        newEntropy = 0.0
        for value in uniqueVals:
            # print((i, value))
            subDataSet = splitDataSet(dataSet, i, value)
            # print(subDataSet)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        # print((infoGain, baseEntropy, newEntropy))
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    # print bestFeature
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        # print classList[0]
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        # print majorityCnt(classList)
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # print bestFeatLabel
    myTree = {bestFeatLabel:{}}
    # print myTree
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # print uniqueVals
    for value in uniqueVals:
        # print value
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        # print subLabels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
        # print myTree[bestFeatLabel]
    return myTree
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
# import matplotlib.pylit as plt
def main():
    dataSet,labels = createDataSet()
    # calcShannonEnt(dataSet)
    # dataSet[0][-1]='maybe'
    # print dataSet
    # calcShannonEnt(dataSet)
    # print "==========decollator========="
    # # splitDataSet(dataSet, 1, 1)
    # # chooseBestFeatureToSplit(dataSet)
    # myTree = createTree(dataSet, labels)
    # # print myTree
    # storeTree(myTree, 'classifierStorage.txt')
    # trr = grabTree('classifierStorage.txt')
    # print trr
    print "==========decollator========="
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # print lenses
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    print lensesTree
    import treePlotter
    treePlotter.createPlot(lensesTree)





if __name__ == '__main__':
    main()

