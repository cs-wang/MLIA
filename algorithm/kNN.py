#encoding:utf-8
from numpy import *
import operator
#创建数据集
def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels
#kNN实现0
def classify0(inX, dataSet, labels, k):
	#shape 返回行列数，shape[0]是行数，有多少元组
	dataSetSize = dataSet.shape[0]
	#tile 复制inX，使其与dataset一样大小
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	#**表示乘方
	sqDiffMat = diffMat ** 2
	#按行将计算结果求和
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	#使用argsort排序，返回索引值  
	sortDistIndicies = distances.argsort()
	#用于计数，计算结果
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	#按照第二个元素降序排列
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	#返回出现次数最多的那一个label的值
	return sortedClassCount[0][0]

#从txt中读入数据  
def file2matrix(filename):                         
    fr = open(filename)
    #打开文件，按行读入
    arrayOLines = fr.readlines()    
    #获得文件行数 
    numberOfLines = len(arrayOLines)  
    #创建m行n列的零矩阵 
    returnMat = zeros((numberOfLines,3))          
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        #删除行前面的空格
        listFromLine = line.split('\t')
         #根据分隔符划分
        returnMat[index,:] = listFromLine[0:3]
        #取得每一行的内容存起来
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector


#归一化数据
def autoNorm(dataSet):
	#找出样本集中的最小值
    minVals = dataSet.min(0)          
    #找出样本集中的最大值
    maxVals = dataSet.max(0)        
    #最大最小值之间的差值
    ranges = maxVals - minVals          
    #创建与样本集一样大小的零矩阵
    normDataSet = zeros(shape(dataSet))           
    m = dataSet.shape[0]                         
    #样本集中的元素与最小值的差值
    normDataSet = dataSet - tile(minVals, (m,1))
    #数据相除，归一化
    normDataSet = normDataSet/tile(ranges, (m,1))  
    return normDataSet, ranges, minVals


def datingClassTest():
	#选取多少数据测试分类器
    hoRatio = 0.10
    #从datingTestSet2.txt中获取数据
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       
    #归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat) 
    m = normMat.shape[0]
    #设置测试个数
    numTestVecs = int(m*hoRatio)
    #记录错误数量
    errorCount = 0.0                               
    for i in range(numTestVecs):
    	#分类算法
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)  
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0  
    #计算错误率
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount

def main():
	import numpy
	import matplotlib
	import matplotlib.pyplot as plt
	import kNN
	datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*numpy.array(datingLabels),15.0*numpy.array(datingLabels))

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*numpy.array(datingLabels),15.0*numpy.array(datingLabels))
	plt.show()

if __name__ == '__main__':
	main()
