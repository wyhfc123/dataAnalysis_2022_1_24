# -*-coding:utf-8-*-
from math import log
import operator


def createDataset():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    # no surfacing :不浮出水面  flippers: 脚蹼
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


myDat, labels = createDataset()


# myDat[0][-1]='maybe'
# print(myDat)
# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    # 获得数据集大小
    numEntries = len(dataSet)

    labelCounts = {}
    for fecVec in dataSet:
        # 取分类信息
        currentLabel = fecVec[-1]
        # print(currentLabel,'25')
        # 如果当前分类不在字典labelCounts的键中，则让该键的值等于0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 不管有没有该键，值都相加
        labelCounts[currentLabel] += 1

    # print(labelCounts,'30行')
    shannoEnt = 0.0
    # 遍历labelCounts字典的键
    for key in labelCounts:
        # 取出该字典所有的值，并让其除以数据集的大小  p(Xi)
        prob = float(labelCounts[key]) / numEntries
        # print(prob,'37')
        shannoEnt -= prob * log(prob, 2)

        # print(shannoEnt)
    # 香农熵
    return shannoEnt


# res=calcShannonEnt(myDat)
# print(res)
# dataSet:带划分的数据集 axis:划分数据集的特征 value:需要返回特征的值
def splitDataSet(dataSet, axis, value):
    # print(dataSet,'51行',axis,value)
    # 创建新的列表
    retDataSet = []
    # 获得数据中的数据
    # print(dataSet,'44行')
    for featVec in dataSet:
        print(featVec[axis], value, axis,"57****")
        # 判断数据中的该列特征数据是否等于该值
        if featVec[axis] == value:
            # print(featVec[:axis],'48行')
            # 去掉axis的特征
            reducedFeatVec = featVec[:axis]
            # 则把该值之后的所有特征添加到reducedFeatVec中
            reducedFeatVec.extend(featVec[axis + 1:])
            # print(featVec[axis+1:],'60行---------------------')
            retDataSet.append(reducedFeatVec)
    # print(retDataSet,'59')
    return retDataSet


# res1=splitDataSet(myDat,0,1)
# res2=splitDataSet(myDat,0,0)
# #选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    # 获取特征长度（除去最后一列分类）
    numFeatures = len(dataSet[0]) - 1
    # 原始数据集香农熵
    baseEntropy = calcShannonEnt(dataSet)  # 0.9709505944546686
    # print(baseEntropy,'69')
    # 最好的信息是0.0
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 取出每一个特征列的值
        featList = [example[i] for example in dataSet]

        # print(dataSet)
        # print(featList,'75行')
        # 将其转为集合去重(不能有重复)
        uniqueVals = set(featList)
        # print(set(dataSet[i]),87)
        # print(uniqueVals)

        newEntropy = 0.0
        # print(uniqueVals,'83')

        # 遍历集合，计算信息增益
        for value in uniqueVals:
            print(i, 95)
            # 划分数据集 i从2开始截至
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算数据集的新熵
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # print('****************')
        # print(infoGain,'90',i)
        # print('****************')
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
        # print(bestInfoGain,'96******************')
    return bestFeature


# data=chooseBestFeatureToSplit(myDat)
# print(data)
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclassCount[0][0]


# 创建决策树
def createTree(dataSet, labels):
    # 类别列表
    classList = [example[-1] for example in dataSet]
    # print(classList,'116')
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)

    bestFeatLabel = labels[bestFeat]
    print(bestFeatLabel, 126)

    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        sublabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), sublabels)
    return myTree


res = createTree(myDat, labels)
print(res)
import matplotlib.pyplot as plt
# 解决中文问题
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)

decisionNode = dict(boxstyle='sawtooth', fc="0.8")
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                            va="center", bbox=nodeType, arrowprops=arrow_args, fontproperties=font)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode("决策节点", (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode("叶节点", (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


createPlot()
def getNumLeafs(myTree):
    numleafs=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            numleafs+=getNumLeafs(secondDict[key])
#         else:
            numleafs+=1
    return numleafs
def getTreeDepth(myTree):
    maxDepth=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth
getNumLeafs(res)
getTreeDepth(res)