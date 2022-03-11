import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy.random
import time

data = pd.read_csv("sklearn_data/LogiReg_data.txt", header=None, names=['成绩1', '成绩2', '通过'])


# print(pd.value_counts(data['通过']))
# print(data.shape)
# print(data.head())
# data_one=data[data['通过']==1]
# data_two=data[data['通过']==0]
# fig,ax=plt.subplots(figsize=(20,10))
# ax.scatter(data_one['成绩1'],data_one['成绩2'],c='b',s=50,marker='o',label='通过')
# ax.scatter(data_two['成绩1'],data_two['成绩2'],c='r',s=50,marker='x',label='不通过')
# ax.legend()
# ax.set_xlabel('测试1的分数')
# ax.set_ylabel('测试2的分数')
# plt.show()
def sigmod(z):
    
    return 1 / (1 + np.exp(-z))


# num=np.arange(-10,10,1)#创建分类器，在x轴上设置数值间距为1（横轴）
# fig,ax=plt.subplots(figsize=(12,4))#figsize设置图形的宽高，其中宽度为12，高度为4
# ax.plot(num,sigmod(num),'-r')#通过sigmod函数计算横轴值得出该分类器在y轴上的数值,'-r':线的颜色
# plt.show()#画图


def model(X, theta):
    #计算所有样本的sigmod值
    return sigmod(np.dot(X, theta.T))  # np.dot(X,theta.T):算出来的结果为矩阵x乘于矩阵theta.T的值


data.insert(loc=0, column='Ones', value=1)  # 添加一列到datafreme对象中

orig_data = data.values  # 将datafreme对象转化为二维数组ndarray对象
# print(orig_data)
col = orig_data.shape[1]  # 0代表行，1代表列,结果等于4
# print(col)
X = orig_data[:, 0:col - 1]  # 获取从0列到第三列的数据,这半部分数据做特征向量
# print(X)
y = orig_data[:, col - 1:]  # 获取最后一列数据（本列数据为预测值）
# print(Y)
theta = np.zeros([1, 3])  # 创建一行三列零矩阵


# print(theta)
#                                                 n
# 梯度下降计算（损失函数）     公式为𝐷(ℎ𝜃(𝑥),𝑦)= - ∑ 𝑦log(ℎ𝜃(𝑥))−(1−𝑦)log(1−ℎ𝜃(𝑥))
#                                                𝑖=1

#                                          n
# 梯度下降计算（损失函数）     公式为𝐷(𝑦’,𝑦)= - ∑ 𝑦log(𝑦')−(1−𝑦)log(1−𝑦')
#                                         𝑖=1

# 求平均损失         则𝐽(𝜃)=1/n𝐷(ℎ𝜃(𝑥),𝑦)
# 定义损失函数
def cost(X, y, theta):
    # model(X,theta)：ℎ𝜃(𝑥)
    print(X,y,model(X, theta))
    left = np.multiply(-y, np.log(model(X, theta)))  # 公式为-𝑦log(ℎ𝜃(𝑥))，其中ℎ𝜃(𝑥)=g(𝜃(𝑥))=1/1+e的-𝜃Tx次密

    right = np.multiply(1 - y, np.log(1 - model(X, theta)))  # 公式为(1−𝑦)log(1−ℎ𝜃(𝑥))
    # print(left,right)
    return np.sum(left - right) / (len(X))  # 此步骤转换为梯度下降


# print(cost(X,y,theta))
# 计算梯度
#                   ∂𝐽       1   𝑛
# 对𝜃进行求导，公式为 —  = - —  ∑ (𝑦𝑖−ℎ𝜃(𝑥𝑖))𝑥𝑖𝑗
#                  ∂𝜃𝑗      𝑚  𝑖=1
# 定义梯度函数

def gradient(X, y, theta):
    grad = np.zeros(theta.shape)  # 定义梯度,写一占位符，占位符以0进行站位，生成的grad为1*3零矩阵

    error = (model(X, theta) - y).ravel()  # ℎ𝜃(𝑥𝑖）-yi  ravel():将矩阵a重新拉伸成一个向量，拉伸后可以重新reshape成一个新矩阵
    for j in range(len(theta.ravel())):  # 给其设定梯度下降3次
        term = np.multiply(error, X[:, j])  # 梯度每下降一次，就取X中的数据和error相乘一次
        # grad[0,j] : 取第0个下标的第j个元素
        grad[0, j] = np.sum(term) / len(X)  # 替换该占位符中的0
        # print(j)
        # print(grad[0,j],'65行')
        # print(grad,'66行')
    return grad


# 比较3中不同梯度下降方法
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


# type ：停止策略  value为最大的停止数，threshold :该停止策略对应的阈值
def stopCriterion(type, value, threshold):
    # print(value)
    if type == STOP_ITER:  # 如果停止策略是最大迭代次数
        return value > threshold  # 则返回value大于阈值
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold  # norm:表示范数


def shuffleData(data):
    np.random.shuffle(data)

    cols = data.shape[1]

    X = data[:, 0:cols - 1]
    y = data[:, cols - 1:]
    return X, y


# shuffleData(orig_data)


n = 100

#theta  # 创建一行三列零矩阵
# data:数据  theta：参数 batchSize匹配随机下降算法，根据给定的值，确定使用哪种算法 stopType：停止策略 thresh：策略对应的阈值，alpha：学习率
def descent(data, theta, batchSize, stopType, thresh, alpha):
    init_time = time.time()  # 查看时间对结果的影响
    i = 0  # 初始化迭代次数，从0次开始
    k = 0  # 初始化算法，从第0个batch开始
    X, y = shuffleData(data)  # 获取洗牌后重新返回的X数据集，y预测列（）
    grad = np.zeros(theta.shape)  # 定义一个占位符，以便后期使用
    costs = [cost(X, y, theta)]  # 获取损失值
    while True:
        grad = gradient(X[k:k + batchSize], y[k:k + batchSize], theta)  # 获取不同的梯度下降算法的偏导数值
        k += batchSize  # 每获取一次，迭代次数+1
        if k >= n:  # 如果batch次数切换达到100次时，重新给其初始化
            k = 0  # 初始化算法，从0开始切换
            X, y = shuffleData(data)  # 重新洗牌
        theta = theta - alpha * grad  # 参数更新，新的参数等于旧参数-学习率*下降梯度（偏导数）
        costs.append(cost(X, y, theta))  # 通过损失函数计算新的损失，并将其返回出来添加到costs中（用于画图）
        i += 1  # 对迭代次数+1
        value = 0
        if stopType == STOP_ITER:  # 如果停止策略等于停止迭代
            value = i  # 则最大迭代次数复制给value
        elif stopType == STOP_COST:  # 如果停止策略等于损失函数（两次损失函数相差无几）

            value = costs  # 则该包含所有损失函数值的列表复制给value

        elif stopType == STOP_GRAD:  # 如果停止策略等于梯度（当梯度下降趋近于0时，停止回归）
            value = grad
        if stopCriterion(stopType, value, thresh):  # 返回结果如果为True，则结束循环
            break
            # 返回新的参数值，新的迭代次数值，新的损失值，新的梯度值，新的时间值
    return theta, i - 1, costs, grad, time.time() - init_time


def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    # 获取下降后的最新值，并赋值新的变量
    # print(batchSize,'122')
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    # data[:,1]>2 判断是否为True，如果为True,则求和大于1，如果为False则小于1
    name = 'Original' if (data[:, 1] > 2).sum() > 1 else 'Scaled'  # 如果第一列的数据大于2,且求和大于1，则开始计算
    name += '学习率为{} '.format(alpha)
    if batchSize == n:
        strDesctype = '批量梯度下降开始：'  # 从右往左运算，先复制，后判断
    elif batchSize == 1:
        strDesctype = '随机梯度下降开始：'
    else:
        strDesctype = '小批量梯度下降开始：{}'.format(stopType)
    name += strDesctype + '停止下降:'
    # print(stopType,STOP_COST,'137')
    if stopType == STOP_ITER:
        strstop = '最大迭代阈值为:{}'.format(thresh)
    elif stopType == STOP_COST:
        strstop = '最大函数损失阈值<{}'.format(thresh)
    else:
        strstop = '最大梯度下降<{}'.format(thresh)
    name += strstop
    print(name, '\n')
    return theta


runExpe(orig_data, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)
# runExpe(orig_data, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)
# runExpe(orig_data, theta, n, STOP_GRAD, thresh=0.005, alpha=0.001)
# runExpe(orig_data, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)
# runExpe(orig_data, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)

from sklearn import preprocessing as pp  # 利用sklearn建模

scale_data = orig_data.copy()
scale_data[:, 1:3] = pp.scale(orig_data[:, 1:3])
theta = runExpe(scale_data, theta, 1, STOP_GRAD, thresh=0.002 / 5, alpha=0.001)  # 阈值不同时，准确率会跟着相应不同


# runExpe(scale_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)
# runExpe(scale_data, theta, 1, STOP_GRAD, thresh=0.002/5, alpha=0.001)
# print(theta)
# 设定阈值


def predict(X, theta):
    # print("X:  ",X,"theta: ",theta)
    # 这里设置大于0.5通过，不大于0.5未通过
    # 如果通过则就输出1，不通过输出0
    return [1 if x >= 0.5 else 0 for x in model(X, theta)]  # model(X,theta):返回的是二维数组，遍历出的一维数组可以直接跟数值进行判断


scale_X = scale_data[:, :3]  # 获取数据集结果前的数据

y1 = scale_data[:, 3]  # 获取最后一列数据集结果
predictions = predict(scale_X, theta)  # 获取包含通过的和不通过的一维数组
# print(predictions,'172行')
# print(y1,'173行')
# for i in zip(predictions,y1):
#     print(i)

# 将预测的值和最后一列数据集结果通过zip()打包，遍历，并赋值给两个变量，当预测值和最后一列数据集结果相等时，即给其赋值为1，否则赋值为0
# 获取后的current为一维数组
current = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y1)]

# 将所得结果相加并除以该一维数组的总数，即为准确率(此间map()函数可以不写)
# accuracy=(sum(map(int,current))%len(current))
accuracy = (sum(current) % len(current))
print(accuracy)
