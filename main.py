# _*_coding:utf-8 _*_
# numpy  :数据分析和机器学习的底层库

# import numpy as np
# arr=np.arange(10)
# print(arr.shape)
# print(arr.ndim)
# print(arr.size)
# print(arr.itemsize)
# print(arr.dtype)
# print(arr.nbytes)
# arr_r = arr.reshape(2,5)
# print(arr_r)
# print(arr)
# print("************************************")
# a = np.arange(12).reshape(3,4)
# b = np.arange(12,24).reshape(3,4)
# print(a,b)
# print(a+b)
# print(a*b)
# print(a/b)
# print(a-b)
# print(a%b)
# print(a * 3)
# print(a + 3)
# print(a / 3)
# print(a % 3)
# print(a - 3)


# a.shape = (4,3)
# print(a)

# b = a.astype("float32")
# print(b)
# print(a)
# start = 0
# end = 4
# length = 2
# print(a[start:end:length,start:end-1:1])


# arr = np.arange(24).reshape(2,3,4)
# print(arr[1,0:3:1,0:4:2])

# arr = np.arange(24).reshape(2,3,2,2)
# print(arr)
# print(arr[0,1,1,0])

# datas=[
#     ("zhang3",[10,21,22],17),
#     ("tom",[11,31,52],21),
#     ("jack",[13,13,20],31)
# ]
# # arr = np.array(datas,dtype="U7,3int8,int8")
# # print(arr)
# arr = np.array(datas,dtype=[
#     #別名,类型，字节数
#     ("name","str_",8),
#     ("scores","int32",3),
#     ("age","int32",1),
# ])
# print(arr[0]["name"])
#
# arr  = np.array(datas,dtype = {
#     "names":["name","score","age"],
#     "formats":["U8","3int32","int32"]
# })
# print(arr[1]["score"])


# a= np.array(datas,dtype={
#     # 别名:(类型,字节偏移量)
#     "names":("U8",0),
#     "scores":("3int32",33),
#     "age":("int32",45)
# })
# print(a)

# a=np.array([0x1234,0x5667],dtype=(
#     "u2",{
#         "lowc":("u1",0),
#         "hignc":("u1",1),
#     }))

#日期数据格式
# from matplotlib.font_manager import FontManager

# date=["2020-01-02","1995-01-03","2017-01","2019-01-03 02:30:00"]
# array_date=np.array(date)
# array_date=array_date.astype("M8[Y]")
# print(array_date)
# array_date=array_date.astype("M8[M]")
# print(array_date)
# array_date=array_date.astype("M8[D]")
# print(array_date)

# print(array_date[2] - array_date[1])
# print(array_date.astype("int32"))   #从1970年到现在的天数
# array_date=array_date.astype("M8[h]")
# print(array_date)
# array_date=array_date.astype("M8[m]")
# print(array_date)
# array_date=array_date.astype("M8[s]")
# print(array_date)

# a = np.arange(15)
# s=a.reshape(3,5)
# print(s)
# s[0,0]=200
# print(s)
# print(a)
# a1 = a.ravel()
# print(a1)
# a1[0] = 300
# print(a1)
# print(a)

# a1 = a.flatten()
# print(a1)
# a1[0]=500
# print(a1)
# print(a)
# a.shape = (5,3)
# print(a)
# a.resize((3,5))
# print(a)

#数组的掩码操作
#bool掩码
# mask = [True,False,True,False,True,False]
# arr = np.arange(6)
# print(arr[mask])

#索引掩码
# mask_num = [1,4,2,3,0,5]
# arr = np.array(["a","b","c","d","e","f"])
# print(arr[mask_num])

# arr = np.arange(6)
#获取大于2的数
# print(arr[arr > 2])
# print(arr[(arr > 2) & (arr < 4)])

#数组的合并于拆分
# a = np.arange(12).reshape(3,4)
# b = np.arange(12,24).reshape(3,4)
#水平方向合并
# h_c=np.hstack((a, b))
# print(h_c)

#垂直方向合并
# v_c =np.vstack((a,b))
# print(v_c)

#深度方向合并
# d_c=np.dstack((a,b))
# print(d_c)

#水平方向切分
# a,b = np.hsplit(h_c,2)
# print(a,b)
#垂直方向切分
# a,b,c = np.vsplit(h_c,3)
# print(a,b,c)
#深度方向切分
# a,b = np.dsplit(d_c,2)
# print(a,b)

#按轴合并
# a_c = np.concatenate((a,b),axis=0)
# print(a_c)

#按轴切分
# a,b = np.split(h_c,2,axis=1)
# print(a,b)

#一维数组组合方案
# a = np.arange(6)
# b = np.arange(6,12)
# s=np.row_stack((a,b))
# print(s)
# s = np.column_stack((a,b))
# print(s)
#填充数组
# a = np.arange(6)
# b = np.arange(5)
# print(a,b)
# c = np.pad(b,pad_width=(0,1),mode="constant",constant_values=0)
# print(a,b,c)
# print(a + c)

#常用属性
# arr = np.arange(20).reshape(4,5)
# print([i for i in arr.flat])
# print(arr.T)
# print(arr.transpose())
# print(arr.shape)
# a = np.ones((3, 4))
# b = np.zeros((3, 4))
# print(np.zeros_like(a))
# print(np.ones_like(b))
# print(np.eye(3,3))



# import numpy as np

# from matplotlib import pyplot as plt
#
#
# plt.figure(num="red",facecolor="red")
# plt.figure(num="blue",facecolor="blue")
# plt.show()
#
# from matplotlib import pyplot as plt
# from matplotlib import gridspec as mg
# from matplotlib import pyplot as plt
# plt.figure("Flow LayOut",facecolor="lightgray")
#
#         # x     y   width higtht
# plt.axes([0.03,0.5,0.94,0.4])
# plt.text(0.5,0.5,"1",ha = "center",va="center",size=36)
# plt.axes([0.03,0.03,0.54,0.4])
# plt.text(0.5,0.5,"1",ha = "center",va="center",size=36)
#
# plt.show()

# from matplotlib import pyplot as plt
# plt.figure("Grid Line",facecolor="lightgray")
# ax = plt.gca()
# # ax.grid()
# ax.grid(which ="major",axis="both",color="red",linewidth=0.75)
# #绘制曲线
# y=[1,10,100,1000,100,10,1]
# plt.plot(y,"o-",color="blue")
# plt.show()
#
# import numpy as np

# print(np.random.normal())
# print(4/(20/3))
# d=list(map(int,np.random.sample(12)*100))
# print(d)

# import tushare as ts
# print(ts.get_today_all())

# from matplotlib import pyplot as plt
#新增加的两行
# import matplotlib
# matplotlib.rc("font",family='FangSong')
# mpl_fonts = set(f.name for f in FontManager().ttflist)
#
# print('all font list get from matplotlib.font_manager:')
# for f in sorted(mpl_fonts):
#     print('\t' + f)
# for i in FontManager().ttflist:
#     print(i.name)
# a = ["一月份","二月份","三月份","四月份","五月份","六月份"]
#
# b=[56.01,26.94,17.53,16.49,15.45,12.96]
#
# plt.figure(figsize=(20,8),dpi=80)
#
# plt.bar(range(len(a)),b)
#
# #绘制x轴
# plt.xticks(range(len(a)),a)
#
# plt.xlabel("月份")
# plt.ylabel("数量")
# plt.title("每月数量")
#
# plt.show()
# import numpy as np
# from matplotlib import dates as md
# from matplotlib import pyplot as mp
# from datetime import datetime
# def dmy2ymd(dmy):
#     dmy = str(dmy,encoding="utf-8")
#     time = datetime.strptime(dmy,"%d-%m-%Y").date()
#     t = time.strftime("%Y-%m-%d")
#     return t
# dates, bhp_closing_prices = np.loadtxt('da_data/bhp.csv',
#                                        delimiter=',',usecols=(1, 6), unpack=True,
#                                        dtype='M8[D], f8', converters={1: dmy2ymd})
# vale_closing_prices = np.loadtxt('da_data/vale.csv', delimiter=',',
#                                  usecols=(6), unpack=True)
# diff_closing_prices = bhp_closing_prices - vale_closing_prices
# days = dates.astype(int)
# p = np.polyfit(days, diff_closing_prices, 5)
# poly_closing_prices = np.polyval(p, days)
# q = np.polyder(p)
# roots_x = np.roots(q)
# roots_y = np.polyval(p, roots_x)
# mp.figure('Polynomial Fitting', facecolor='lightgray')
# mp.title('Polynomial Fitting', fontsize=20)
# mp.xlabel('Date', fontsize=14)
# mp.ylabel('Difference Price', fontsize=14)
# ax = mp.gca()
# ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=md.MO))
# ax.xaxis.set_minor_locator(md.DayLocator())
# ax.xaxis.set_major_formatter(md.DateFormatter('%d %b %Y'))
# mp.tick_params(labelsize=10)
# mp.grid(linestyle=':')
# dates = dates.astype(md.datetime.datetime)
# mp.plot(dates, poly_closing_prices, c='limegreen',
#         linewidth=3, label='Polynomial Fitting')
# mp.scatter(dates, diff_closing_prices, c='dodgerblue',
#            alpha=0.5, s=60, label='Difference Price')
# roots_x = roots_x.astype(int).astype('M8[D]').astype(
#     		md.datetime.datetime)
# mp.scatter(roots_x, roots_y, marker='^', s=80,
#            c='orangered', label='Peek', zorder=4)
# mp.legend()
# mp.gcf().autofmt_xdate()
# mp.show()
# import numpy as np
# a=np.array([1,2,3])
# a1=np.array([1,2])
#
# a2=np.array([3,4])
# print(a1.ndim,a2.ndim,a.ndim)
# print(a.compress(( a1[0]> a2[0])))
import numpy as np
# print(original)
# import pandas as pd
#
# dates = pd.Series(["2011","2011-02","2011-03-01","2011/04/01","2011/05/01 01:01:01","01Jun 2011"])
# dates = pd.to_datetime(dates)
# # print(dates,type(dates),dates.dtype)
# # datetime类型数据支持日期运算
# delta = dates - pd.to_datetime('2010-01-01')  #日期偏移量
# # 获取天数数值
# print(delta.dt.days)
# # 获取秒数数值
# print(delta.dt.seconds)
#
#
# print(type(dates),66)
# print(type(delta))
#
# import pandas as pd
# left = pd.DataFrame({
#          'student_id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
#          'student_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung', 'Billy', 'Brian', 'Bran', 'Bryce', 'Betty', 'Emma', 'Marry', 'Allen', 'Jean', 'Rose', 'David', 'Tom', 'Jack', 'Daniel', 'Andrew'],
#          'class_id':[1,1,1,2,2,2,3,3,3,4,1,1,1,2,2,2,3,3,3,2],
#          'gender':['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
#          'age':[20,21,22,20,21,22,23,20,21,22,20,21,22,23,20,21,22,20,21,22],
#          'score':[98,74,67,38,65,29,32,34,85,64,52,38,26,89,68,46,32,78,79,87]})
# right = pd.DataFrame(
#          {'class_id':[1,2,3,5],
#          'class_name': ['ClassA', 'ClassB', 'ClassC', 'ClassE']})
# # 合并两个DataFrame
# data = pd.merge(left,right)
# print(data.pivot_table(index=['class_id', 'gender'], values=['score'],
#                        columns=['age']))

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API

from pytdx.params import TDXParams

# api = TdxHq_API()
# exapi = TdxExHq_API()
# BLOCK_SZ = "block_zs.dat"
# BLOCK_FG = "block_fg.dat"
# BLOCK_GN = "block_gn.dat"
# BLOCK_DEFAULT = "block.dat"
# with api.connect('119.147.212.81', 7709):
#     wk = api.get_security_bars(7,0,"000831", 0, 800)
#     wk_df = api.to_df(wk)
#     #拿到五矿稀土今天的数据
#     print(wk_df[wk_df["datetime"]>"2022-01-25 00:00"])
    # stock_df = api.to_df(api.get_security_list(0,1))
    # print(stock_df)
    # print(stock_df["code"])
    # content = api.to_df(api.get_company_info_category(TDXParams.MARKET_SZ, "000681"))
    #
    # data = api.get_company_info_content(0, '000001', '000001.txt', 0,100000)
    # print(data)
    # print(api.to_df(api.get_and_parse_block_info(TDXParams.BLOCK_SZ)))
    # exapi.get_instrument_info(0,)

# import cv2 as cv
# image = cv.imread("1.jpg")
# GRAY=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv.imshow("GRAY",GRAY)
# cv.waitKey(0)
# cv.destroyAllWindows()


# import numpy as np
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import axes3d   #必须导入
#
# n=1000
# x,y = np.meshgrid(np.linspace(-3,3,n),np.linspace(-3,3,n)) #x,y直接组成坐标点矩阵
# z = x*y
# plt.figure("3D Surface",facecolor="lightgray")
# # ax3d = plt.gca(projection="3d")
# # ax3d.set_xlabel("x")
# # ax3d.set_ylabel("y")
# # ax3d.set_zlabel("z")
# ax3d = plt.axes(projection="3d")
# ax3d.set_xlabel("x")
# ax3d.set_ylabel("y")
# ax3d.set_zlabel("z")
#
# ax3d.plot_surface(x,y,z,cstride=30,rstride=30,cmap="jet")
# plt.show()
# import numpy as np
# import matplotlib.pyplot as mp
# train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
# train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])
# test_x = np.array([0.45, 0.55, 1.0, 1.3, 1.5])
# test_y = np.array([4.8, 5.3, 6.4, 6.9, 7.3])
#
# times = 1000	# 定义梯度下降次数
# lrate = 0.01	# 记录每次梯度下降参数变化率
# epoches = []	# 记录每次梯度下降的索引
# w0, w1, losses = [1], [1], []
# for i in range(1, times + 1):
#     epoches.append(i)
#     loss = (((w0[-1] + w1[-1] * train_x) - train_y) ** 2).sum() / 2
#     losses.append(loss)
#     d0 = ((w0[-1] + w1[-1] * train_x) - train_y).sum()
#     d1 = (((w0[-1] + w1[-1] * train_x) - train_y) * train_x).sum()
#     print('{:4}> w0={:.8f}, w1={:.8f}, loss={:.8f}'.format(epoches[-1], w0[-1], w1[-1], losses[-1]))
#     w0.append(w0[-1] - lrate * d0)
#     w1.append(w1[-1] - lrate * d1)
#
# pred_test_y = w0[-1] + w1[-1] * test_x
# w0 = w0[:-1]
# w1 = w1[:-1]
#
# import mpl_toolkits.mplot3d as axes3d
#
# grid_w0, grid_w1 = np.meshgrid(
#     np.linspace(0, 9, 500),
#     np.linspace(0, 3.5, 500))
#
# grid_loss = np.zeros_like(grid_w0)
# for x, y in zip(train_x, train_y):
#     grid_loss += ((grid_w0 + x*grid_w1 - y) ** 2) / 2
#
# mp.figure('Loss Function')
# ax = mp.axes(projection="3d")
# mp.title('Loss Function', fontsize=20)
# ax.set_xlabel('w0', fontsize=14)
# ax.set_ylabel('w1', fontsize=14)
# ax.set_zlabel('loss', fontsize=14)
# ax.plot_surface(grid_w0, grid_w1, grid_loss, rstride=10, cstride=10, cmap='jet')
# ax.plot(w0, w1, losses, 'o-', c='orangered', label='BGD')
# mp.legend()
# mp.show()
# import  numpy as np
# aa = np.array([])
# a=np.array([2, 2, 2, 2, 2])
# aa.append([1,2])
# print(aa)

# a= np.arange(1,7).reshape(2,3)
# print(a.shape[1])


# # 案例：预测波士顿地区房屋价格。
# import sklearn.datasets as sd
# # 打乱数据集用
# import sklearn.utils as su
# import sklearn.ensemble as se
# import numpy as np
# # 获得波士顿地区房屋价格的第一种方式，注意：load_boston()函数在1.2版本中已经被分离
# boston = sd.load_boston()
# x, y = su.shuffle(boston.data, boston.target, random_state=7)
# # print(boston.data.shape,boston.target.shape)
# # print(boston.feature_names,len(boston.feature_names))
#
# # 划分训练集和测试集
# train_size = int(len(x) * 0.8)  #从总样本中挑出80%用作训练集，剩下20%用作测试集
# train_x, train_y, test_x, test_y = x[:train_size], y[:train_size], x[train_size:], y[train_size:]
#
# # train_y = train_y.reshape(-1,1)
# # data = d=np.hstack((train_x,train_y))
# # np.savetxt("1.csv",data,delimiter=',')
#
#
# def split_sample_space(y:"result set"):
#      # 通过结果集划分子样本空间
#     sub_sample_set = {}
#     # print(y,type(y))
#     for data in y:
#         if data not in sub_sample_set.keys():
#             sub_sample_set[data] = 0
#         sub_sample_set[data] += 1
#     return sub_sample_set
#
#
#
# #计算信息熵
# def calcInfoEnt(x:"Feature set",y:"result set"):
#
#     sub_sample_set = split_sample_space(y)
#     # 信息熵公式 H(X) = = -∑P(Xᵢ)log₂(P(Xᵢ))
#     sample_length = len(x)
#
#     # 计算信息熵
#     infoEnt = 0.0
#     for key,value in sub_sample_set.items():
#         '''
#             :key     子样本空间划分依据(程序中未使用)
#             :value   子样本空间大小
#         '''
#         # 根据子样本空间大小计算P(Xᵢ)
#         pxi = float(value) / sample_length
#         # 计算 ∑P(Xᵢ)log₂(P(Xᵢ))
#         infoEnt += pxi * np.log2(pxi)
#     #返回H(X)
#     return -infoEnt
#
# #划分数据集
# def split_dataSet(x:"Feature set",y:"result set",split_index:"column index",unique_value):
#     newdataSet=[]
#     newResultSet=[]
#     for row in x:
#         # 判断该列值是否等于唯一值
#         if row[split_index] == unique_value:
#             # print(unique_value,917)
#
#             #获取该数的行级索引
#             row_index = np.where((x == row).all(axis = 1))
#             # print(row_index,924)
#
#             subDataSet = list(row[:split_index])
#             #根据唯一值获取第2-n所构成的子表
#             subDataSet.extend(row[split_index+1:])
#             # 将子表添加到新的数据集中
#             newdataSet.append(np.array(subDataSet))
#
#             # 将子表对应的结果添加到新的结果集中
#             newResultSet.append(y[row_index])
#
#
#
#
#
#     return np.array(newdataSet),np.array(newResultSet).flatten()
#
# #对结果集进行投票排序
# import operator
# def majorityCnt(resultSet):
#     classCount = {}
#     for vote in resultSet:
#         if vote not in classCount.keys():
#             classCount[vote] = 0
#         classCount[vote] += 1
#     sortedclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
#     return sortedclassCount[0][0]
#
#
# #选择最好的数据集划分方式
# def choose_best_feature_to_split(x:"Feature set",y:"result set"):
#
#     #计算未切分前数据集信息熵
#     baseInfoEnt = calcInfoEnt(x,y)
#     #最好的信息增益为0.0
#     bestInfoGain = 0.0
#     #设置初始值
#     bestFeature = -1
#     #获取特征集的长度
#     sample_length = x.shape[1]
#     for i in range(sample_length):
#
#         # 获取数据集的每列特征值
#         columns_feature_value_list=[row[i] for row in x]
#         #分表
#         unique_value_list = list(set(columns_feature_value_list))
#         newInfoEnt =  0.0
#         for unique_value in unique_value_list:
#             #切分数据集
#             newdataSet,newResultSet = split_dataSet(x,y,i,unique_value)
#             #计算子表的熵
#             proportion = len(newdataSet) / float(len(x))
#             newInfoEnt += proportion * calcInfoEnt(newdataSet,newResultSet)
#
#         #计算信息增益    划分前的信息熵 - 划分后的信息熵
#         infoGain = baseInfoEnt - newInfoEnt
#         if infoGain > bestInfoGain:
#             baseInfoEnt = infoGain
#             bestFeature = i
#
#     return bestFeature
#
#
#
#
#
# def createTree(x:"Feature set",y:"result set"):
#     y1 =list(y)
#     # print(y1)
#     if y1.count(y1[0]) == len(y1):
#
#         return y1[0]
#     if len(x[0]) == 1:
#         return majorityCnt(y1)
#
#     bestFeature = choose_best_feature_to_split(x,y)
#     print(bestFeature,607,y)
#
#     myTree = {y[bestFeature]: {}}
#
#     featValues = [row[bestFeature] for row in x]
#
#     unique_value_list = list(set(featValues))
#
#     for unique_value in unique_value_list:
#         newdataSet,newResultSet = split_dataSet(x,y,bestFeature,unique_value)
#         myTree[y[bestFeature]][unique_value] = createTree(newdataSet,newResultSet)
#     # return myTree
# res = createTree(train_x, train_y)
# print(res)
# import matplotlib.pyplot as plt
# # 解决中文问题
# from matplotlib.font_manager import FontProperties
#
# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
#
# decisionNode = dict(boxstyle='sawtooth', fc="0.8")
# leafNode = dict(boxstyle='round4', fc='0.8')
# arrow_args = dict(arrowstyle='<-')
#
#
#
# def createPlot():
#
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     plt.subplot(111, frameon=False)
#
#
#     plt.annotate("决策节点", xy=(0.1, 0.5), xycoords='axes fraction', xytext=(0.5, 0.1), textcoords='axes fraction',
#                             va="center", bbox=decisionNode, arrowprops=arrow_args, fontproperties=font)
#     plt.annotate("叶节点", xy=(0.3, 0.8), xycoords='axes fraction', xytext=(0.8, 0.1),
#                             textcoords='axes fraction',
#                             va="center", bbox=leafNode, arrowprops=arrow_args, fontproperties=font)
#
#     plt.show()
#
#
# createPlot()
# def getNumLeafs(myTree):
#     numleafs=0
#     firstStr=myTree.keys()[0]
#     secondDict=myTree[firstStr]
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=="dict":
#             numleafs+=getNumLeafs(secondDict[key])
# #         else:
#             numleafs+=1
#     return numleafs
# def getTreeDepth(myTree):
#     maxDepth=0
#     firstStr=myTree.keys()[0]
#     secondDict=myTree[firstStr]
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__=="dict":
#             thisDepth=1+getTreeDepth(secondDict[key])
#         else:
#             thisDepth=1
#         if thisDepth>maxDepth:
#             maxDepth=thisDepth
#     return maxDepth
# getNumLeafs(res)
# getTreeDepth(res)

# s = {0.0: 'setosa', 1.0: 'versicolor', 2.0: 'virginica'}
# s1 = [2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0]
# [s[i] for i in s1]

def read(path):
    with open(path,"r") as fp:
        iris = [inst.strip("\n").split(",") for inst in fp.readlines()]
        return iris


''' 将第i个特征和类标签组合起来
 如:
    [
      [0.2,'Iris-setosa'],
      [0.2,'Iris-setosa'],
                ...
                          ]'''


def split(instances, i):
    log = []
    for line in instances:
        log.append([line[i], line[4]])
    return log


''' 统计每个属性值所具有的实例数量
 [['4.3', 'Iris-setosa', 1], ['4.4', 'Iris-setosa', 3],...]'''


def count(log):
    log_cnt = []
    # 以第0列进行排序的 升序排序
    log.sort(key=lambda attr: attr[0])
    i = 0
    while i < len(log):
        cnt = log.count(log[i])
        record = log[i][:]
        record.append(cnt)
        log_cnt.append(record)
        i += cnt
    return log_cnt


''' log_cnt  是形如： ['4.4', 'Iris-setosa', 3]
    的统计对于某个属性值，对于三个类所含有的数量
    返回结果形如：{4.4:[0,1,3],...}
    属性值为4.4的对于三个类的实例数量分别是：0、1、3 '''


def build(log_cnt):
    log_dict = {}
    for record in log_cnt:
        if record[0] not in log_dict.keys():
            log_dict[record[0]] = [0, 0, 0]
        if record[1] == 'setosa':
            print(log_dict[record[0]])
            log_dict[record[0]][0] = record[2]
        elif record[1] == 'versicolor':
            log_dict[record[0]][1] = record[2]
        elif record[1] == 'virginica':
            log_dict[record[0]][2] = record[2]
        else:
            raise TypeError('Data Exception')
    # print(log_dict,740)
    log_truple = sorted(log_dict.items())
    return log_truple


def collect(instances, i):
    log = split(instances, i)
    log_cnt = count(log)
    log_tuple = build(log_cnt)
    return log_tuple


def combine(a, b):
    """''  a=('4.4', [3, 1, 0]), b=('4.5', [1, 0, 2])
         combine(a,b)=('4.4', [4, 1, 2])  """
    c = a[:]
    for i in range(len(a[1])):
        c[1][i] += b[1][i]
    return c


def chi2(a):
    """计算两个区间的卡方值"""
    m = len(a)
    k = len(a[0])
    r = []
    '''第i个区间的实例数'''
    for i in range(m):
        sum = 0
        for j in range(k):
            sum += a[i][j]
        r.append(sum)
    c = []
    '''第j个类的实例数'''
    for j in range(k):
        sum = 0
        for i in range(m):
            sum += a[i][j]
        c.append(sum)
    n = 0
    '''总的实例数'''
    for ele in c:
        n += ele
    res = 0.0
    for i in range(m):
        for j in range(k):
            Eij = 1.0 * r[i] * c[j] / n
            if Eij != 0:
                res = 1.0 * res + 1.0 * (a[i][j] - Eij) ** 2 / Eij
    return res


'''ChiMerge 算法'''
'''下面的程序可以看出，合并一个区间之后相邻区间的卡方值进行了重新计算，而原作者论文中是计算一次后根据大小直接进行合并的
下面在合并时候只是根据相邻最小的卡方值进行合并的，这个在实际操作中还是比较好的
'''


def chimerge(log_tuple, max_interval):
    num_interval = len(log_tuple)
    while num_interval > max_interval:
        num_pair = num_interval - 1
        chi_values = []
        ''' 计算相邻区间的卡方值'''
        for i in range(num_pair):
            arr = [log_tuple[i][1], log_tuple[i + 1][1]]
            chi_values.append(chi2(arr))
        min_chi = min(chi_values)
        for i in range(num_pair - 1, -1, -1):
            if chi_values[i] == min_chi:
                log_tuple[i] = combine(log_tuple[i], log_tuple[i + 1])
                log_tuple[i + 1] = 'Merged'
        while 'Merged' in log_tuple:
            log_tuple.remove('Merged')
        num_interval = len(log_tuple)
    split_points = [record[0] for record in log_tuple]
    return split_points


def discrete(path):
    instances = read(path)
    max_interval = 6
    num_log = 4
    for i in range(num_log):
        log_tuple = collect(instances, i)
        split_points = chimerge(log_tuple, max_interval)
        # print(split_points,1567)

iris = discrete("sklearn_data/iris.data")