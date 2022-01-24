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
import pandas as pd
left = pd.DataFrame({
         'student_id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
         'student_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung', 'Billy', 'Brian', 'Bran', 'Bryce', 'Betty', 'Emma', 'Marry', 'Allen', 'Jean', 'Rose', 'David', 'Tom', 'Jack', 'Daniel', 'Andrew'],
         'class_id':[1,1,1,2,2,2,3,3,3,4,1,1,1,2,2,2,3,3,3,2],
         'gender':['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
         'age':[20,21,22,20,21,22,23,20,21,22,20,21,22,23,20,21,22,20,21,22],
         'score':[98,74,67,38,65,29,32,34,85,64,52,38,26,89,68,46,32,78,79,87]})
right = pd.DataFrame(
         {'class_id':[1,2,3,5],
         'class_name': ['ClassA', 'ClassB', 'ClassC', 'ClassE']})
# 合并两个DataFrame
data = pd.merge(left,right)
print(data.pivot_table(index=['class_id', 'gender'], values=['score'],
                       columns=['age']))
