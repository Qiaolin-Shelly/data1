# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from pandas import Series
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import statsmodels.api as sm

# 读取时间数据的格式化
#def parser(x):
#   return x

batch_size = 200
test_size = 10079
epoch_size = 100
neurons_num = 50
memory_value = 4
parameter_num = 2
# 转换成有监督数据
def timeseries_to_supervised(data, lag=2):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(lag, 0, -1)]  # 数据滑动一格，作为input，df原数据为output
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# 转换成差分数据
def difference(dataset, interval=1):
    diff1 = list()
    diff2 = list()
    for i in range(interval, len(dataset)):
        #value1 = dataset[i] - dataset[i - interval]
        value1 = dataset[i][0] - dataset[i - interval][0]
        diff1.append(value1)
        value2 = dataset[i][1] - dataset[i - interval][1]
        diff2.append(value2)
    diff = list()
    diff.append(diff1)
    diff.append(diff2)
    diff = numpy.array(diff)
    diff = diff.T
    return diff


# 逆差分
def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]


# 缩放
def scale(train, test):
    # 根据训练数据建立缩放器
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # 转换train data
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # 转换test data
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# 逆缩放
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# fit LSTM来训练数据
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    # 添加LSTM层
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))  # 输出层1个node
    # 编译，损失函数mse+优化算法adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # 按照batch_size，一次读取batch_size个数据
        model.fit(X, y, epochs=1, batch_size = batch_size, verbose=0, shuffle=False)
        model.reset_states()
        print("当前计算次数："+str(i))
    return model


# 1步长预测
def forcast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# 加载数据
series = read_csv('data_set/UDP_original.csv', header=0, parse_dates=[0], index_col=0, squeeze=True) # TCP_ceil_del

# 让数据变成稳定的
raw_values = series.values
diff_values = difference(raw_values, 1)#转换成差分数据

# 把稳定的数据变成有监督数据
#supervised = timeseries_to_supervised(diff_values, 1)
supervised = timeseries_to_supervised(diff_values, memory_value) # 最后一个元素是记忆性
supervised_values = supervised.values
supervised_values = supervised_values[:, :-1]
supervised_values = supervised_values[memory_value:, :]
# 数据拆分：训练数据、测试数据，前24行是训练集，后12行是测试集
train, test = supervised_values[0:-test_size], supervised_values[-test_size:]

# train_tt = train.reshape(220, 1, 900)
# train_tt = train_tt[: :2]
# tt3 = train_tt.reshape(11000,1,9)
# tt3 = tt3.reshape(11000,9)
# train = tt3
# 数据缩放
scaler, train_scaled, test_scaled = scale(train, test)

# fit 模型
lstm_model = fit_lstm(train_scaled, batch_size, epoch_size, neurons_num)  # 训练数据，batch_size，epoche次数, 神经元个数
# 预测
#a1=train_scaled[:, 0:1]
#a2=train_scaled[:, 0:-1]
train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), 1, parameter_num*memory_value)#训练数据集转换为可输入的矩阵 # 最后一个元素是记忆性（如果是单一元素预测），否则是记忆性*元素个数
lstm_model.predict(train_reshaped, batch_size = batch_size)#用模型对训练数据矩阵进行预测

batch_size1 = 1
new_model = Sequential()
# 添加LSTM层
new_model.add(LSTM(neurons_num, batch_input_shape=(batch_size1, 1, parameter_num*memory_value), stateful=True)) # X.shape[1], X.shape[2]
new_model.add(Dense(1))  # 输出层1个node
old_weights = lstm_model.get_weights()
new_model.set_weights(old_weights)
new_model.compile(loss='mean_squared_error', optimizer='adam')


# 测试数据的前向验证，实验发现，如果训练次数很少的话，模型回简单的把数据后移，以昨天的数据作为今天的预测值，当训练次数足够多的时候
# 才会体现出来训练结果
predictions = list()
for i in range(len(test_scaled)):#根据测试数据进行预测，取测试数据的一个数值作为输入，计算出下一个预测值，以此类推
    # 1步长预测
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forcast_lstm(new_model, 1, X)
    # 逆缩放
    yhat = invert_scale(scaler, X, yhat)
    # 逆差分
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    #predictions.append(yhat)
    predictions.append(yhat[0])
    #expected = raw_values[len(train) + i + 1]
    expected = raw_values[len(train) + i + 1][0]
    #print('Moth=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
    print('Moth=%d, Predicted=%f, Expected=%f' % (i + 1, yhat[0], expected))

# 性能报告
#rmse = sqrt(mean_squared_error(raw_values[-test_size:], predictions))
rmse = sqrt(mean_squared_error(raw_values[-test_size:,0], predictions))
print('Test RMSE:%.3f' % rmse)
# 绘图
pyplot.rcParams["font.sans-serif"]=["SimHei"]
pyplot.rcParams["font.family"]="sans-serif"
pyplot.rcParams['axes.unicode_minus'] =False
#pyplot.plot(raw_values[-test_size:])
pyplot.plot(raw_values[-test_size:,0])
pyplot.plot(predictions)
pyplot.show()



# 绘制真实数据包到达时间差
raw_values_pre = raw_values[-test_size:] # 测试数据集 真实到达时间
diff_raw_values_pre = difference(raw_values_pre, 1) # 转换成差分数据
diff_raw_values_pre = diff_raw_values_pre[:,0]
fig, ax = pyplot.subplots(1, 1)
ecdf = sm.distributions.ECDF(diff_raw_values_pre)
x_x = numpy.linspace(min(diff_raw_values_pre), max(diff_raw_values_pre), 10000)
y_y = ecdf(x_x)
l2, = pyplot.plot(x_x*1000, y_y)
pyplot.xlim((0, 100))

# # 对预测值按照0.0005秒聚合
# predictions_ceiling = numpy.ceil(numpy.array(predictions)/0.0005)*0.0005 # 聚合后的预测包到达时间：将预测的包到达时间，ceiling 0.0005s
# pre_diff = raw_values_pre - predictions_ceiling # 预测偏差 = 真实包到达时间 - 预测包到达时间
pre_diff = raw_values_pre[:,0] - predictions
pre_diff_pos = [x for x in pre_diff if x > 0]
pre_diff_neg = [x for x in pre_diff if x <= 0]


# 预测滞后的CDF
# import statsmodels.api as sm
# fig, ax = pyplot.subplots(1, 1)
ecdf = sm.distributions.ECDF(pre_diff_pos)
x_x = numpy.linspace(min(pre_diff_pos), max(pre_diff_pos), 10000)
y_y = ecdf(x_x)
l1, = pyplot.plot(x_x*1000, y_y)
pyplot.xlim((0, 100))
pyplot.legend(handles=[l1,l2],labels=['预测到达时间差','真实到达时间差'],loc='best')

# 预测超前的CDF
ecdf = sm.distributions.ECDF(pre_diff_neg)
x_x = numpy.linspace(min(pre_diff_neg), max(pre_diff_neg), 10000)
y_y = ecdf(x_x)
pyplot.step(x_x*1000, y_y)
#
#
#
#
# numpy.save('raw_values_pre_100_30', raw_values_pre)
# numpy.save('predictions_ceiling_100_30', predictions_ceiling)


# x_axis_data = range(0,20000)
# fig, ax = pyplot.subplots(1, 1)
# ax.plot(x_axis_data, pre_diff)
# axins = ax.inset_axes((0.1, 0.5, 0.4, 0.3))
# axins.plot(x_axis_data, pre_diff)
#
# from matplotlib.patches import ConnectionPatch
# zone_left = 16990
# zone_right = 17015
#
# # 坐标轴的扩展比例（根据实际数据调整）
# x_ratio = 0  # x轴显示范围的扩展比例
# y_ratio = 0.05  # y轴显示范围的扩展比例
#
# # X轴的显示范围
# xlim0 = x_axis_data[zone_left]-(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
# xlim1 = x_axis_data[zone_right]+(x_axis_data[zone_right]-x_axis_data[zone_left])*x_ratio
#
# # Y轴的显示范围
# y = numpy.hstack((pre_diff[zone_left:zone_right]))
# ylim0 = numpy.min(y)-(numpy.max(y)-numpy.min(y))*y_ratio
# ylim1 = numpy.max(y)+(numpy.max(y)-numpy.min(y))*y_ratio
#
# # 调整子坐标系的显示范围
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)
#
# tx0 = xlim0
# tx1 = xlim1
# ty0 = ylim0
# ty1 = ylim1
# sx = [tx0,tx1,tx1,tx0,tx0]
# sy = [ty0,ty0,ty1,ty1,ty0]
# ax.plot(sx,sy,"black")
#
# # 画两条线
# xy = (xlim0,ylim0)
# xy2 = (xlim1,ylim0)
# con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
#         axesA=axins,axesB=ax)
# axins.add_artist(con)
#
# xy = (xlim0,ylim1)
# xy2 = (xlim1,ylim1)
# con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
#         axesA=axins,axesB=ax)
# axins.add_artist(con)


# # # 绘制CDF曲线
# # import statsmodels.api as sm
# ecdf = sm.distributions.ECDF(pre_diff)
# fig, ax = pyplot.subplots(1, 1)
# x_x = numpy.linspace(min(pre_diff), max(pre_diff))
# y_y = ecdf(x_x)
# pyplot.step(x_x, y_y)
#
# # 绘制直方图
# fig,ax0 = pyplot.subplots(nrows=1,figsize=(8,8))
# ax0.hist(pre_diff,100,density=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)

