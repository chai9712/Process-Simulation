# N = 5
#
# def func1(x):
#     y = np.exp(np.cos(x/2)+x/3-np.tanh(x))
#     return y
#
#
# def func2(x):
#     y = 2*x + np.cos(x) + 2
#     return y
#
#
# def func3(x):
#     return np.sin(x)+np.cos(3 * x)
#
#
# def get_data(nums, func):
#     x = np.random.uniform(-10, 10, nums).reshape([nums, 1])
#     return x, func(x)
#
#
# def get_test_data(nums):
#     x = np.random.uniform(-10, 10, nums).reshape([nums, 1])
#     return x
#
#
# def func_luzi01(X, P):
#     W11 = np.random.random((4, N)) * 8 - 4
#     W11p = np.random.random((3, N)) * 8 - 4
#     W12 = np.random.random((4, N)) * 8 - 4
#     W12p = np.random.random((3, N)) * 8 - 4
#     W13 = np.random.random((4, N)) * 8 - 4
#     W13p = np.random.random((3, N)) * 8 - 4
#     if -10 <= P[-1] and P[-1] <= 10:
#         Y = np.matmul(X - 0.001, W11) * np.tanh(np.matmul(P, W11p))
#     elif 10 < P[-1] and P[-1] <20:
#         Y = np.matmul((0.000002 * X**2 + 0.0002 * X - 0.001), W12) * np.tanh(np.matmul(P, W12p))
#     else:
#         Y = np.matmul((0.00000002 * X**3 + 0.000002 * X**2 + 0.0002 * X - 0.001), W13) * np.tanh(np.matmul(P, W13p))
#     return Y
#
#
# def func_luzi02(X, P):
#     W21 = np.random.random((N, N)) * 8 - 4
#     W21p = np.random.random((3, N)) * 8 - 4
#     W22 = np.random.random((N, N)) * 8 - 4
#     W22p = np.random.random((3, N)) * 8 - 4
#     W23 = np.random.random((N, N)) * 8 - 4
#     W23p = np.random.random((3, N)) * 8 - 4
#     if -10 <= P[-1] and P[-1] <= 10:
#         Y = np.matmul(X - 0.001, W21) * np.tanh(np.matmul(P, W21p))
#     elif 10 < P[-1] and P[-1] < 20:
#         Y = np.matmul((0.000002 * X**2 + 0.0002 * X - 0.001), W22) * np.tanh(np.matmul(P, W22p))
#     else:
#         Y = np.matmul((0.00000002 * X**3 + 0.000002 * X**2 + 0.0002 * X - 0.001), W23) * np.tanh(np.matmul(P, W23p))
#     return Y
#
#
# def func_luzi03(X, P):
#     W31 = np.random.random((N, N)) * 8 - 4
#     W31p = np.random.random((4, N)) * 8 - 4
#     W32 = np.random.random((N, N)) * 8 - 4
#     W32p = np.random.random((4, N)) * 8 - 4
#     W33 = np.random.random((N, N)) * 8 - 4
#     W33p = np.random.random((4, N)) * 8 - 4
#     if -10 <= P[-1] and P[-1] <= 10:
#         Y = np.matmul(X - 0.001, W31) * np.tanh(np.matmul(P, W31p))
#     elif 10 < P[-1] and P[-1] < 20:
#         Y = np.matmul((0.000002 * X**2 + 0.0002 * X - 0.001), W32) * np.tanh(np.matmul(P, W32p))
#     else:
#         Y = np.matmul((0.00000002 * X**3 + 0.000002 * X**2 + 0.0002 * X - 0.001), W33) * np.tanh(np.matmul(P, W33p))
#     return Y
#
#
#
# def generate_data():
#     sample_cnt = 20000
#     all_variable = []
#     all_output = []
#     for i in range(sample_cnt):
#         X1 = np.random.rand(4)*200
#         P1 = np.random.rand(3)*40 - 10
#         P2 = np.random.rand(3)*40 - 10
#         P3 = np.random.rand(4)*40 - 10
#         all_variable.append(np.concatenate((X1, P1, P2, P3), axis=0).tolist())
#         Y1 = func_luzi01(X1, P1)
#         Y2 = func_luzi02(Y1, P2)
#         Y = func_luzi03(Y2, P3)
#         all_output.append(np.concatenate((Y1, Y2, Y), axis=0).tolist())
#     np.save('data/input.npy', all_variable)
#     np.save('data/output.npy', all_output)
#     print(np.shape(np.array(all_output)))
#
#
# def get_batch(input, label, batch_size=50, shuffle=True):
#     list = np.arange(input.shape[0])
#     if shuffle==True:
#         np.random.shuffle(list)
#         start = 0
#         for step in range(0, math.ceil(input.shape[0]/batch_size)):
#             one_batch = list[start: start + batch_size]
#             start += batch_size
#             out_in = input[one_batch]
#             out_la = label[one_batch]
#             yield out_in, out_la
#     else:
#         start = 0
#         for step in range(0, math.ceil(input.shape[0] / batch_size)):
#             one_batch = list[start: start + batch_size]
#             start += batch_size
#             out_in = input[one_batch]
#             out_la = label[one_batch]
#             yield out_in, out_la
#
#
#
#
# def compare_plot(x, y, func):
#     x = np.array(x).reshape([-1, 1])
#     y = np.array(y).reshape([-1, 1])
#     plt.scatter(x, y, c='r')
#     x_min = np.array(x).min()
#     x_max = np.array(x).max()
#     x_n = np.arange(x_min, x_max, 0.01)
#     y_n = func(x_n)
#     plt.plot(x_n, y_n, c='b')
#     plt.show()

import tensorflow as tf
import numpy as np
import math
import random
import matplotlib.pyplot as plt

def dataPreprocess(path):
    allData = np.loadtxt(path, delimiter=",")
    assert allData.shape==(allData.shape[0], 60)
    # 数据标准化
    dataMean = allData.mean(0)
    assert (dataMean.shape == (60,))
    dataStd = allData.std(0)
    assert (dataStd.shape == (60,))
    allData[:, 0:50] = (allData[:, 0:50] - dataMean[0:50]) / (dataStd[0:50] + 0.000001)
    # 数据划分
    trainLen = int(allData.shape[0] * 0.8)
    trainData = allData[0:trainLen, :]
    testData = allData[trainLen:allData.shape[0], :]
    return trainData, testData, dataMean, dataStd


def generateBatchData(traindata, batchsize=20, shuffle=True):
    list = np.arange(traindata.shape[0])
    if shuffle==True:
        np.random.shuffle(list)
        start = 0
        for step in range(0, math.ceil(traindata.shape[0]/batchsize)):
            one_batch = list[start: start + batchsize]
            start += batchsize
            out_in = traindata[one_batch, 0:50]
            out_la = traindata[one_batch, 50:60]
            yield out_in, out_la
    else:
        start = 0
        for step in range(0, math.ceil(input.shape[0] / batch_size)):
            one_batch = list[start: start + batchsize]
            start += batchsize
            out_in = traindata[one_batch, 0:50]
            out_la = traindata[one_batch, 50:60]
            yield out_in, out_la



def main():
    trainData, testData, _, _ = dataPreprocess("data/experiment_data.csv")
    for train_X, train_Y in generateBatchData(trainData):
        print(train_X.shape, train_Y.shape)




if __name__ == "__main__":
    main()