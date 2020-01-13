import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


def func1(x):
    y = np.exp(np.cos(x/2)+x/3-np.tanh(x))
    return y


def func2(x):
    y = 2*x + np.cos(x) + 2
    return y


def func3(x):
    return np.sin(x)+np.cos(3 * x)


def get_data(nums, func):
    x = np.random.uniform(-10, 10, nums).reshape([nums, 1])
    return x, func(x)


def get_test_data(nums):
    x = np.random.uniform(-10, 10, nums).reshape([nums, 1])
    return x


def generate_data():
    X1_cnt = 2000
    P1_cnt = 10
    P2_cnt = 10
    P3_cnt = 10
    sample_cnt = 300000
    for i in sample_cnt:
        X1 = np
    pass



def compare_plot(x, y, func):
    x = np.array(x).reshape([-1, 1])
    y = np.array(y).reshape([-1, 1])
    plt.scatter(x, y, c='r')
    x_min = np.array(x).min()
    x_max = np.array(x).max()
    x_n = np.arange(x_min, x_max, 0.01)
    y_n = func(x_n)
    plt.plot(x_n, y_n, c='b')
    plt.show()





def main():
    x, y = get_data(100, func2)
    compare_plot(x, y, func2)


if __name__ == "__main__":
    main()