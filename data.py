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
    sample_cnt = 1
    for i in range(sample_cnt):
        X1 = np.random.rand(1, 4)*200

        P1 = np.random.rand(1, 3)*40 - 10
        P2 = np.random.rand(1, 3)*40 - 10
        P3 = np.random.rand(1, 4)*40 - 10
        print(np.concatenate((X1, P1, P2, P3), axis=1))
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
    generate_data()


if __name__ == "__main__":
    main()