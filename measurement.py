import numpy as np

def getRSquare(Y_actual, Y_predict, Y_mean):
    '''
    :param Y_actual: m*n
    :param Y_predict: m*n
    :param Y_mean: n
    :return:
    '''
    Y_predict = np.array(Y_predict)
    Y_actual = np.array(Y_actual)
    Y_mean = np.array(Y_mean)
    d1 = (Y_predict - Y_actual) ** 2
    d2 = (Y_actual - Y_mean) ** 2
    output = 1 - np.sum(d1)/np.sum(d2)
    return output


def getAdjustedRSquare(Y_actual, Y_predict, Y_mean):
    Y_predict = np.array(Y_predict)
    Y_actual = np.array(Y_actual)
    Y_mean = np.array(Y_mean)
    n = Y_actual.shape[1]
    m = Y_actual.shape[0]
    R2 = getRSquare(Y_actual, Y_predict, Y_mean)
    output = 1 - (1-R2)*(m-1)/(m-n-1)
    return output
