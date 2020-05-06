import tensorflow as tf
from functools import reduce
from operator import mul
output_size = 10

def weight_variable(shape, name=None, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name, dtype=tf.float32, trainable=trainable)


def bias_variable(shape, name=None, trainable=True):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name, dtype=tf.float32, trainable=trainable)


def get_weight_variable(shape, name=None, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.get_variable(name=name, dtype=tf.float32, initializer=initial, trainable=trainable)


def get_bias_variable(shape, name=None, trainable=True):
    initial = tf.constant(0.01, shape=shape)
    return tf.get_variable(name=name, dtype=tf.float32, initializer=initial, trainable=trainable)


def getNumParams():
    num = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num += reduce(mul, [dim.value for dim in shape], 1)
    return num



def cell(input, pre_ct=None, pre_ht=None, name=None, reuse=False, origin_input=None, hidden_size=20):
    shape_in = input.get_shape().as_list() # 1 in
    l_in = shape_in[-1]
    with tf.variable_scope(name, reuse=reuse):
        if origin_input != None:
            shape_in_origin = origin_input.get_shape().as_list()
            l_in_origin = shape_in_origin[-1]
        if pre_ht == None:
            with tf.variable_scope(name, reuse=reuse):
                weight = get_weight_variable([l_in_origin, hidden_size], "ht_start_weight")
                bias = get_bias_variable([hidden_size], "ht_start_bias")
                pre_ht = tf.matmul(origin_input, weight) + bias  # 1 h
        if pre_ct == None:
            with tf.variable_scope(name, reuse=reuse):
                weight = get_weight_variable([l_in_origin, hidden_size], "ct_start_weight")
                bias = get_bias_variable([hidden_size], "ct_start_bias")
                pre_ct = tf.matmul(origin_input, weight) + bias  # 1 h
        combine = tf.concat([input, pre_ht], axis=1) # 1 h+in
        weight = get_weight_variable([hidden_size + l_in, hidden_size], "in_weight")
        bias = get_bias_variable([hidden_size], "in_bias")
        in_gate = tf.sigmoid(tf.matmul(combine, weight) + bias) # 1 h

        weight = get_weight_variable([hidden_size + l_in, hidden_size], "ou_weight")
        bias = get_bias_variable([hidden_size], "ou_bias")
        ou_gate = tf.sigmoid(tf.matmul(combine, weight) + bias) # 1 h

        weight = get_weight_variable([hidden_size + l_in, hidden_size], "fo_weight")
        bias = get_bias_variable([hidden_size], "fo_bias")
        fo_gate = tf.sigmoid(tf.matmul(combine, weight) + bias)

        weight = get_weight_variable([l_in + hidden_size, hidden_size], "ct_weight")
        bias = get_bias_variable([hidden_size], "ct_bias")
        ct_ = tf.tanh(tf.matmul(combine, weight) + bias)

        ct = fo_gate * pre_ct + in_gate * ct_
        ht = ou_gate * tf.tanh(ct) # 1 h

        weight = get_weight_variable([hidden_size, output_size], "output_weight")
        bias = get_bias_variable([output_size], "output_bias")
        output = tf.matmul(ht, weight) + bias
        #output = tf.tanh(output)
        #print(output.get_shape())
        return output, ht, ct


def LSTMModel01():
    # 所有权重共享
    allInput = tf.placeholder(tf.float32, [None, 50], name="allinput")
    hidden_size = 8
    origin_input = allInput[:, 0:4]

    input = allInput[:, 4:8]
    output, ht, ct = cell(input=input, name="cell00", reuse=False, origin_input=origin_input, hidden_size=hidden_size)
    for i in range(1, 15):
        input = allInput[:, i*3+5:i*3+8]
        output, ht, ct = cell(input=input, pre_ct=ct, pre_ht=ht, name="cell", reuse=tf.AUTO_REUSE, hidden_size=hidden_size)
    groundTruth = tf.placeholder(tf.float32, [None, 10], "groundtruth")
    print("model01, params num=", getNumParams())
    return output

def LSTMModel02():
    # 部分权重共享
    hidden_size = 8
    allInput = tf.placeholder(tf.float32, [None, 50], name="allinput")
    origin_input = allInput[:, 0:4]
    input = allInput[:, 4:8]
    output, ht, ct = cell(input=input, name="cell00", reuse=False, origin_input=origin_input, hidden_size=hidden_size)
    # 2-9牵伸
    for i in range(1, 9):
        input = allInput[:, i*3+5:i*3+8]
        output, ht, ct = cell(input=input, pre_ct=ct, pre_ht=ht, name="qianshen", reuse=tf.AUTO_REUSE, hidden_size=hidden_size)
    # 10油剂
    input = allInput[:, 32:35]
    output, ht, ct = cell(input=input, pre_ct=ct, pre_ht=ht, name="youji", reuse=tf.AUTO_REUSE, hidden_size=hidden_size)
    # 11致密化
    input = allInput[:, 35:38]
    output, ht, ct = cell(input=input, pre_ct=ct, pre_ht=ht, name="zhimihua", reuse=False, hidden_size=hidden_size)
    # 12 13蒸汽
    for i in range(11, 13):
        input = allInput[:, 3*i+5:3*i+8]
        output, ht, ct = cell(input=input, pre_ct=ct, pre_ht=ht, name="zhengqi", reuse=tf.AUTO_REUSE, hidden_size=hidden_size)
    # 14油剂2

    input = allInput[:, 44:47]
    output, ht, ct = cell(input=input, pre_ct=ct, pre_ht=ht, name="youji", reuse=tf.AUTO_REUSE, hidden_size=hidden_size)
    # 15干燥
    input = allInput[:, 47:50]
    output, ht, ct = cell(input=input, pre_ct=ct, pre_ht=ht, name="ganzao", reuse=False, hidden_size=hidden_size)
    groundTruth = tf.placeholder(tf.float32, [None, 10], "groundtruth")
    print("model01, params num=", getNumParams())
    return output


def main():
    LSTMModel01()


if __name__ == "__main__":
    main()