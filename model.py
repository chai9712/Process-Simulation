import tensorflow as tf
import numpy as np

hidden_size = 100
output_size = 1

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


def cell(input, pre_ct, pre_ht, name, reuse):
    shape_in = input.get_shape().as_list() # 1 in
    l_in = shape_in[-1]
    if pre_ht == None:
        with tf.variable_scope(name, reuse=reuse):
            weight = get_weight_variable([l_in, hidden_size], "ht_start_weight")
            bias = get_bias_variable([hidden_size], "ht_start_bias")
            pre_ht = tf.matmul(input, weight) + bias # 1 h
    if pre_ct == None:
        with tf.variable_scope(name, reuse=reuse):
            weight = get_weight_variable([l_in, hidden_size], "ct_start_weight")
            bias = get_bias_variable([hidden_size], "ct_start_bias")
            pre_ct = tf.matmul(input, weight) + bias # 1 h
    with tf.variable_scope(name, reuse=reuse):
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
        print(output.get_shape())
        return output, ht, ct




def create_model():

    pass


def main():
    weight = weight_variable([5, 1, 2])
    weight2 = weight_variable([5, 1, 3])
    meg = tf.concat([weight, weight2], axis=2)
    print(meg.get_shape())



if __name__ =='__main__':
    main()