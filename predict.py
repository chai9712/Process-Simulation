import tensorflow as tf
import numpy as np
from data import *
import model
import matplotlib.pyplot as plt
test_num = 100

def main():
    test_x, test_y = get_data(test_num, func1)
    X = tf.placeholder(tf.float32, [None, 1], name="input")
    Y = tf.placeholder(tf.float32, [None, 1], name="output")
    predict, _, _ = model.cell(X, None, None, "cell", tf.AUTO_REUSE)
    loss = tf.reduce_mean(tf.square(Y-predict))
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        ckpt = tf.train.get_checkpoint_state('checkpoint/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        var_pred, var_loss = sess.run([predict, loss], feed_dict={X: test_x, Y:test_y})
    print(var_loss)
    compare_plot(test_x, var_pred, func1)


if __name__ == "__main__":
    main()