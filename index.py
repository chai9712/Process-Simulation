import tensorflow as tf
import numpy as np
from data import *
import model
import matplotlib.pyplot as plt

learning_rate = 0.001
train_iter = 5500
train_num = 500
test_num = 100
save_step = 20


def main():
    train_x, train_y = get_data(train_num, func1)

    X = tf.placeholder(tf.float32, [None, 1], name="input")
    Y = tf.placeholder(tf.float32, [None, 1], name="output")
    pre_ht = None
    pre_ct = None
    for i in range(2):
        predict, ht, ct = model.cell(X, pre_ht, pre_ct, "cell%d" % i, reuse=False)
        pre_ht = ht
        pre_ct = ct
    #predict, _, _ = model.cell(X, None, None, "cell", tf.AUTO_REUSE)
    loss = tf.reduce_mean(tf.square(Y-predict))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    epoch = tf.Variable(0, name="epoch", trainable=False)
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        ckpt = tf.train.get_checkpoint_state('checkpoint/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        start = sess.run(epoch)
        for ep in range(start, train_iter):
            _, var_loss = sess.run([optimizer, loss], feed_dict={X: train_x, Y: train_y} )
            print(var_loss)
            if ep!=0 and ep % save_step == 0:
                saver.save(sess, "checkpoint/model_%d.ckpt" % ep)
            sess.run(epoch.assign(ep + 1))


if __name__ == "__main__":
    main()