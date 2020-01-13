import tensorflow as tf
import model
learning_rate = 0.001
train_iter = 5500
train_num = 500
test_num = 100
save_step = 20

def main():
    pre_ht = None
    pre_ct = None
    for i in range(10):
        input = tf.placeholder(tf.float32, [None, 1], name='input%d' % i)
        if i == 9:
            pred, ht, ct = model.cell(input, pre_ht, pre_ct, "cell", reuse=tf.AUTO_REUSE)
        else:
            _, ht, ct = model.cell(input, pre_ht, pre_ct, "cell", reuse=tf.AUTO_REUSE)
        pre_ht = ht
        pre_ct = ct
    Y = tf.placeholder(tf.float32, [None, 1], name='output')
    writer = tf.summary.FileWriter('./log')
    loss = tf.reduce_mean(tf.square(pred-Y))
    optimizer = tf.train.Adamoption()
    with tf.Session() as sess:
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

    writer.close()



if __name__ == '__main__':
    main()