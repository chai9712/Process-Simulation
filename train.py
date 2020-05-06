import tensorflow as tf
import numpy as np
from measurement import *
from data import *
from LSTM import *

learning_rate = 1
train_iters = 5000
save_step = 1

trainData, testData, dataMean, dataStd = dataPreprocess("data/experiment_data.csv")
graph = tf.get_default_graph()
pred = LSTMModel01()
groundTruth = graph.get_tensor_by_name("groundtruth:0")
loss = tf.reduce_mean(tf.square(pred-groundTruth))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

curEpoch = tf.Variable(0, name="epoch", trainable=False)
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    ckpt = tf.train.get_checkpoint_state('checkpoint/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    pred_val = sess.run(pred, feed_dict={graph.get_tensor_by_name("allinput:0"): testData[:, 0:50], groundTruth: testData[:, 50:60]})
    #print(dataStd[50:60], dataMean[50:60])
    #pred_val = pred_val * (dataStd[50:60] + 0.000001) + dataMean[50:60]
    #test_label = testData[:, 50:60] * (dataStd[50:60] + 0.000001) + dataMean[50:60]
    test_label = testData[:, 50:60]
    print(pred_val[0])
    print(test_label[0])
    R2 = getRSquare(test_label, pred_val, dataMean[50:60])
    R2adj = getAdjustedRSquare(test_label, pred_val, dataMean[50:60])
    print(R2, R2adj)
    # start = sess.run(curEpoch)
    # for i in range(start, train_iters):
    #     for train_X, train_Y in generateBatchData(trainData, 90):
    #         _, loss_val = sess.run([optimizer, loss], feed_dict={graph.get_tensor_by_name("allinput:0"): train_X, groundTruth: train_Y})
    #     loss_val_test = sess.run(loss, feed_dict={graph.get_tensor_by_name("allinput:0"): testData[:, 0:50], groundTruth: testData[:, 50:60]})
    #     if i % save_step == 0:
    #         saver.save(sess, "checkpoint/model_%d.ckpt" % i)
    #         #print("train_loss=", loss_val, "test_loss=", loss_val_test)
    #         pred_val = sess.run([pred], feed_dict={graph.get_tensor_by_name("allinput:0"): testData[:, 0:50],
    #                                                groundTruth: testData[:, 50:60]})
    #         test_label = testData[:, 50:60]
    #         R2 = getRSquare(test_label, pred_val, dataMean[50:60])
    #         R2adj = getAdjustedRSquare(test_label, pred_val, dataMean[50:60])
    #         print(i, ":", "R2=", R2, "R2adj=", R2adj)
    #     sess.run(curEpoch.assign(i + 1))



