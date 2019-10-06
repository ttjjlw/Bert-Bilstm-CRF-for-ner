# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# Date
import tensorflow as tf
import tf_metrics
import numpy as np

y_true = [0, 1,0, 0, 0, 2, 3, 0, 0, 1]
y_pred = [0, 1, 0, 0, 1, 2, 0, 3, 3,1]
pos_indices = [ 1, 2, 3]  # Class 1 is the 'negative' class
num_classes = 4
average = 'micro'
weights=tf.constant([1,2,1,1,1,0,1,1,1,0],dtype=tf.int32)
# Tuple of (value, update_op)
precision = tf_metrics.precision(
    y_true, y_pred, num_classes, pos_indices, average=average,weights=weights)
recall = tf_metrics.recall(
    y_true, y_pred, num_classes, pos_indices, average=average)
f2 = tf_metrics.fbeta(
    y_true, y_pred, num_classes, pos_indices, average=average, beta=2)
f1 = tf_metrics.f1(
    y_true, y_pred, num_classes, pos_indices, average=average)
# acc=tf.metrics.accuracy(y_true,y_pred)
# pre=tf.metrics.precision(y_true,y_pred)
# rec=tf.metrics.recall(y_true,y_pred)

# Run the update op and get the updated value
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    print(sess.run([precision[1],recall[1],f1[1]]))