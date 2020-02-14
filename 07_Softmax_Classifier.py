import numpy as np
import tensorflow as tf

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])

W = tf.Variable(tf.random_normal([16,7]), name = 'weight')
b = tf.Variable(tf.random_normal([7]), name = 'bias')

#logits = tf.matmul(X,W)+b
#hypothesis = tf.nn.softmax(logits)
# cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)
# cost = tf.reduce_mean(cost_i)

hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y_one_hot*tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(2000):
		sess.run(optimizer, feed_dict = {X: x_data, Y: y_data})
		if step % 100 == 0:
			loss, acc = sess.run([cost, accuracy], feed_dict = {X: x_data, Y: y_data})
			print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
	pred = sess.run(prediction, feed_dict={X: x_data})
	for p, y in zip(pred, y_data.flatten()):
		print("[{}]Prediction: {} True Y: {}".format(p == int(y), p, int(y)))