from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([784, 784]), name='weight')
b = tf.Variable(tf.random_normal([784]), name = 'bais')
layer1 = tf.sigmoid(tf.matmul(X,W)+b)

W1 = tf.Variable(tf.random_normal([784,784]), name='weight1')
b1 = tf.Variable(tf.random_normal([784]), name = 'bias1')
layer2 = tf.sigmoid(tf.matmul(layer1,W1)+b1)

W2 = tf.Variable(tf.random_normal([784,1], name = 'weight2'))
b2 = tf.Variable(tf.random_normal([1], name = 'bais2'))

hypothesis = tf.sigmoid(tf.matmul(layer2,W2)+b2)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)

		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
			avg_cost += c/total_batch

		print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))
	print("Accuracy: ",accuracy.eval(session = sess, feed_dict = {X: mnist.test.images, Y: mnist.test.labels}))