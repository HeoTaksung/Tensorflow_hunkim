from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
X = tf.placeholder(tf.float32, [None,784])
Y = tf.placeholder(tf.float32, [None, 10])
W1 = tf.Variable(tf.random_uniform([784,784], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([784,784], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([784,784], -1.0, 1.0))
W4 = tf.Variable(tf.random_uniform([784,784], -1.0, 1.0))
W5 = tf.Variable(tf.random_uniform([784,10], -1.0, 1.0))

b1 = tf.Variable(tf.random_uniform([784], -1.0, 1.0))
b2 = tf.Variable(tf.random_uniform([784], -1.0, 1.0))
b3 = tf.Variable(tf.random_uniform([784], -1.0, 1.0))
b4 = tf.Variable(tf.random_uniform([784], -1.0, 1.0))
b5 = tf.Variable(tf.random_uniform([10], -1.0, 1.0))

L1 = tf.sigmoid(tf.matmul(X, W1)+b1)
L2 = tf.sigmoid(tf.matmul(L1, W2)+b2)
L3 = tf.sigmoid(tf.matmul(L2, W3)+b3)
L4 = tf.sigmoid(tf.matmul(L3, W4)+b4)

hypothesis = tf.sigmoid(tf.matmul(L4, W5)+b5)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 10
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