import tensorflow as tf
import numpy as np

X_train, X_test, y_train, y_test = np.load("./protective/5obj.npy", encoding='bytes')

X = tf.placeholder(tf.float32, [None, None, None, None])
X_img = tf.reshape(X, [-1,64,64,3])
Y = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([4,4,3,128], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
print(L1.shape)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
# ((64-2)/2+1) = 32
W2 = tf.Variable(tf.random_normal([4,4,128,256], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# ((32-2)/2+1) = 16

W3 = tf.Variable(tf.random_normal([4,4,256,512], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3 = tf.reshape(L3, [-1,8*8*512])

W4 = tf.get_variable("W4", shape=[8*8*512,2], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2]))
hypothesis = tf.matmul(L3, W4)+b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.0005).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
training_epochs = 100
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		c, _ = sess.run([cost, optimizer], feed_dict={X: X_train, Y: y_train})
		print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(c))
	print("Accuracy: ", accuracy.eval(session = sess, feed_dict = {X: X_test, Y: y_test}))
