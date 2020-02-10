import tensorflow as tf

x_data = [[1.8, 16.3],[1.8,18],[1.8,20],[1.8,16.3],[0,20],[1.8,20],[1.8,18.3],[1.8,20],[1.8,20],[1.8,19.7],[1.8,0]]
y_data = [[15.5],[13.7],[16.5],[11.1],[11.3],[13.1],[13.3],[7.8],[16.8],[14.1],[10.1]]


X = tf.placeholder(tf.float32, shape = [None,2]) 
Y = tf.placeholder(tf.float32, shape = [None,1])
W = tf.Variable(tf.random_normal([2,1]), name='weight') 
b = tf.Variable(tf.random_normal([1]), name='bias') 

hypothesis = tf.matmul(X, W)+b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(200001):
	cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X:x_data, Y:y_data})
	if step % 10000 == 0:
		print(step, "Cost: ", cost_val)
		print("Prediction:")
		print(hy_val)
		