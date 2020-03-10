import tensorflow as tf
import numpy as np

xy = np.loadtxt('jeju.csv', delimiter=',', dtype=np.float32)
# xy = np.loadtxt('inner.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 6])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([6, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

epochs = 200001
for step in range(epochs):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10000 == 0:
        print(step, "Cost: ", cost_val)
        print("Prediction:")
        print(hy_val)
print('y의 예측 값 : ', sess.run(hypothesis, feed_dict={X:[[24, 20, 0.5, 0.5, 10, 5]]}))


