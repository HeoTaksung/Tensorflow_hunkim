import numpy as np
import tensorflow as tf

se = tf.Session()

x_train = [1,2,3]
y_train = [2.1,3.1,4.1]

w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train*w + b
#reduce_mean = 합하여 평균
#cost = 합 평균(H(xi)-yi의 제곱)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

se.run(tf.global_variables_initializer())

for step in range(2001):
    se.run(train)
    if step%20 == 0:
        print(step, se.run(cost), se.run(w), se.run(b))
