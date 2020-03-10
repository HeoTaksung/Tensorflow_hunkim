import numpy as np
import matplotlib.pyplot as plt

ab = np.loadtxt('jeju.csv', delimiter=',', dtype=np.float32)
xy = np.loadtxt('inner.csv', delimiter=',', dtype=np.float32)
# x_data = xy[:, 0:-1]
x_data = xy[:, 1:2]
y_data = xy[:, [-1]]
z_data = ab[:, [-1]]

x = [0.0, 15.0, 20.0, 25.0]
y1 = [12.0, 12.0, 1.017, 0.517]
y2 = [12.0, 12.0, 1.15, 0.433]
y3 = [12.0, 11.217, 0.8, 0.517]


z1 = [12.0, 9.75, 0.983, 0.533]
z2 = [12.0, 4.25, 0.917, 0.517]
z3 = [12.0, 12.0, 0.917, 0.75]
z4 = [12.0, 12.0, 0.9, 0.6]
y = []
z = []

plt.plot(x, y1, 'b', label = 'inner1')
plt.plot(x, y2, 'y', label = 'inner2')
plt.plot(x, y3, 'g', label = 'inner3')
plt.plot(x, z1, '--', label = 'jeju1')
plt.plot(x, z2, '--', label = 'jeju2')
plt.plot(x, z3, '--', label = 'jeju3')
plt.plot(x, z4, '--', label = 'jeju4')
plt.ylim(0, 13)
plt.xlabel('Salinity')
plt.ylabel('Motality Time')
plt.legend()
plt.show()
