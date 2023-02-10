import numpy as np
import matplotlib.pyplot as plt

def load_data():
    path = "./task1/ex1data1.txt"
    data_set = np.fromfile(path, sep=' ')
    data_set = data_set.reshape(data_set.shape[0] // 2, 2)
    train_set = data_set[:int(0.6 * data_set.shape[0])]
    test_set = data_set[int(0.6 * data_set.shape[0]):]
    train_x, train_y = train_set[:, :-1], train_set[:, -1:]
    test_x, test_y = test_set[:, :-1], test_set[:, -1:]
    return train_x, train_y, test_x, test_y
train_x, train_y, test_x, test_y = load_data()
fig, (ax1, ax2) = plt.subplots(1, 2)    # 1行2列
ax1.plot(train_x, train_y, 'o')     # 'o' -> 画离散点
class network():
    def __init__(self, seed, n):
        np.random.seed(seed)
        self.theta = np.random.randn(n, 1)
        self.b = 0
    
    def forward(self, x):
        z = np.dot(x, self.theta) + self.b
        return z

    def gradient_descend(self, x, y, alpha):
        z = self.forward(x)
        self.theta = self.theta - alpha * np.dot(x.T, z - y) / x.shape[0]
        self.b = self.b - alpha * np.mean(z - y)
        
    def compute_cost(self, x, y):
        z = self.forward(x)
        return np.mean((z - y) * (z - y)) / 2

    def train(self, x, y, iteration, alpha):
        x_axis = [_ for _ in range(iteration)]
        y_axis = []
        for i in range(iteration):
            self.gradient_descend(x, y, alpha)
            y_axis.append(self.compute_cost(x, y))
        return x_axis, y_axis
    
net = network(0, 1)
(x_axis, y_axis) = net.train(train_x, train_y, 1000, 0.003)
ax2.plot(x_axis, y_axis)
x_label = [_ for _ in range(25)]
y_label = [float(net.theta * x_axis[_] + net.b) for _ in range(25)]
ax1.plot(x_label, y_label)
ax1.plot(test_x, test_y, 'o', color = 'black')
plt.show()
print("theta={}, b={}".format(net.theta, net.b))
print("J_theta={}".format(y_axis[-1]))
