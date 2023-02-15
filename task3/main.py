# 线性的逻辑回归
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    path = "./task3/ex2data1.txt"
    data_set = np.fromfile(path, sep=' ')
    data_set = data_set.reshape(data_set.shape[0] // 3, 3)
    train_set = data_set[:int(0.6 * data_set.shape[0])]
    test_set = data_set[int(0.6 * data_set.shape[0]):]
    return train_set, test_set

train_set, test_set = load_data()
train_x, train_y = train_set[:, :-1], train_set[:, -1:]

class network():
    def __init__(self, seed, n):
        np.random.seed(seed)
        self.theta = np.random.randn(n, 1)
        self.b = 0.

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descend(self, x, y, alpha):
        h = self.sigmoid(np.dot(x, self.theta) + self.b)
        self.theta = self.theta - alpha * np.dot(x.T, h - y) / x.shape[0]
        self.b = self.b - alpha * np.mean(h - y)
        
    def compute_cost(self, x, y):
        h = self.sigmoid(np.dot(x, self.theta) + self.b)
        return -np.mean((y * np.log(h) + (1 - y) * np.log(1 - h)))

    def train(self, x, y, iteration, alpha):
        x_axis = [_ for _ in range(iteration)]
        y_axis = []
        for i in range(iteration):
            self.gradient_descend(x, y, alpha)
            y_axis.append(self.compute_cost(x, y))
        return x_axis, y_axis

net = network(0, 2)
(x_axis, y_axis) = net.train(train_x, train_y, 100000, 0.001)
print("net.theta = {}, net.b = {}".format(net.theta, net.b))
plt.plot(x_axis, y_axis)
plt.show()
right = 0
ture_pos, false_pos, false_neg, ture_neg = 0, 0, 0, 0
for i in range(train_x.shape[0]):
    h = net.sigmoid(np.dot(train_x[i:i+1,], net.theta) + net.b)
    if h >= 0.5: h = 1
    else: h = 0
    if h == train_y[i]: 
        right += 1
        if h == 1: ture_pos += 1
        else: ture_neg += 1
    else:
        if h == 1: false_pos += 1
        else: false_neg += 1
print("right percent = {}".format(right / train_x.shape[0]))
print("precision = {}".format(ture_pos / (ture_pos + false_pos)))
print("recall = {}".format(ture_pos / (ture_pos + false_neg)))

def draw(set):
    colors = []
    for i in range(set.shape[0]):
        if(set[i][-1] == 0): colors.append('blue')
        else: colors.append('red')
    x1, x2 = set[:, :1], set[:, 1:-1]
    plt.scatter(x1, x2, c=colors)
draw(train_set)
x1 = np.arange(0, 130, 0.1)
x2 = -(net.b + net.theta[0] * x1) / net.theta[1] 
plt.plot(x1, x2)
plt.show()
