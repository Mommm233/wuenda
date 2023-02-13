import numpy as np
import matplotlib.pyplot as plt


def normalization(data_set):
    maxnum = np.max(data_set, axis=0)
    minnum = np.min(data_set, axis=0)
    return (data_set - minnum) / (maxnum - minnum), maxnum, minnum

def load_data():
    path = "./task2/ex1data2.txt"
    data_set = np.fromfile(path, sep=' ')
    data_set = data_set.reshape(data_set.shape[0] // 3, 3)
    (data_set, maxnum, minnum) = normalization(data_set)
    train_set = data_set[:int(0.6 * data_set.shape[0])]
    test_set = data_set[int(0.6 * data_set.shape[0]):]
    train_x, train_y = train_set[:, :-1], train_set[:, -1:]
    test_x, test_y = test_set[:, :-1], test_set[:, -1:]
    return train_x, train_y, test_x, test_y, maxnum, minnum
train_x, train_y, test_x, test_y, maxnum, minnum = load_data()

class network():
    def __init__(self, seed, n):
        np.random.seed(seed)
        self.theta = np.random.randn(n, 1)
        self.b = 0.
    
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
    
net = network(0, 2)
(x_axis, y_axis) = net.train(train_x, train_y, 1000, 0.003)
plt.plot(x_axis, y_axis)
plt.show()

print("theta={}, b={}".format(net.theta, net.b))
print("J_theta={}".format(y_axis[-1]))
predict_y = np.dot(test_x, net.theta) + net.b
predict_error = np.mean((predict_y - test_y) * (predict_y - test_y)) / 2
print("predict_error={}".format(predict_error))


                                

