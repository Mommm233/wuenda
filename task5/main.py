# 线性svm
import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.axisartist as axisartist

def load_data():
    path = "./task5/ex2data1.txt"
    data_set = np.fromfile(path, sep=' ')
    data_set = data_set.reshape(data_set.shape[0] // 3, 3)
    train_set = data_set[:int(0.6 * data_set.shape[0])]
    test_set = data_set[int(0.6 * data_set.shape[0]):]
    return train_set, test_set

train_set, test_set = load_data()
train_x, train_y = train_set[:, :-1], train_set[:, -1:]

class network():
    def __init__(self, seed, n, k):
        np.random.seed(seed)
        self.theta = np.random.randn(n, 1)
        self.b = 0.
        self.k = k

    def devide(self, x, y):
        x_one, y_one, x_zero, y_zero = [], [], [], []
        for i in range(x.shape[0]):
            if(y[i][0] == 1):
                x_one.append(x[i:i + 1,][0])
                y_one.append(y[i:i + 1,][0])
            else:
                x_zero.append(x[i:i + 1,][0])
                y_zero.append(y[i:i + 1,][0])
        return np.array(x_one), np.array(y_one), np.array(x_zero), np.array(y_zero)
    
    def get_delta(self, x, sf, z):
        nx = self.k * x
        if sf == 1:
            for i in range(z.shape[0]):
                if z[i][0] >= 1:
                    nx[i:i+1, ] = 0.
            delta = np.sum(nx, axis=0)[:, np.newaxis]
        else:
            for i in range(z.shape[0]):
                if z[i][0] <= -1:
                    nx[i:i+1, ] = 0.
            delta = np.sum(-nx, axis=0)[:, np.newaxis]
        #print("delta.shape={}".format(delta.shape))
        return delta

    def gradient_descend(self, alpha, x_one, x_zero, C):
        z_one = np.dot(x_one, self.theta) + self.b
        z_zero = np.dot(x_zero, self.theta) + self.b
        self.theta = self.theta - alpha * (self.get_delta(x_one, 1, z_one) + self.get_delta(x_zero, 0, z_zero)) * C
        ones0, ones1 = np.ones((x_zero.shape[0], 1)), np.ones((x_one.shape[0], 1))
        self.b = self.b - alpha * (self.get_delta(ones1, 1, z_one) + self.get_delta(ones0, 0, z_zero))[0] * C
        
    def compute_cost(self, x, sf):
        z = np.dot(x, self.theta) + self.b
        cost = 0.
        if sf == 1:
            for i in range(z.shape[0]):
                if z[i][0] < 1:
                    cost += self.k * (z[i][0] - 1)
        else:
            for i in range(z.shape[0]):
                if z[i][0] > -1:
                    cost += -self.k * (z[i][0] + 1)
        return cost
    
    def train(self, x, y, iteration, alpha, C):
        (x_one, y_one, x_zero, y_zero) = self.devide(x, y)
        x_axis = [_ for _ in range(iteration)]
        y_axis = []
        for i in range(iteration):
            self.gradient_descend(alpha, x_one, x_zero, C)
            y_axis.append(self.compute_cost(x_one, 1) + self.compute_cost(x_zero, 0))
        return x_axis, y_axis
    
# def get_k():
#     x_0 = 0.
#     alpha = 0.001
#     cost = []
#     ite = []
#     for i in range(100000):
#         x_0 -= alpha * ((np.exp(-x_0) / (1 + np.exp(-x_0)) \
#                 * (1 - x_0) - np.log(1 + np.exp(-x_0))) \
#                 * (np.exp(-x_0) * (x_0 - 1) / (1 + np.exp(-x_0)) ** 2)) 
#         cost.append((np.exp(-x_0) / (1 + np.exp(-x_0)) \
#                 * (1 - x_0) - np.log(1 + np.exp(-x_0))) ** 2 / 2)
#         ite.append(i)
#     # plt.plot(ite, cost)
#     # plt.show()
#     return -np.exp(-x_0) / (1 + np.exp(-x_0))

net = network(0, 2, -np.log(2))
(x_axis, y_axis) = net.train(train_x, train_y, 100000, 0.001, 0.01)
print("net.theta = {}, net.b = {}".format(net.theta, net.b))
plt.plot(x_axis, y_axis)
plt.show()
right = 0
ture_pos, false_pos, false_neg, ture_neg = 0, 0, 0, 0
for i in range(train_x.shape[0]):
    z = np.dot(train_x[i:i+1,], net.theta) + net.b
    if z >= 0: h = 1
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
print(net.theta, net.b)
draw(train_set)
x1 = np.arange(0, 130, 0.1)
x2 = -(net.b + net.theta[0] * x1) / net.theta[1] 
plt.plot(x1, x2)
plt.show()


# k = get_k()
# z = np.linspace(-5, 5, 100)
# J = np.log(1 + np.exp(-z))
# y1 = k * (z - 1)
# y2 = -k * (z + 1)
# fig = plt.figure()
# ax = axisartist.Subplot(fig, 111)
# fig.add_axes(ax)
# ax.axis["x"] = ax.new_floating_axis(0, 0)
# ax.axis["x"].set_axisline_style("->", size = 1.0)  # 给x坐标轴加上箭头
# ax.axis["x"].set_axis_direction("top")
# ax.axis["y"] = ax.new_floating_axis(1, 0)
# ax.axis["y"].set_axisline_style("-|>", size = 1.0)  # 给x坐标轴加上箭头
# ax.axis["y"].set_axis_direction("right")
# plt.xlim(-5,5)
# plt.ylim(-5,5)
# plt.plot(z, J)
# plt.plot(z, z + np.log(1 + np.exp(-z)))
# plt.plot(z, y1)
# plt.plot(z, y2)
# plt.show()