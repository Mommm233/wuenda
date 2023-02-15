# 非线性的逻辑回归
import numpy as np
import matplotlib.pyplot as plt


def load_data(seed):
    path = "./task4/ex2data2.txt"
    data_set = np.fromfile(path, sep=' ')
    data_set = data_set.reshape(data_set.shape[0] // 3, 3)
    # 按顺序非常离谱，打乱
    idx = [_ for _ in range(data_set.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_set, test_set = [], []
    for i in range(int(0.6 * data_set.shape[0])):
        train_set.append(data_set[idx[i]:idx[i] + 1,][0])
    for i in range(int(0.6 * data_set.shape[0]), data_set.shape[0], 1):
        test_set.append(data_set[idx[i]:idx[i] + 1,][0])
    train_set = np.array(train_set)
    test_set = np.array(test_set)
    return train_set, test_set

train_set, test_set = load_data(0)

def map(x1, x2, power):     # 2维映射为48维
    res = []
    for i in range(power + 1):
        for j in range(power + 1):
            if i == 0 and j == 0: continue
            res.append((np.power(x1, i) * np.power(x2, j)).T[0])
    return np.array(res).T

train_x = map(train_set[:, :1],train_set[:, 1:2],6)
train_y = train_set[:, -1:]

class network():
    def __init__(self, seed, n):
        np.random.seed(seed)
        self.theta = np.random.randn(n, 1)
        self.b = 0.

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descend(self, x, y, alpha, _lambda):
        h = self.sigmoid(np.dot(x, self.theta) + self.b)
        self.theta = self.theta - alpha * (np.dot(x.T, h - y) + _lambda * self.theta) / x.shape[0]
        self.b = self.b - alpha * np.mean(h - y)
        
    def compute_cost(self, x, y, _lambda):
        h = self.sigmoid(np.dot(x, self.theta) + self.b)
        return -np.mean((y * np.log(h) + (1 - y) * np.log(1 - h))) + _lambda * np.mean(self.theta * self.theta) / 2

    def train(self, x, y, iteration, alpha, _lambda):
        x_axis = [_ for _ in range(iteration)]
        y_axis = []
        for i in range(iteration):
            self.gradient_descend(x, y, alpha, _lambda)
            y_axis.append(self.compute_cost(x, y, _lambda))
        return x_axis, y_axis

net = network(0, train_x.shape[1])
(x_axis, y_axis) = net.train(train_x, train_y, 200000, 0.001, 0.01)
plt.plot(x_axis, y_axis)
plt.show()

def get_percent():
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

get_percent()

def draw(set):
    colors = []
    for i in range(set.shape[0]):
        if(set[i][-1] == 0): colors.append('blue')
        else: colors.append('red')
    x1, x2 = set[:, :1], set[:, 1:-1]
    plt.scatter(x1, x2, c=colors)

draw(train_set)
# draw(test_set)

def draw_dicision_boundary():
    l = np.linspace(-1, 1, 250)
    x1, x2 = np.meshgrid(l, l)      
    '''
    x1 = -1         x2 = -1
         -0.7            -1
          .               .
          .               .
          .               .
          1              -1
         -1              -0.7
         -0.7            -0.7
          .
          .
          .
          1
          .
          .
          .
    '''
    # np.set_printoptions(threshold=1e6)
    x = map(x1.ravel()[:, np.newaxis], x2.ravel()[:, np.newaxis], 6)
    z = np.dot(x, net.theta) + net.b
    z = z.reshape(x1.shape)
    plt.contour(x1, x2, z, 0)       # 画等高线为0的图
draw_dicision_boundary()
plt.show()

