import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    # print(np.unique(y))   [1, 2, ..., 10] 标签
    # print(X.shape, y.shape)   (5000, 400)    (5000,1)
    # 按顺序非常离谱，打乱
    idx = [_ for _ in range(X.shape[0])]
    np.random.seed(0)
    np.random.shuffle(idx)
    tX, ty = [], []
    for i in range(X.shape[0]):
        tX.append(X[idx[i]:idx[i] + 1, :][0])
        ty.append(y[idx[i]:idx[i] + 1, :][0])

    tX = np.array(tX)
    ty = np.array(ty)
    train_X = tX[:int(tX.shape[0] * 0.6)]
    train_y = ty[:int(ty.shape[0] * 0.6)]
    test_X = tX[int(tX.shape[0] * 0.6):]
    test_y = ty[int(ty.shape[0] * 0.6):]
    return train_X, train_y, test_X, test_y

(train_X, train_y, test_X, test_y) = load_data("./task7/ex3data1.mat")
cnt = np.unique(train_y).shape[0]
# np.set_printoptions(threshold=10e6)
# print(test_X.shape)

def plot_an_image(X, y):
    pick_one = np.random.randint(0, X.shape[0])
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')      # 20 * 20 = 400
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print("this should be {}".format(y[pick_one]))

# for i in range(10):
#     plot_an_image(train_X, train_y)

class network():
    def __init__(self, seed, n, cnt):
        np.random.seed(seed)
        self.w = np.random.randn(n, cnt)
        self.b = [0] * cnt

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient_descend(self, x, y, i, alpha, _lambda):
        h = self.sigmoid(np.dot(x, self.w[:, i][:, np.newaxis]) + self.b[i])
        self.w[:, i][:, np.newaxis] = self.w[:, i][:, np.newaxis] - \
            alpha * (np.dot(x.T, h - y) + _lambda * self.w[:, i][:, np.newaxis]) / x.shape[0]
        self.b[i] = self.b[i] - alpha * np.mean(h - y)
        
    def compute_cost(self, x, y, i, _lambda):
        h = self.sigmoid(np.dot(x, self.w[:, i][:, np.newaxis]) + self.b[i])
        return -np.mean((y * np.log(h) + (1 - y) * np.log(1 - h))) + \
            _lambda * np.mean(self.w[:, i] * self.w[:, i]) / 2

    def train(self, x, y, i, iteration, alpha, _lambda):
        ty = np.zeros(y.shape)      # 不能改y，传的好像是指针
        for j in range(ty.shape[0]):
            if y[j][0] == i: ty[j][0] = 1
            else: ty[j][0] = 0
        # x_axis = [_ for _ in range(iteration)]
        # y_axis = []
        for j in range(iteration):
            if (j + 1) % 50 == 0:
                print("正在进行第{}个分类器的第{}轮迭代".format(i, (j + 1) / 50 + 1))
            self.gradient_descend(x, ty, i, alpha, _lambda)
        #     y_axis.append(self.compute_cost(x, ty, i, _lambda))
        # plt.plot(x_axis, y_axis)
        # plt.show()


# net = network(0, train_X.shape[1], cnt)
# for i in range(cnt):
#     net.train(train_X, train_y, i, 100000, 0.001, 0.01)
# np.save("./task7/w.npy", net.w)
# np.save("./task7/b.npy", net.b)

def get_score(x, y):
    w = np.load("./task7/w.npy")
    b = np.load("./task7/b.npy")
    right = 0
    h = np.dot(x, w) + b
    for i in range(h.shape[0]):
        idx, max = 0, h[i, 0]
        for j in range(1, cnt, 1):
            if h[i, j] > max:
                idx, max = j, h[i, j]
        if idx == y[i][0]: right += 1
        #print("第{}个样本预测值为{},实际为{}".format(i, idx, y[i][0]))
    print("predict percent = {}%".format(right * 100 / x.shape[0]))

get_score(train_X, train_y)
get_score(test_X, test_y)
