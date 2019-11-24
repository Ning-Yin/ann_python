import numpy as np
import numpy.matlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':

    data_x = np.loadtxt("ex4x.dat")
    data_y = np.loadtxt("ex4y.dat")
    min_x = data_x.min(axis=0)
    max_x = data_x.max(axis=0)
    data_x = (data_x - min_x) / (max_x - min_x)  # 80*2
    pos_idx = np.where(data_y == 1)
    neg_idx = np.where(data_y == 0)
    encode = OneHotEncoder(sparse=False, categories='auto')
    data_y = encode.fit_transform(data_y.reshape(-1, 1))
    num = 200
    xx, yy = np.meshgrid(np.linspace(-0.1, 1.1, num), np.linspace(-0.1, 1.1, num))
    cm_light = mpl.colors.ListedColormap(['#EE82EE', '#FFB6C1'])
    fig = plt.figure(figsize=(16, 9))
    fold = 5
    test_num = int(data_x.shape[0] / fold)
    learningrate = 0.15
    layernum = 7
    layernode = 5
    x_ = data_x
    y_ = data_y
    total = 0
    for m in range(5):
        losslist = []
        test_x = x_[m, :]
        test_y = y_[m, :]
        test_index = np.ones(test_num)
        for i in range(1, test_num):
            test_x = np.c_[test_x, x_[5 * i + m, :]]
            test_y = np.c_[test_y, y_[5 * i + m, :]]
            test_index[i] = 5 * i + m
        test_x = test_x.T  # 测试数据
        test_y = test_y.T
        data_x = x_[0, :]
        data_y = y_[0, :]
        for i in range(80):
            if i % 5 != m:
                data_x = np.c_[data_x, x_[i, :]]
                data_y = np.c_[data_y, y_[i, :]]
        data_x = data_x[:, 1::]
        data_y = data_y[:, 1::]
        data_x = data_x.T  # 训练数据
        data_y = data_y.T

        W = {}
        B = {}
        for i in range(1, layernum):
            if i == 1:
                W[i] = np.random.rand(layernode, 2)  # l * 2
                B[i] = np.random.rand(layernode, 1)
            elif i == layernum - 1:
                W[i] = np.random.rand(2, layernode)  # 2 * l
                B[i] = np.random.rand(2, 1)
            else:
                W[i] = np.random.rand(layernode, layernode)  # l * l
                B[i] = np.random.rand(layernode, 1)
        steps = 800
        temp = np.ones(data_x.shape[0])
        for i in range(steps):
            '''前向传播'''
            Hidden_in = {}
            Hidden_out = {}
            for k in range(1, layernum):
                if k == 1:
                    Hidden_in[k] = W[k].dot(data_x.T)  # l*2  2*m
                    Hidden_out[k] = sigmoid(Hidden_in[k] + B[k])  # l * m
                else:
                    Hidden_in[k] = W[k].dot(Hidden_out[k - 1])  # l * m
                    Hidden_out[k] = sigmoid(Hidden_in[k] + B[k])  # l * m

            # print(Hidden_out[layernum-1].T)
            loss = 0.5 * np.sum((Hidden_out[layernum - 1] - data_y.T) ** 2)
            losslist.append(loss)

            '''后向反馈'''
            Error = {}
            Grad = {}
            for k in range(layernum - 1, 0, -1):
                if k == layernum - 1:
                    Error[k] = np.multiply(np.multiply((Hidden_out[k] - data_y.T), Hidden_out[k]),
                                           (1 - Hidden_out[k]))  # 2 * m
                    Grad[k] = np.dot(Error[k], Hidden_out[k - 1].T)  # 2 * l （W ）*（X | 1）
                elif k == 1:
                    Error[k] = np.multiply(np.multiply(np.dot(W[k + 1].T, Error[k + 1]), Hidden_out[k]),
                                           1 - Hidden_out[k])
                    Grad[k] = np.dot(Error[k], data_x)
                else:
                    Error[k] = np.multiply(np.multiply(np.dot(W[k + 1].T, Error[k + 1]), Hidden_out[k]),
                                           1 - Hidden_out[k])  # l*m
                    Grad[k] = np.dot(Error[k], Hidden_out[k - 1].T)  # l* l

            for k in range(1, layernum, 1):
                W[k] = W[k] - learningrate * Grad[k]
                B[k] = B[k] - learningrate * np.sum(Error[k], axis=1, keepdims=True)

            '''绘图'''

            Hidden_in_p = {}
            Hidden_out_p = {}
            for k in range(1, layernum):
                if k == 1:
                    Hidden_in_p[k] = W[k].dot(np.c_[xx.ravel(), yy.ravel()].T)
                    Hidden_out_p[k] = sigmoid(Hidden_in_p[k] + B[k])
                else:
                    Hidden_in_p[k] = W[k].dot(Hidden_out_p[k - 1])
                    Hidden_out_p[k] = sigmoid(Hidden_in_p[k] + B[k])  # l * m

            output = np.argmax(Hidden_out_p[layernum - 1], axis=0)
            output = output.reshape(xx.shape)
            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)

            ax1.pcolormesh(xx, yy, output, cmap=cm_light)
            ax1.scatter(x_[pos_idx, 0], x_[pos_idx, 1], s=100, marker='+', label='admit')
            ax1.scatter(x_[neg_idx, 0], x_[neg_idx, 1], s=100, marker='o', label='not admit')

            for k in range(test_x.shape[0]):
                if test_y[k][0] == 0:
                    ax1.scatter(test_x[k][0], test_x[k][1], c="#FF0000", s=100, marker='+', label='test')
                else:
                    ax1.scatter(test_x[k][0], test_x[k][1], c="#FF0000", s=100, marker='o', label='test')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title("cost")
            ax2.plot(losslist)
            plt.pause(0.01)
        '''测试'''

        Hidden_in_t = {}
        Hidden_out_t = {}
        for k in range(1, layernum):
            if k == 1:
                Hidden_in_t[k] = W[k].dot(test_x.T)  # m * l
                Hidden_out_t[k] = sigmoid(Hidden_in_t[k] + B[k])  # m*l
            else:
                Hidden_in_t[k] = W[k].dot(Hidden_out_t[k - 1])
                Hidden_out_t[k] = sigmoid(Hidden_in_t[k] + B[k])

        print(Hidden_out_t[layernum - 1].T)
        print(test_y)
        predict = np.argmax(Hidden_out_t[layernum - 1], axis=0)
        truevalue = np.argmax(test_y, axis=1)
        count = np.sum(np.equal(predict, truevalue))
        print("test accuracy", m + 1, ":  ", count / test_num)
        total = total + count / test_num
    print("accuracy:", total / 5)
