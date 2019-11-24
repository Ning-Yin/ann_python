import numpy as np
import matplotlib.pyplot as plt
import torch

class Draw:
  def __init__(self, x, y, x_mg, use_torch = False, one_hot = False, C = 2):
    self.use_torch = use_torch
    self.one_hot = one_hot
    self.C = C
    if self.use_torch:
      x = x.numpy()
      y = y.numpy()
    y = np.squeeze(y)
    if self.one_hot:
      self.levels = [i - 0.5 for i in range(self.C + 1)]
    else:
      self.levels = [i / 10 for i in range(11)]
    self.x_sample = [x[np.where(y == i)] for i in range(self.C)]
    # self.x_pos = x[np.where(y == 1)]
    # self.x_neg = x[np.where(y == 0)]
    self.x1_mg = x_mg[0]
    self.x2_mg = x_mg[1]
    plt.ion()
    _, self.ax = plt.subplots(1, 2)
    self.samp_marker = '.'
    # self.pos_samp_color = 'crimson'
    # self.neg_samp_color = 'indigo'
    # self.pos_pred = 'om'
    # self.neg_pred = 'oc'
    self.samp_color = ['crimson', 'indigo', 'greenyellow', 'darkmagenta', 'turquoise'][:self.C]
    self.iter = []
    self.loss = []
    self.drawSamples()
    plt.pause(0.5)
    return
  def drawSamples(self):
    # if self.one_hot:
    for i in range(self.C):
      self.ax[0].scatter(self.x_sample[i][:, 0], self.x_sample[i][:, 1], color = self.samp_color[i], marker = self.samp_marker)
    # else:
      # self.ax[0].scatter(self.x_pos[:, 0], self.x_pos[:, 1], color = self.pos_samp_color, marker = self.samp_marker)
      # self.ax[0].scatter(self.x_neg[:, 0], self.x_neg[:, 1], color = self.neg_samp_color, marker = self.samp_marker)
    return
  def drawPredict(self, y_):
    if self.use_torch:
      y_ = y_.detach().numpy()
    y_ = np.reshape(y_, self.x1_mg.shape)
    self.ax[0].contourf(self.x1_mg, self.x2_mg, y_, self.levels, cmap = plt.cm.rainbow, zorder = 0)
    plt.pause(0.001)
    # x_pos = x[np.where(y_ <= 0.5)]
    # x_neg = x[np.where(y_ > 0.5)]
    # self.ax[0].plot(x_pos[:, 0], x_pos[:, 1], self.pos_pred)
    # self.ax[0].plot(x_neg[:, 0], x_neg[:, 1], self.neg_pred)
    # self.drawSamples()
    return
  def drawLoss(self, iter, loss):
    self.iter.append(iter)
    self.loss.append(loss)
    self.ax[1].plot(self.iter, self.loss, 'c', label = 'loss')
    print(loss)
    return
