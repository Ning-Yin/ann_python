import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util
import draw
import numpy as np

class Net(nn.Module):
  def __init__(self, avg = 0, std = 1):
    super(Net, self).__init__()
    self.norm = lambda x: (x - avg) / std
    self.fc1 = nn.Linear(2, 3)
    self.fc2 = nn.Linear(3, 1)
    return
  def forward(self, x):
    x = self.norm(x)
    x = self.fc1(x)
    x = torch.sigmoid(x)
    x = self.fc2(x)
    x = torch.sigmoid(x)
    return x

if __name__ == '__main__':
  x = util.read_data('ex4x.dat', use_torch = True)
  y = util.read_data('ex4y.dat', use_torch = True)
  x_avg, x_std = util.compute_normal_param(x, use_torch = True)

  x1_test_mg, x2_test_mg, x_test = util.generate_test_data(x, use_torch = True)

  draw = draw.Draw(x, y, x1_test_mg, x2_test_mg, use_torch = True)

  net = Net(x_avg, x_std)
  print(net)
  optimizer = optim.SGD(net.parameters(), lr = 0.03)
  criterion = nn.MSELoss()
  # TODO implement one-hot and cross-entropy-loss
  # criterion = nn.CrossEntropyLoss()
  sample_num = x.shape[0]
  batch_size = 5
  for epoch in range(1000):
    loss_avg = 0
    for it in range(sample_num // batch_size):
      beg_idx = it * batch_size
      end_idx = sample_num if beg_idx + batch_size > sample_num else beg_idx + batch_size
      net.zero_grad()
      x_train = x[beg_idx:end_idx]
      y_train = y[beg_idx:end_idx]
      y_train_ = net(x_train)
      loss = criterion(y_train_, y_train)
      loss.backward()
      optimizer.step()
      loss_avg += loss * (end_idx - beg_idx)
    loss_avg /= sample_num
    draw.drawLoss(epoch, loss_avg)
    y_test = net(x_test)
    draw.drawPredict(y_test)
