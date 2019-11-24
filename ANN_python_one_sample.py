import numpy as np
import abc
import util
import draw

class Operator(abc.ABC):
  @abc.abstractmethod
  def forward(self, x):
    pass
  @abc.abstractmethod
  def backward(self, e):
    pass
  @abc.abstractmethod
  def update(self, lr):
    pass

class AddConstant(Operator):
  def __init__(self, c):
    self.c = c
    return
  def forward(self, x):
    self.x = x
    self.y = self.x + self.c
    return self.y
  def backward(self, e):
    return e
  def update(self, lr):
    pass

class SubConstant(Operator):
  def __init__(self, c):
    self.c = c
    return
  def forward(self, x):
    self.x = x
    self.y = self.x - self.c
    return self.y
  def backward(self, e):
    return e
  def update(self, lr):
    pass

class MulConstant(Operator):
  def __init__(self, c):
    self.c = c
    return
  def forward(self, x):
    self.x = x
    self.y = self.x * self.c
    return self.y
  def backward(self, e):
    e = e * self.c
    return e
  def update(self, lr):
    pass

class DivConstant(Operator):
  def __init__(self, c):
    self.c = c
    return
  def forward(self, x):
    self.x = x
    self.y = self.x / self.c
    return self.y
  def backward(self, e):
    e = e / self.c
    return e
  def update(self, lr):
    pass

class Linear(Operator):
  def __init__(self, nx, ny):
    self.w = np.random.randn(nx, ny)
    self.b = np.random.randn(1, ny)
    self.dw = np.zeros((nx, ny))
    self.db = np.zeros((1, ny))
    return
  def forward(self, x):
    self.x = x
    self.y = np.matmul(self.x, self.w) + self.b
    return self.y
  def backward(self, e):
    self.dw = np.matmul(self.x.T, e)
    self.db = e
    e = np.matmul(e, self.w.T)
    return e
  def update(self, lr):
    self.w -= lr * self.dw
    self.b -= lr * self.db
    return

class Sigmoid(Operator):
  # def __init__(self, n):
  #   self.x = np.zeros(n)
  #   self.y = np.zeros(n)
  #   return
  def __init__(self):
    pass
  def forward(self, x):
    self.x = x
    self.y = 1/(1+np.exp(-self.x))
    return self.y
  def backward(self, e):
    e = e * (self.y * (1 - self.y))
    return e
  def update(self, lr):
    pass

class Relu(Operator):
  # def __init__(self, n):
  #   self.x = np.zeros(n)
  #   self.y = np.zeros(n)
  #   return
  def __init__(self):
    pass
  def forward(self, x):
    self.x = x
    self.y = np.maximum(0, x)
    return self.y
  def backward(self, e):
    e = e * (self.x > 0)
    return e
  def update(self, lr):
    pass

class Tanh(Operator):
  # def __init__(self, n):
  #   self.x = np.zeros(n)
  #   self.y = np.zeros(n)
  #   return
  def __init__(self):
    pass
  def forward(self, x):
    self.x = x
    t1 = np.exp(x)
    t2 = np.exp(-x)
    self.y = (t1 - t2) / (t1 + t2)
    return self.y
  def backward(self, e):
    e = e * (1 - self.y ** 2)
    return e
  def update(self, lr):
    pass

class Identity(Operator):
  def __init__(self, n):
    self.x = np.zeros(n)
    self.y = np.zeros(n)
    return
  def forward(self, x):
    self.x = x
    self.y = self.x
    return self.y
  def backward(self, e):
    return e
  def update(self, lr):
    pass

class CostFunction(abc.ABC):
  @abc.abstractmethod
  def cost(self, y_, y):
    pass
  @abc.abstractmethod
  def error(self, y_, y):
    pass

class LeastSquare(CostFunction):
  def cost(self, y_, y, avg = True):
    if avg:
      J = np.mean((y_ - y) ** 2)
    else:
      J = np.sum((y_ - y) ** 2)
    return J
  def error(self, y_, y):
    e = y_ - y
    return e

# TODO implement
# one-hot
class Entropy(CostFunction):
  def cost(self, y_, y):
    # TODO correct the function
    J = -np.log(y_[np.where(y == 1)])-np.log(1 - y_[np.where(y == 0)])
    return J
  def error(self, y_, y):
    # TODO correct the function
    e = -1 / y_[np.where(y == 1)]
    return e

class Optimizer(abc.ABC):
  @abc.abstractmethod
  def step(self):
    pass

class GradDesc(Optimizer):
  def __init__(self, lr):
    self.lr = lr
    return
  def step(self):
    return self.lr

class DecayGradDesc(Optimizer):
  def __init__(self, init_lr, alpha):
    self.init_lr = init_lr
    self.alpha = alpha
    self.init()
    return
  def init(self):
    self.lr = self.init_lr
    return
  def step(self):
    lr = self.lr
    self.lr *= self.alpha
    return lr

class Net:
  def __init__(self, layer_list, cost_func, optimizer):
    self.ll = layer_list
    self.cf = cost_func
    self.optim = optimizer
    return
  def forward(self, x):
    # print('x_shape: {}'.format(x.shape))
    for layer in self.ll:
      x = layer.forward(x)
      # print('x_shape: {}'.format(x.shape))
    return x
  def backward(self, e):
    for layer in reversed(self.ll):
      e = layer.backward(e)
    return
  def update(self, lr):
    for layer in self.ll:
      layer.update(lr)
    return
  def train(self, x, y, cost = True):
    y_ = self.forward(x)
    e = self.cf.error(y_, y)
    self.backward(e)
    self.update(self.optim.step())
    if cost:
      J = self.cf.cost(y_, y)
      return J
    else:
      return
  def predict(self, x):
    y_ = self.forward(x)
    return y_

if __name__ == '__main__':
  x = util.read_data('ex4x.dat')
  y = util.read_data('ex4y.dat', squeeze = True)
  x_avg, x_std = util.compute_normal_param(x)

  x1_test_mg, x2_test_mg, x_test = util.generate_test_data(x)

  draw = draw.Draw(x, y, x1_test_mg, x2_test_mg)

  sub1 = SubConstant(x_avg)
  div1 = DivConstant(x_std)
  fc1 = Linear(2, 10)
  active1 = Sigmoid()
  fc2 = Linear(10, 1)
  active2 = Sigmoid()
  layer_list = [sub1, div1, fc1, active1, fc2, active2]
  cost_func = LeastSquare()
  optimizer = GradDesc(0.1)
  net = Net(layer_list, cost_func, optimizer)

  sample_num = x.shape[0]
  batch_size = 1
  for epoch in range(1000):
    cost_avg = 0
    for it in range(sample_num // batch_size):
      beg_idx = it * batch_size
      end_idx = sample_num if beg_idx + batch_size > sample_num else beg_idx + batch_size
      x_train = x[beg_idx:end_idx]
      y_train = y[beg_idx:end_idx]
      cost = net.train(x_train, y_train)
      # draw.drawLoss((epoch * (sample_num // batch_size) + it), cost)
      cost_avg += cost * (end_idx - beg_idx)
    cost_avg /= sample_num
    draw.drawLoss(epoch, cost_avg)
    y_test = net.predict(x_test)
    draw.drawPredict(y_test)