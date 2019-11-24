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
    self.dw = np.mean(np.matmul(self.x.T, e), axis = 0)
    self.db = np.mean(e, axis = 0)
    e = np.matmul(e, self.w.T)
    return e
  def update(self, lr):
    self.w -= lr * self.dw
    self.b -= lr * self.db
    return

# TODO verify
class Softmax(Operator):
  def __init__(self):
    pass
  def forward(self, x):
    # in order to avoid the overflow when using exponential function
    x_max = np.expand_dims(np.max(x, axis = 1), axis = 1)
    x_modify = x - x_max
    exp_x = np.exp(x_modify)
    sum_exp = np.expand_dims(np.sum(exp_x, axis = 1), axis = 1)
    self.y = exp_x / sum_exp
    return self.y
  def backward(self, e):
    # e.shape: (sample_num, ny)
    # y_3dim.shape: (sample_num, ny, 1)
    y_3dim = np.expand_dims(self.y, axis = 2)
    # print('e.shape:{}'.format(e.shape))
    # print('y.shape:{}'.format(self.y.shape))
    # print('y_3dim.shape:{}'.format(y_3dim.shape))
    # gradient_3dim.shape: (sample_num, ny, ny)
    gradient_3dim = -np.matmul(y_3dim, np.transpose(y_3dim, (0, 2, 1)))
    # diagonal_3dim.shape: (sample_num, ny, ny)
    diagonal_3dim = np.zeros_like(gradient_3dim)
    for i in range(self.y.shape[1]):
      diagonal_3dim[:, i, i] = self.y[:,i]
    gradient_3dim = gradient_3dim + diagonal_3dim
    # print('gradient_3dim.shape:{}'.format(gradient_3dim.shape))
    # e_3dim.shape: (sample_num, 1, ny)
    e_3dim = np.expand_dims(e, axis = 1)
    # print('e_3dim.shape:{}'.format(e_3dim.shape))
    # e_new_3dim.shape: (sample_num, 1, ny)
    e_new_3dim = np.matmul(e_3dim, gradient_3dim)
    # print('e_new_3dim.shape:{}'.format(e_new_3dim.shape))
    # e_new.shape: (sample_num, ny)
    e_new = np.squeeze(e_new_3dim, axis = 1)
    # print('e_new.shape:{}'.format(e_new.shape))
    return e_new
  def update(self, lr):
    pass

class Sigmoid(Operator):
  # def __init__(self, n):
  #   self.x = np.zeros(n)
  #   self.y = np.zeros(n)
  #   return
  def __init__(self):
    pass
  def forward(self, x):
    self.y = 1/(1 + np.exp(-x))
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
  def cost(self, y_, y, avg):
    pass
  @abc.abstractmethod
  def error(self, y_, y):
    pass

class LeastSquare(CostFunction):
  def cost(self, y_, y):
    J = np.mean((y_ - y) ** 2)
    return J
  def error(self, y_, y):
    e = y_ - y
    return e

class CrossEntropy(CostFunction):
  def __init__(self, one_hot = False):
    self.one_hot = one_hot
  def cost(self, y_, y):
    if self.one_hot:
      J = np.mean(np.sum(np.where(y == 1, -np.log(y_), 0), axis = 1))
      # input()
    else:
      J = np.concatenate((-np.log(y_[np.where(y == 1)]), -np.log(1 - y_[np.where(y == 0)])))
      J = np.mean(J)
    return J
  def error(self, y_, y):
    if self.one_hot:
      e = np.where(y == 1, -1 / y_, np.zeros_like(y_))
      # print(e)
      # input()
    else:
      e = -1 / y_
      e[np.where(y == 0)] = 1 / (1 - y_[np.where(y == 0)])
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
  def __init__(self, layer_list, cost_func, optimizer, one_hot = False):
    self.ll = layer_list
    self.cf = cost_func
    self.optim = optimizer
    self.one_hot = one_hot
    return
  def forward(self, x):
    for layer in self.ll:
      x = layer.forward(x)
    return x
  def backward(self, e):
    for layer in reversed(self.ll):
      e = layer.backward(e)
    return
  def update(self, lr):
    for layer in self.ll:
      layer.update(lr)
    return
  def train(self, x, y):
    y_ = self.forward(x)
    e = self.cf.error(y_, y)
    J = self.cf.cost(y_, y)
    self.backward(e)
    self.update(self.optim.step())
    return J
  def evaluate(self, x, y):
    y_ = self.forward(x)
    J = self.cf.cost(y_, y)
    return y_, J
  def predict(self, x):
    y_ = self.forward(x)
    if self.one_hot:
      y_ = np.argmax(y_, axis = 1)
    return y_

def establish_net(net_cfg, x = None):
  layer_list = []
  if net_cfg['normal']:
    x_avg, x_std = util.compute_normal_param(x)
    sub1 = SubConstant(x_avg)
    div1 = DivConstant(x_std)
    layer_list.append(sub1)
    layer_list.append(div1)
  for i in range(len(net_cfg['bone'])):
    op_param = net_cfg['bone'][i]
    if type(op_param) == type([]):
      op = Linear(op_param[0], op_param[1])
    elif type(op_param) == type(''):
      if op_param == 'sigmoid':
        op = Sigmoid()
      elif op_param == 'relu':
        op = Relu()
      elif op_param == 'tanh':
        op = Tanh()
      elif op_param == 'softmax':
        op = Softmax()
    layer_list.append(op)
  if net_cfg['cost_function'] == 'least_square':
    cost_func = LeastSquare()
  elif net_cfg['cost_function'] == 'cross_entropy':
    cost_func = CrossEntropy(one_hot = net_cfg['one_hot'])
  else:
    raise Exception('Invalid cost function')
  if net_cfg['optimizer'][0] == 'gradient_descent':
    optimizer = GradDesc(net_cfg['optimizer'][1])
  elif net_cfg['optimizer'][0] == 'decay_gradient':
    optimizer = DecayGradDesc(net_cfg['optimizer'][1], net_cfg['optimizer'][2])
  net = Net(layer_list, cost_func, optimizer, one_hot = net_cfg['one_hot'])
  return net

if __name__ == '__main__':
  cfg = util.config('config.1.json')
  one_hot = cfg['net']['one_hot']
  x = util.read_data(cfg['dataset']['x_path'])
  y = util.read_data(cfg['dataset']['y_path'])
  # x, y = x[38:44], y[38:44]
  x_test_mg, x_test = util.generate_test_data(x)
  draw = draw.Draw(x, y, x_test_mg, one_hot = one_hot, C = cfg['class'])
  net = establish_net(cfg['net'], x)
  if one_hot:
    y = util.encode_onehot(y, cfg['class'])
  sample_num = x.shape[0]
  batch_size = cfg['batch_size']
  for epoch in range(100000):
    cost_avg = 0
    for it in range(sample_num // batch_size):
      beg_idx = it * batch_size
      end_idx = sample_num if beg_idx + batch_size > sample_num else beg_idx + batch_size
      x_train = x[beg_idx:end_idx]
      y_train = y[beg_idx:end_idx]
      cost = net.train(x_train, y_train)
      # if (epoch * (sample_num // batch_size) + it) % 50 == 0:
      #   draw.drawLoss(epoch * (sample_num // batch_size) + it, cost)
      cost_avg += cost * (end_idx - beg_idx)
      y_train_ = net.predict(x_train)
      # print(y_train_.T)
    if epoch % 100 == 0:
      cost_avg /= sample_num
      draw.drawLoss(epoch, cost_avg)
      y_test = net.predict(x_test)
      draw.drawPredict(y_test)