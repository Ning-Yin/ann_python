import json
import pandas as pd
import numpy as np
import torch

def config(path):
  cfg = json.load(open(path))
  return cfg

def read_data(path, use_torch = False):
  res = np.array(pd.read_csv(path, sep = '\s+', engine = 'python', header = None)).astype(np.float32)
  if use_torch:
    res = torch.from_numpy(res)
  print('read: {}, shape: {}'.format(path, res.shape))
  return res

def compute_normal_param(x, use_torch = False):
  if use_torch:
    avg = torch.mean(x, dim = 0, keepdim = True)
    std = torch.mean((x - avg) ** 2, dim = 0, keepdim = True) ** 0.5
  else:
    avg = np.mean(x, axis = 0, keepdims = True)
    std = np.mean((x - avg) ** 2, axis = 0, keepdims = True) ** 0.5
  print('avg_shape: {}, std_shape: {}'.format(avg.shape, std.shape))
  return avg, std

def generate_test_data(x_train, use_torch = False):
  if use_torch:
    x_train = x_train.numpy()
  x1_min, x1_max = np.min(x_train[:, 0]), np.max(x_train[:, 0])
  x2_min, x2_max = np.min(x_train[:, 1]), np.max(x_train[:, 1])
  padding = max(x1_max - x1_min, x2_max - x2_min) / 10
  step = min(x1_max - x1_min, x2_max - x2_min) / 1000
  x1_test = np.arange(x1_min - padding, x1_max + padding, step, dtype = np.float32)
  x2_test = np.arange(x2_min - padding, x2_max + padding, step, dtype = np.float32)
  x1_test_mg, x2_test_mg = np.meshgrid(x1_test, x2_test)
  x1_test = np.reshape(x1_test_mg, (-1, 1))
  x2_test = np.reshape(x2_test_mg, (-1, 1))
  x_test = np.concatenate((x1_test, x2_test), 1)
  if use_torch:
    x_test = torch.from_numpy(x_test)
  x_test_mg = (x1_test_mg, x2_test_mg)
  return x_test_mg, x_test

def encode_onehot(label, C):
  label_onehot = np.eye(C)[label.astype(np.int)].squeeze()
  return label_onehot

def accuracy(y_, y):
  pass

def shuffle_data(x, y):
  idx = np.random.permutation(x.shape[0])
  x = x[idx, :]
  y = y[idx, :]
  return x, y

if __name__ == '__main__':
  path = 'ex4x.dat'
  data = read_data(path)
  data_torch = torch.from_numpy(data)
  data_np = data_torch.numpy()
  print(data_torch)