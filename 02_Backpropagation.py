import math
import random

# Activations

class Identity:
  def __init__(self):
    self.diff_value = 1

  def __call__(self, x):
    return x

class Sigmoid:
  def __init__(self):
    self.diff_value = None

  def __call__(self, x):
    value = 1 / (1 + math.exp(-x))
    self.diff_value = value * (1-value)
    return value
  
class ReLU:
  def __init__(self):
    self.diff_value = None
    
  def __call__(self, x):
    value = max(0, x)
    self.diff_value = 1 if x > 0 else 0
    return value

class TanH:
  def __init__(self):
    self.diff_value = None
    
  def __call__(self, x):
    value = math.tanh(x)
    self.diff_value = 1 - value**2
    return value

# Loss functions

class MSE:
  def __init__(self):
    self.diff_value = None

  def set_d(self, d):
    self.d = d

  def __call__(self, y, d):
    self.diff_value = y-d
    return 0.5*(y-d)**2

# Weight initializers

def zero_init():
  return 0

class random_init:
  def __init__(self, min_value, max_value):
    self.min_value = min_value
    self.max_value = max_value

  def __call__(self):
    return random.uniform(self.min_value, self.max_value)

# Network elements

class Input:
  def __init__(self, value=None):
    self.forward_value = value

  def set_x(self, x):
    self.forward_value = x


class Neuron:
  def __init__(self, activation_fn=None, weight_init_fn=None, use_bias=False):
    self.activation_fn = activation_fn
    self.weight_init_fn = weight_init_fn
    self.use_bias = use_bias
    
    self.weights = None
    self.idx = None
    self.fan_in = None
    self.fan_out = None

    self.backprop_value = None
    self.forward_value = None

  def forward(self):
    linear_out = sum(w * n.forward_value for w, n in zip(self.weights, self.fan_in))
    self.forward_value = self.activation_fn(linear_out)
    return self.forward_value

  def backprop(self, learning_rate, loss_diff=None):
    if self.fan_out is None:
      self.backprop_value = self.activation_fn.diff_value * loss_diff
    else:
      self.backprop_value = sum(n.weights[self.idx] * n.backprop_value for n in self.fan_out) * self.activation_fn.diff_value

    for i in range(len(self.weights)):
      self.weights[i] -= learning_rate * self.fan_in[i].forward_value * self.backprop_value
      
  def connect_to_prev(self, idx, prev_layer):
    self.idx = idx
    if self.use_bias:
      self.fan_in = prev_layer + [Input(1.)]
    else:
      self.fan_in = prev_layer
    self.weights = [self.weight_init_fn() for _ in range(len(self.fan_in))]


class Net:
  def __init__(self, layers, input_dim):
    self.inputs = [Input() for _ in range(input_dim)]
    self.layers = layers
    for prev_layer, next_layer in zip([self.inputs] + self.layers[:-1], self.layers):
      for n in prev_layer:
        n.fan_out = next_layer
      for idx, n in enumerate(next_layer):
        n.connect_to_prev(idx, prev_layer)

  def __call__(self, x):
    for inp, x_value in zip(self.inputs, x):
      inp.set_x(x_value)
    for layer in self.layers:
      for neuron in layer:
        neuron.forward()
    return [neuron.forward_value for neuron in self.layers[-1]] 

  def gradient_descent(self, loss_diff, lr=1e-3):
    it = iter(reversed(self.layers))
    last_layer = next(it)
    for neuron in last_layer:
      neuron.backprop(lr, loss_diff)
    for layer in it:
      for neuron in layer:
        neuron.backprop(lr)


init = random_init(-0.2, 0.2)
net = Net([
  [Neuron(activation_fn=TanH(), weight_init_fn=init, use_bias=True) for _ in range(50)],
  [Neuron(activation_fn=TanH(), weight_init_fn=init, use_bias=True) for _ in range(50)],
  [Neuron(activation_fn=Identity(), weight_init_fn=init, use_bias=True)]
], 1)

import matplotlib.pyplot as plt

def linspace(min_val, max_val, step):
  return [min_val + x / step * (max_val - min_val) for x in range(step + 1)]

function_to_learn = lambda x: math.cos(x)

real_x = linspace(-2 * math.pi, 2 * math.pi, 50)
real_y = [function_to_learn(x) for x in real_x]

train_x = [random.uniform(-math.pi, math.pi) for _ in range(50)]
train_y = [function_to_learn(x) + random.gauss(0, 0.1) for x in train_x]

predict_x = linspace(-2 * math.pi, 2 * math.pi, 50)

def draw_plot(full_pause=False):
  predict_y = [net([x])[0] for x in predict_x]
  plt.clf()
  plt.plot(real_x, real_y, train_x, train_y, 'ro', predict_x, predict_y, 'g--')
  plt.draw()
  if full_pause:
    plt.show()
  else:
    plt.pause(0.01)

draw_plot()

def learning_rate(epoch, max_epoch, max_lr, min_lr):
  if epoch < .05 * max_epoch:
    return max_lr / 8
  if epoch < .10 * max_epoch:
    return max_lr / 4
  if epoch < .15 * max_epoch:
    return max_lr / 2
  if epoch < .20 * max_epoch:
    return max_lr
  return max_lr - (epoch - .20 * max_epoch)/(.80 * max_epoch)*(max_lr - min_lr)

loss = MSE()

from tqdm import tqdm

max_epoch = 100
with tqdm(range(max_epoch)) as t:
  for epoch in t:
    train_samples = list(zip(train_x, train_y))
    random.shuffle(train_samples)
    total_loss = 0
    for batch_idx, (x, d) in enumerate(train_samples):
      y = net([x])[0]
      total_loss += loss(y, d)
      net.gradient_descent(loss.diff_value, learning_rate(epoch, max_epoch, 1e-2, 1e-4))
      if batch_idx % 10 == 0:
        draw_plot()
    t.set_postfix({'loss': total_loss / len(train_samples)})

draw_plot(True)


