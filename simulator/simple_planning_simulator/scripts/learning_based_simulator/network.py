import torch as T
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(Network, self).__init__()

    input_dim = input_dim
    output_dim = output_dim

    self.fc1 = nn.Linear(input_dim, 100)
    self.fc2 = nn.Linear(100, 100)
    self.fc3 = nn.Linear(100, 100)
    self.fc4 = nn.Linear(100, 100)
    self.fc5 = nn.Linear(100, output_dim)

  def __call__(self, x):
    x = T.relu(self.fc1(x))
    x = T.relu(self.fc2(x))
    x = T.relu(self.fc3(x))
    x = T.relu(self.fc4(x))
    x = self.fc5(x)
    return x
