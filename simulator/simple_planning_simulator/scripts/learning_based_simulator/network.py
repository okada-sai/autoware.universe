import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F

class AffineDynamicsModel(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim):
    super(AffineDynamicsModel, self).__init__()

    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    self.fc4 = nn.Linear(hidden_dim, hidden_dim)
    self.fc5 = nn.Linear(hidden_dim, output_dim)

  def __call__(self, x):
    x = T.relu(self.fc1(x))
    x = T.relu(self.fc2(x))
    x = T.relu(self.fc3(x))
    x = T.relu(self.fc4(x))
    x = self.fc5(x)
    return x

class RecurrentDynamicsModel(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, batch_first=True):
    super(RecurrentDynamicsModel, self).__init__()

    self.rnn = nn.LSTM(input_size=input_dim,
                       hidden_size=hidden_dim,
                       batch_first=batch_first)
    self.fc = nn.Linear(hidden_dim, output_dim)

    nn.init.xavier_normal_(self.rnn.weight_ih_l0)
    nn.init.orthogonal_(self.rnn.weight_hh_l0)

  def __call__(self, x):
    h, _ = self.rnn(x)
    x = h[:, -1, :]
    x = self.fc(x)
    return x
