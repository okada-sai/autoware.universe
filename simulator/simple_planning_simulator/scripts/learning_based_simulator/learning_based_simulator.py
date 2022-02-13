import os
import copy
import matplotlib.pyplot as plt

import torch

from simple_planning_simulator.trainer import Trainer
from network import AffineDynamicsModel, RecurrentDynamicsModel

# common functions
def flatten(lst):
  return [x for row in lst for x in row]

def get_index(lst, val):
  if val in lst:
    return lst.index(val)
  return False

def set_val(data_vec, input_idx, val):
  if input_idx:
    data_vec[input_idx] = val
  return  data_vec

def rostime_to_sec(t):
  return t / 1000000000.0

def filter_noise(values_vec, v_idx, num_lpf):
  target_value_vec = copy.deepcopy(values_vec[v_idx])

  begin_idx = int(num_lpf / 2.0)
  end_idx = int(len(target_value_vec) - num_lpf / 2.0) - 1

  # apply low-pass filter
  for t_idx in range(len(target_value_vec)):
    if t_idx < begin_idx or end_idx < t_idx:
      continue

    lpf_value = 0.0
    for diff_idx in range(int(-num_lpf / 2), int(num_lpf / 2), 1):
      lpf_value += target_value_vec[t_idx + diff_idx] / num_lpf

    values_vec[v_idx][t_idx] = lpf_value

  # extract from begin_idx to end_idx
  for v_idx in range(len(values_vec)):
    values_vec[v_idx] = values_vec[v_idx][begin_idx:end_idx]

  return values_vec

def filter_time(values_vec, rostime_idx, start_time):
  rostime_vec = values_vec[rostime_idx]

  for t_start_idx in range(len(rostime_vec)):
    if rostime_to_sec(rostime_vec[t_start_idx] - rostime_vec[0]) > start_time:
      break

  filtered_values_vec = [value_vec[t_start_idx:] for value_vec in values_vec]
  return filtered_values_vec

def load_log(log_path, idx_vec):
  values_vec = [[] for i in range(len(idx_vec))]
  with open(log_path, 'r') as lines:
    for l_idx, line in enumerate(lines):
      elements = line.split(', ')

      for v_idx in idx_vec:
        values_vec[v_idx].append(float(elements[v_idx]))

  return values_vec

def generate_data(rostime_vec, input_values_vec, output_values_vec):
  time_vec = []
  data_vec = []
  label_vec = []
  for t_idx in range(len(rostime_vec)):
    if t_idx == 0:
      continue

    rostime = rostime_vec[t_idx]
    data = [input_value_vec[t_idx - 1] for input_value_vec in input_values_vec]
    label = [output_value_vec[t_idx] for output_value_vec in output_values_vec]

    time_vec.append(rostime)
    data_vec.append(data)
    label_vec.append(label)

  return time_vec, data_vec, label_vec

def generate_lstm_data(rostime_vec, input_values_vec, output_values_vec, sequence_length):
  time_vec = []
  data_vec = []
  label_vec = []
  for t_idx in range(len(rostime_vec)):
    if t_idx + sequence_length == len(rostime_vec) - 1:
      break

    rostime = rostime_vec[t_idx]
    data = [[input_value_vec[t_idx + diff_t_idx] for input_value_vec in input_values_vec] for diff_t_idx in range(sequence_length)]
    label = [output_value_vec[t_idx + sequence_length] for output_value_vec in output_values_vec]

    time_vec.append(rostime)
    data_vec.append(data)
    label_vec.append(label)

  return time_vec, data_vec, label_vec

class LearningBasedSimulator:
  def __init__(self, model_path='', use_lstm=True):
    self.use_lstm = use_lstm
    self.root_path = os.path.dirname(os.path.abspath(__file__)) + '/../../'

    # param for LSTM
    self.sequence_length = 4

    # 100ms data
    # row index in log file
    self.rostime_idx = 0
    self.vel_idx = 1
    self.acc_idx = 2
    self.pitch_comp_acc_idx = 3
    self.final_acc_idx = 4
    self.throttle_idx = 5
    self.brake_idx = 6
    self.throttle_speed = 7
    self.brake_speed = 8
    self.pitch = 9
    self.steer_idx = 10
    # self.omega_idx

    # define input and output
    self.input_idx_vec = [self.vel_idx, self.throttle_idx, self.brake_idx]
    # self.input_idx_vec = [self.vel_idx, self.throttle_idx, self.brake_idx, self.steer_idx]
    # self.input_idx_vec = [self.vel_idx, self.acc_idx, self.throttle_idx, self.brake_idx, self.steer_idx]

    self.output_idx_vec = [self.acc_idx]
    # self.output_idx_vec = [self.pitch_comp_acc_idx] # [self.final_acc_idx]


    # define network
    input_dim = len(self.input_idx_vec)
    output_dim = len(self.output_idx_vec)
    hidden_dim = 8 if use_lstm else 100
    self.net = RecurrentDynamicsModel(input_dim, output_dim, hidden_dim) if use_lstm else AffineDynamicsModel(input_dim, output_dim, hidden_dim)

    if model_path != '':
      self.net.load_state_dict(torch.load(self.root_path + model_path))

    self.idx_vec = [self.rostime_idx, self.vel_idx, self.acc_idx, self.pitch_comp_acc_idx, self.final_acc_idx, \
      self.throttle_idx,  self.brake_idx, self.throttle_speed, self.brake_speed, self.pitch, self.steer_idx]

  def train_offline(self, log_path, output_model_path, epoch, batch_size, test_size):
    # instantiate trainer
    self.trainer = Trainer(self.root_path, self.net, output_model_path)

    # load each value vectors
    values_vec = load_log(self.root_path + log_path, self.idx_vec)

    rostime_vec = values_vec[self.rostime_idx]
    input_values_vec = [values_vec[input_idx] for input_idx in self.input_idx_vec]
    output_values_vec = [values_vec[output_idx] for output_idx in self.output_idx_vec]

    # filter noise
    values_vec = filter_noise(values_vec, self.acc_idx, 6)

    # generate data
    rostime_vec, data, label = generate_lstm_data(rostime_vec, input_values_vec, output_values_vec, self.sequence_length) \
        if self.use_lstm else generate_data(rostime_vec, input_values_vec, output_values_vec)

    # train
    self.trainer.train(data, label, epoch, batch_size, test_size)

  def analysis_log(self, log_path, value):
    # load each value vectors
    values_vec = load_log(self.root_path + log_path, self.idx_vec)

    # calculate target_idx
    if value == 'acc':
      target_idx = self.acc_idx
    elif value == 'vel':
      target_idx = self.vel_idx
    elif value == 'throttle':
      target_idx = self.throttle_idx
    elif value == 'brake':
      target_idx = self.brake_idx
    elif value == 'steer':
      target_idx = self.steer_idx
    else:
      target_idx = self.acc_idx

    # extract rostime/value vec before filtering
    rostime_vec = copy.deepcopy(values_vec[self.rostime_idx])
    value_vec = copy.deepcopy(values_vec[target_idx])

    # filter noise
    lpf_values_vec = filter_noise(values_vec, target_idx, 20)

    # extract time and value to analysis
    lpf_rostime_vec = lpf_values_vec[self.rostime_idx]
    lpf_value_vec = lpf_values_vec[target_idx]

    # calcualte relative time [s]
    relative_rostime_vec = [rostime_to_sec(t - rostime_vec[0]) for t in rostime_vec]
    lpf_relative_rostime_vec = [rostime_to_sec(t - rostime_vec[0]) for t in lpf_rostime_vec]

    # plot
    fig = plt.figure()
    plt.plot(lpf_relative_rostime_vec, lpf_value_vec, marker="o")
    plt.plot(relative_rostime_vec, value_vec, marker="o")
    plt.show()

  def predict(self, x):
    x = torch.tensor(x, dtype=torch.float)
    return self.net(x)

  def predict_offline(self, log_path, start_time):
    # load each value vectors
    values_vec = load_log(self.root_path + log_path, self.idx_vec)

    # filter time
    values_vec = filter_time(values_vec, self.rostime_idx, start_time)

    # extract time and input/output values
    rostime_vec = values_vec[self.rostime_idx]
    input_values_vec = [values_vec[input_idx] for input_idx in self.input_idx_vec]
    output_values_vec = [values_vec[output_idx] for output_idx in self.output_idx_vec]

    # generate data
    rostime_vec, data_vec, _ = generate_lstm_data(rostime_vec, input_values_vec, output_values_vec, self.sequence_length) \
        if self.use_lstm else generate_data(rostime_vec, input_values_vec, output_values_vec)

    # filter noise
    # values_vec = filter_noise(values_vec, self.acc_idx, 6)

    # assign initial values to current ones
    current_vel = copy.deepcopy(values_vec[self.vel_idx][0])
    current_acc = copy.deepcopy(values_vec[self.acc_idx][0])

    relative_time_vec = [rostime_to_sec(t - rostime_vec[0]) for t in rostime_vec]

    input_vel_idx = get_index(self.input_idx_vec, self.vel_idx)
    input_acc_idx = get_index(self.input_idx_vec, self.acc_idx)

    # predict recursively
    output_log_path = self.root_path + 'log/predict.log'
    with open(output_log_path, 'w') as f:
      for d_idx in range(len(data_vec)):
        if d_idx == len(data_vec) - 1:
          break

        # if data_vec has vel/acc information, assign current ones instead of true ones
        if d_idx != 0:
          if self.use_lstm:
            for t in range(self.sequence_length - 1): # 0, 1 range(3 - 1)
              target_time = self.sequence_length - 1 - t # 2, 1
              # 2 -> 1, 1 -> 0
              set_val(data_vec[d_idx][target_time - 1], input_vel_idx, data_vec[d_idx - 1][target_time][input_vel_idx])
              set_val(data_vec[d_idx][target_time - 1], input_acc_idx, data_vec[d_idx - 1][target_time][input_acc_idx])
            # current -> 2
            set_val(data_vec[d_idx][-1], input_vel_idx, current_vel)
            set_val(data_vec[d_idx][-1], input_acc_idx, current_acc)
          else:
            set_val(data_vec[d_idx], input_vel_idx, current_vel)
            set_val(data_vec[d_idx], input_acc_idx, current_acc)

        diff_time = rostime_to_sec(rostime_vec[d_idx + 1] - rostime_vec[d_idx])
        current_acc = self.predict([data_vec[d_idx]])[0].item()
        current_vel += current_acc * diff_time

        true_vel = copy.deepcopy(values_vec[self.vel_idx][d_idx + 1])
        true_acc = copy.deepcopy(values_vec[self.acc_idx][d_idx + 1])
        f.write('{} {} {} {} {}\n'.format(relative_time_vec[d_idx], true_vel, true_acc, current_vel, current_acc))

  def predict_online(self):
    # pre-process data
    x = torch.tensor(x, dtype=torch.float32)

    # predict
    self.predict(x)
