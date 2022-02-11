import torch
import torch.nn as nn
import torch.optim as optim

try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

class Trainer:
  def __init__(self, root_path, net, output_model_path=''):
    self.root_path = root_path
    self.net = net
    self.output_model_path = output_model_path

  def train(self, data, label, epoch, batch_size, test_size):
    # split to train/test data
    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=test_size)

    # transform to tensor
    train_data = torch.tensor(train_data, dtype=torch.float)
    train_label = torch.tensor(train_label, dtype=torch.float)
    test_data = torch.tensor(test_data, dtype=torch.float)
    test_label = torch.tensor(test_label, dtype=torch.float)

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_label), batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_label), batch_size=batch_size, shuffle=False)

    # setup
    optimizer = optim.Adam(self.net.parameters())
    MSE = nn.MSELoss()

    # train
    train_loss = []
    test_loss = []
    for e in range(epoch):
      print('{}/{}'.format(e + 1, epoch))

      sum_train_loss = 0
      train_data_num = 0
      self.net.train()
      for batch_idx, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # train
        y = self.net(data)
        loss = MSE(y, label)
        loss.backward()
        optimizer.step()

        # calculate loss for training
        sum_train_loss += float(loss.detach()) * len(data)
        train_data_num += len(data)

      # print(sum_train_loss / len(train_dataloader.dataset), train_data_num, len(train_dataloader.dataset))
      train_loss.append(sum_train_loss / len(train_dataloader.dataset))

      # cauclate loss for testing
      sum_test_loss = 0
      test_data_num = 0
      self.net.eval()
      for batch_idx, (data, label) in enumerate(test_dataloader):
        y = self.net(data)
        loss = MSE(y, label)
        loss.backward()

        # calculate loss
        sum_test_loss += float(loss.detach()) * len(data)
        test_data_num += len(data)

      # print(sum_test_loss / len(test_dataloader.dataset), test_data_num, len(test_dataloader.dataset))
      test_loss.append(sum_test_loss / len(test_dataloader.dataset))

      print(sum_train_loss / len(train_dataloader.dataset), sum_test_loss / len(test_dataloader.dataset))

      # save plot image
      if e > 0 and e % 10 == 0:
        fig = plt.figure()

        # save figure of loss train and test
        plt.plot(train_loss)
        plt.plot(test_loss)
        plot_file_name = self.root_path + 'log/train_test_plot_epoch{}.png'.format(e)
        fig.savefig(plot_file_name)

    # save network weight
    if self.output_model_path == '':
      self.output_model_path = self.root_path + 'model/model.pt'
    torch.save(self.net.state_dict(), self.output_model_path)

    # plot train/loss test
    plt.clf()
    fig = plt.figure()

    plt.plot(train_loss)
    plot_file_name = self.root_path + 'log/train_plot.png'
    fig.savefig(plot_file_name)

    plt.plot(test_loss)
    plot_file_name = self.root_path + 'log/train_test_plot.png'
    fig.savefig(plot_file_name)

    plt.show()
