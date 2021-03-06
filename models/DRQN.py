import torch
import torch.nn as nn
import torch.nn.functional as F
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DRQN(nn.Module):
    def __init__(self, num_channels, num_actions, batch_first=True):
        self.num_actions = num_actions
        super(DRQN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(64, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)
        linear_input_size = convw * convw * 64

        self.gru_i_dim = 64  # input dimension of GRU
        self.gru_h_dim = 64  # output dimension of GRU
        self.gru_N_layer = 1  # number of layers of GRU
        self.Conv2GRU = nn.Linear(linear_input_size, self.gru_i_dim)
        self.gru = nn.GRU(input_size=self.gru_i_dim,
                          hidden_size=self.gru_h_dim,
                          num_layers=self.gru_N_layer,
                          batch_first=batch_first)
        self.head = nn.Linear(self.gru_h_dim, self.num_actions)

    def forward(self, x, hidden):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.Conv2GRU(x))
        x = x.unsqueeze(0)  #
        x, new_hidden = self.gru(x, hidden)
        x = F.relu(self.head(x))
        return x, new_hidden

    def init_hidden_state(self, batch_first, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if not batch_first:
            if training is True:
                return torch.zeros([1, batch_size, self.gru_h_dim], device=device)
            else:
                return torch.zeros([1, 1, self.gru_h_dim], device=device)
        else:
            if training is True:
                return torch.zeros([batch_size, 1, self.gru_h_dim], device=device)
            else:
                return torch.zeros([1, 1, self.gru_h_dim], device=device)

    def sample_action(self, obs, epsilon, hidden):
        out, hidden = self.forward(obs, hidden)

        coin = random.random()
        if coin < epsilon:
            return random.randint(0,self.num_actions-1), hidden
        else:
            #print(out)
            return torch.argmax(out), hidden

