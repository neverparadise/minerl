import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A3C_GRU(nn.Module):
    def __init__(self, num_actions, channels=3):
        super(A3C_GRU, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_size = conv2d_size_out(64, 8, 4)
        conv_size = conv2d_size_out(conv_size, 4, 2)
        conv_size = conv2d_size_out(conv_size, 3, 1)
        linear_input_size = conv_size * conv_size * 64 # 4 x 4 x 64 = 1024

        self.gru_i_dim = 64  # input dimension of LSTM
        self.gru_h_dim = 64  # output dimension of LSTM
        self.gru_N_layer = 1  # number of layers of LSTM
        self.Conv2GRU = nn.Linear(linear_input_size, self.gru_i_dim)
        self.gru = nn.GRU(input_size=self.gru_i_dim, hidden_size=self.gru_h_dim, num_layers=self.gru_N_layer, batch_first=True)
        self.fc_pi = nn.Linear(self.gru_h_dim, self.num_actions)
        self.fc_v = nn.Linear(self.gru_h_dim, 1)

    def pi(self, x, softmax_dim=1):
        x = self.fc_pi(x)
        # print(f"after pi : {x.shape}") # torch.Size([1, 1, 19])
        prob = F.softmax(x, dim=softmax_dim)
        #print(f"prob : {prob.shape}") # prob : torch.Size([1, 1, 19])
        return prob

    def v(self, x):
        v = self.fc_v(x)
        return v

    def forward(self, x, hidden, softmax_dim):
        # x : (4, 64, 64), # (batch, seq_len, input)
        self.gru.flatten_parameters()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.Conv2GRU(x))
        # print(f"After Conv2GRU : {x.shape}") # torch.Size([1, 64])
        x = x.unsqueeze(0)  #
        x, new_hidden = self.gru(x, hidden)
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        prob = F.softmax(pi, dim=softmax_dim)


        # print(f"After GRU : {x.shape}") # torch.Size([1, 1, 64])
        return prob, v, new_hidden

    def init_hidden_state(self, batch_size=1, seq_len=1, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([batch_size, seq_len, self.gru_h_dim], device=device)
        else:
            return torch.zeros([1, 1, self.gru_h_dim], device=device)