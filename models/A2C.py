import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A2C(nn.Module):
    def __init__(self, num_actions, channels):
        super(A2C, self).__init__()
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

        self.fc_pi = nn.Linear(linear_input_size, self.num_actions)
        self.fc_v = nn.Linear(linear_input_size, 1)

    def pi(self, x, softmax_dim=1):
        x = self.fc_pi(x)
        # print(f"after pi : {x.shape}") # torch.Size([1, 1, 19])
        prob = F.softmax(x, dim=softmax_dim)
        #print(f"prob : {prob.shape}") # prob : torch.Size([1, 1, 19])
        return prob

    def v(self, x):
        v = self.fc_v(x)
        return v

    def forward(self, x, softmax_dim):
        # x : (4, 64, 64), # (batch, seq_len, input)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        pi = self.fc_pi(x)
        v = self.fc_v(x)
        prob = F.softmax(pi, dim=softmax_dim)

        return prob, v