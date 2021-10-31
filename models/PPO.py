import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, num_channels, num_actions):
        self.num_actions = num_actions
        super(Model, self).__init__()

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
        self.conv_pi = nn.Linear(linear_input_size, self.num_actions)
        self.conv_value = nn.Linear(linear_input_size, 1)
        self.transitions = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # view는 numpy의 reshape 와 같다.
        return x

    def pi(self, x, softmax_dim=0):
        x = self.forward(x)
        x = self.conv_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.conv_value(x)
        return v

    def put_transition(self, transition):
        self.transitions.append(transition)

    def make_transitions(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst, prob_a_lst = [], [], [], [], [], []
        for i, transition in enumerate(self.transitions):
            s, a, r, s_prime, prob_a, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
            prob_a_lst.append([prob_a])

        s_batch = torch.cat(s_lst, dim=0)
        a_batch = torch.tensor(a_lst).to(device)
        r_batch = torch.tensor(r_lst).float().to(device)
        s_prime_batch = torch.cat(s_lst, dim=0)
        done_batch = torch.tensor(done_lst).float().to(device)
        prob_a_batch = torch.tensor(prob_a_lst).float().to(device)
        del self.transitions
        self.transitions = []

        return s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch

    def train_ppo(self, gamma, lambda_, eps_clip, k_epochs, optimizer, writer, steps):
        s, a, r, s_prime, prob_a, done_mask = self.make_transitions()

        for i in range(k_epochs):
            td_target = r + gamma * self.v(s_prime) * done_mask
            v = self.v(s)
            delta = td_target - v
            delta = delta.detach()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta.__reversed__():
                advantage = gamma * lambda_ * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float, device=device)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v, td_target.detach())
            writer.add_scalar('Loss per episodes', loss.mean().item(), steps)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
