import gym
import minerl
from utils.make_actions import *
from utils.converter import *
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os

class Agent:
    def __init__(self, is_value_based, model, env, args):
        # Initialize model and environment
        self.policy = model
        if is_value_based:
            self.target_policy = model
            self.target_policy.load_state_dict(self.policy.state_dict())
        self.env = env
        self.env.make_interactive(realtime=False, port=6666)
        # Optimizer and save path
        self.model_path = os.curdir + args['model_path']
        self.model_name = args['model_name']
        self.save_path = self.model_path+self.model_name
        print(self.save_path)
        # Hyperparams for training
        self.lr = args['lr']
        self.gamma = args['gamma']
        self.lambda_ = args['lambda_']
        self.eps_clip = args['eps_clip']
        self.k_epochs = args['k_epochs']
        self.env_name = args['env_name']
        self.weights_decay = args['weight_decay']
        self.optimizer = optim.Adam(self.policy.parameters(), self.lr, weight_decay=self.weights_decay)
        self.max_epi = args['max_epi']
        self.T_horizon = args['T_horizon']

        # Summary
        self.writer = SummaryWriter('runs/' + self.model_name + "_" + self.env_name)

        if args['num_actions'] == 6:
            self.make_action = make_6action
        elif args['num_actions'] == 9:
            self.make_action = make_9action
        elif args['num_actions'] == 19:
            self.make_action = make_19action

    def explore(self, is_training=True):
        score = 0.0
        for n_epi in range(self.max_epi):
            state = self.env.reset()
            obs = converter(self.env_name, state)
            done = False
            steps = 0
            while not done:
                for t in range(self.T_horizon):
                    steps += 1
                    prob = self.policy.pi(obs, softmax_dim=1)
                    m = Categorical(prob)
                    a = m.sample().item()
                    action = self.make_action(self.env, a)
                    s_prime, r, done, info = self.env.step(action)
                    obs_prime = converter(self.env_name, s_prime)
                    self.policy.put_transition((obs, a, r / 100.0, s_prime, prob[0][a].item(), done))
                    s = s_prime
                    score += r
                    if done:
                        break
                if is_training:
                    self.policy.train_ppo(self.gamma, self.lambda_, self.eps_clip, self.k_epochs, self.optimizer, self.writer, steps)

            torch.save(self.policy, self.save_path + '.pth')
            self.writer.add_scalar('Rewards per episodes', score, n_epi)
            print("# of episode :{}, score : {:.1f}".format(n_epi, score))
            score = 0.0
        self.env.close()
