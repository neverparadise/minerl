import argparse
import os
import yaml

import models.PPO
from pre_train import *
from train import *
from evaluate import *
from agent import *
import gym
import minerl
from models import *

def run():
    parser = argparse.ArgumentParser(description='argparse for running')
    parser.add_argument('--model_type', type=str, default="policy", help='model_type: value or policy')
    parser.add_argument('--model_name', type=str, default="PPO", help='model_type: value_based or policy_based')
    parser.add_argument('--config_path', type=str, default="./config/navigate.yaml", help='config file path')
    parser.add_argument('--pre_training', type=bool, default=False, help='option for pre training')
    parser.add_argument('--training', type=bool, default=True, help='option for training or evaluation')
    parsed_args = parser.parse_args()

    model_type = True if parsed_args.model_type == "value" else False
    model_name = parsed_args.model_name
    config_path = parsed_args.config_path
    pre_training = parsed_args.pre_training
    training = parsed_args.training

    with open(config_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        args['model_name'] = model_name
        print(args)
    global model
    if model_name == "PPO":
        model = models.PPO.Model(args['num_channels'], args['num_actions'])
        model = model.cuda()
    env = gym.make(args['env_name'])
    agent = Agent(model_type, model, env, args)

    if pre_training:
        pre_train()
    if training:
        agent.explore(training)
    else:
        agent.explore(training)



run()