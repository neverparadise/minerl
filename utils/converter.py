import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def converter(env_name, observation):
    if (env_name == 'MineRLNavigateDense-v0' or
            env_name == 'MineRLNavigate-v0'):
        obs = observation['pov']
        obs = obs / 255.0  # [64, 64, 3]
        compass_angle = observation['compassAngle']
        compass_angle_scale = 180
        compass_scaled = compass_angle / compass_angle_scale
        compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
        obs = np.concatenate([obs, compass_channel], axis=-1)
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        if(len(obs.shape) < 4):
            obs = obs.unsqueeze(0).to(device=device)
        return obs.float() # return (1, 4, 64, 64)
    else:
        obs = observation['pov']
        obs = obs / 255.0
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        if(len(obs.shape) < 4):
            obs = obs.unsqueeze(0).to(device=device)
        return obs.float() # return (1, 4, 64, 64)

def converter_for_pretrain(env_name, pov, compassAngle=None):
    if (env_name == 'MineRLNavigateDense-v0' or
            env_name == 'MineRLNavigate-v0'):
        obs = pov
        obs = obs / 255.0  # [64, 64, 3]
        compass_angle = compassAngle
        compass_angle_scale = 180
        compass_scaled = compass_angle / compass_angle_scale
        compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
        obs = np.concatenate([obs, compass_channel], axis=-1)
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        return obs.float()
    else:
        obs = pov
        obs = obs / 255.0
        obs = torch.from_numpy(obs)
        obs = obs.permute(2, 0, 1)
        return obs.float()
