import os,argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.distributions import Categorical

from models.A2C.A2C import *
from models.A2C.monte_carlo_a2c import Trainer
from models.A2C.update_util import *

from env.market import Market

#import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pytorch Actor Critic')
parser.add_argument('--market', type=str, default='C:/Users/sari/Desktop/DeepCover/sample_1.csv',
                    help='path to market data csv')
parser.add_argument('--max_episode', type=int, default=1000, help='num of training episodes')
parser.add_argument('--seed', type=int, default=8,
                    help='random seed (default: 8)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount rate (default: 0.99)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--tau', type=float, default=0.1,
                    help='tau, (default0.1)')

args = parser.parse_args()

#wandb.init(project='deepcover_test_run')


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

S_DIM = 4
A_DIM = 1
A_BOUND = 100000000000 #100000万

trainer = Trainer(S_DIM,A_DIM,A_BOUND, args.lr)

#ToDo: Save final trajectory
#ToDo: reset env to initial state at start of every episode
#ToDo: initial action production -> train.get_initial_action

def normalize_obs(obs, obs_len):
    for i in range(obs_len):
        pass

def main():
    market = Market(args.market)
    for _ep in range(args.max_episode):
        print(f'episode num: {_ep+1}')
        obs = market.reset()
        #divide exposure by absolute max/min value
        obs[0] = obs[0]/100000000
        #subtract mean rate values
        obs[1] = (obs[1]-139.531634539)/10
        obs[2] = (obs[2]-139.526564539)/10
        obs[3] = (obs[3]-139.53555673759)/10
        terminated = False
        #ep_reward = 0
        #ep_i_rewards = []
        zero_action = False

        while terminated == False:
            action = trainer.get_exploitation_action(obs)
            action = action * A_BOUND
            if zero_action == True:
                action = torch.tensor(action)
                action = torch.zeros_like(action)
            next_obs, reward, terminated, reward_list = market.step(action)
            if reward < 0:
                reward = reward * 10
            if terminated:
                print(f'terminated at episode{_ep+1}')
                print(f'reward for ep {_ep} if {reward}')
                #wandb.log({'Reward(PL) throughout epoch': sum(reward_list)})
                reward = 0
            #divide by absolute max/min value
            next_obs[0] = next_obs[0]/100000000
            #subtract mean rate values
            next_obs[1] = (next_obs[1]-139.531634539)/10
            next_obs[2] = (next_obs[2]-139.526564539)/10
            next_obs[3] = (next_obs[3]-139.53555673759)/10
            reward = reward / 100000000000
            if reward == 0:
                print(f'At episode {_ep} and timestep {market.clock} reward is 0!!!!!!!! ')
            action = action / A_BOUND
            #print(f'clock: {market.clock}')
            if (action - 1) < 0:
                print(f'[obs], [action], [reward] at timestep {market.clock}: {obs} {action}, {reward}')
                print(f'ternimated:{terminated}')
            #if market.clock %10 == 0:
            #    print(f'action at timestep {market.clock} is {action}')
            #    print(f'reward at timestep {market.clock} is {reward * 100000000000}')
            #    print(f'sum of reward at timestep {market.clock} is {sum(reward_list)}')
            #print(f'obs and obs.dtype:{obs, type(obs)}',f'action and dtype:{action, action.dtype}')
            loss_critic, loss_actor = trainer.optimize(obs, action, reward, next_obs, args.gamma, args.tau)
            #for parameters in trainer.actor.parameters():
            #    print(parameters)
            #if t%50==0:
                #print(f'reward for step {t} is {reward}')
                #print(f"obs for step {t} is {obs}")
                #print(f"action for step{t} if {action}")
                #print(f'Step:{t}', f'loss_critic:{loss_critic}', f'loss_actor:{loss_actor}')
            obs = next_obs
            #for params in trainer.actor.parameters():
            #    print(f'params of Actors are: {params}')
            #if _ep%3000 == 0:
            #    trainer.save_model(t)
            #if terminated and _ep%100==0:
            #    trainer.save_model(_ep)
            #if terminated:
            #    wandb.log({'Terminater reward': reward})
            #    break
            #wandb.log({'Reward(PL) throughout epoch': reward})
            #wandb.log({'critic_loss': loss_critic})
            #wandb.log({'actor_loss': loss_actor})
            
            
if __name__ == '__main__':
    main()