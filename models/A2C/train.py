import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np
import math

from models.A2C.update_util import soft_update, hard_update
from models.A2C.A2C import *

#LEARNING_RATE = 0.001
#GAMMA = 0.99
#TAU = 0.001

#ToDo: Move Tensors to GPU

class Trainer:

    def __init__(self, state_dim, action_dim, action_bound, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.iter = 0
        #actor
        self.actor = Actor(self.state_dim, self.action_dim)
        self.target_actor = Actor(self.state_dim, self.action_dim)
        self.actor.apply(weights_init_)
        self.target_actor.apply(weights_init_)
        for param in self.actor.parameters():
            param.requires_grad=True
        for param in self.target_actor.parameters():
            param.requires_grad=True
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), learning_rate)
        #critic
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic.apply(weights_init_)
        self.target_critic.apply(weights_init_)
        for param in self.critic.parameters():
            param.requires_grad=True
        for param in self.target_critic.parameters():
            param.requires_grad=True
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), learning_rate)
        #update
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_action(self, state):
        #state: list of floats
        state = torch.tensor(state)
        #print(f'state input to actor is:{state}')
        #ToDo: Check data type of action and modify return values
        action = self.actor.forward(state).detach()
        #print(f'action is:{action}')
        return action.data.numpy()

    def get_exploitation_action(self, state):
        state = torch.tensor(state).clone().detach()
        #print(f'sssssssssssssssssssssssssssssssssstate:{state}')
        action = self.actor.forward(state).detach()
        #print(f'aaaaaaaaction: {action}')
        #for params in self.actor.parameters():
        #    print(f'params of Actors are: {params}')
        return action.data.numpy()
    
#Normalized: s1, s2 \in [-1,1]
#Un-normalized: a1([-1,1] when devided by action bound(1000000)), r1(normalized by div by 10e+9)
    def optimize(self, s1, a1, r1, s2, gamma, tau):
        """
        perform on-policy optimization
        """
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #ToDo: Tensor.to(device)
        s1 = torch.tensor(s1, requires_grad=True)
        a1 = torch.tensor(a1, requires_grad=True, dtype=torch.float32)
        r1 = torch.tensor(r1, requires_grad=True)
        s2 = torch.tensor(s2, requires_grad=True)
        #print(f'input data is state: {s1}, action: {a1}, reward: {r1}, next_state: {s2}')

        # -------------------optimize critic-------------------
        # y_pred = Q(s1,a1)
        y_predicted = torch.reshape(self.critic.forward(s1, a1), (1,-1))
        #print(f'y_predicted_data_type:{type(y_predicted)}')
        print(f'y_predicted_val:{y_predicted}')
        #print(f'y_predicted_dim:{y_predicted.type()}')
        # Use target actor exploitation policy here for loss evaluation(?)
        a2 = self.target_actor.forward(s2).detach()
        next_val = torch.squeeze(self.target_critic.forward(s2,a2)).detach()
        #y_exp = r + gamma*Q(s2,pi(s2))

        #noramlize r1 so that y_expected matches y_predicted
        #print(f'r1: {r1}, next_val: {next_val}')
        y_expected = (r1 + gamma*next_val)
        #print(f'r1:{r1}')
        print(f"y_expected_val:{y_expected}")
        #print(f'y_expected_shape:{y_expected.shape}')
        #compute critic loss, and update critic
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        loss_critic = 0.5 * (y_expected - y_predicted) ** 2 
        print(f'loss_critic:{loss_critic}')
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # -------------------optimize actor-------------------
        pred_a1 = self.actor.forward(s1)
        #print(f'pred_a1: {pred_a1}')
        ############### This is where to focus #################
        loss_actor = -1 * (self.critic.forward(s1,pred_a1)) 
        print(f'loss_actor: {loss_actor}')
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        soft_update(self.target_actor, self.actor, tau)
        soft_update(self.target_critic, self.critic, tau)

        return loss_critic, loss_actor
                
    def save_model(self, episode_count):
        save_dir = R"C:\Users\sari\Desktop\DeepCover\runs"
        os.makedirs(save_dir, exist_ok=True)

        actor_path = os.path.join(save_dir, f'{episode_count}_actor.pt')
        critic_path = os.path.join(save_dir, f'{episode_count}_critic.pt')

        torch.save(self.target_actor.state_dict(), actor_path)
        torch.save(self.target_critic.state_dict(), critic_path)
        print('----------Target Models Saved Successfully----------')

    def load_model(self, load_dir, episode):
        """
        loads the target actor and critic models, and copies them onto actor anc critic models
        :param episode: episode of model to load
        :return: loaded actor and critic models
        """
        actor_dir = os.path.join(load_dir, f'{episode}_actor.py')
        critic_dir = os.path.join(load_dir, f'{episode}_critic.py')
        self.actor.load_state_dict(torch.load(actor_dir))
        self.critic.load_state_dict(torch.load(critic_dir))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('----------Models Loaded Succesfully----------')