import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions import Categorical

def weights_init_(m):
    if isinstance(m, nn.Linear):
        #print('-----------weights are initialized---------------')
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        #self.action_bound = action_bound

        self.ln1 = nn.Linear(self.state_dim, 16)
        self.ln2 = nn.Linear(16, 32)
        self.ln3 = nn.Linear(32, 32)
        self.ln4 = nn.Linear(32, 16)
        self.ln5 = nn.Linear(16, 1)

    def forward(self, state):
        #print(f'x:{state}')
        x = F.tanh(self.ln1(state))
        #print(f'actor x_1: {x}')
        x = F.tanh(self.ln2(x))
        #print(f'actor x_2: {x}')
        x = F.tanh(self.ln3(x))
        #print(f'actor x_3: {x}')
        x = F.tanh(self.ln4(x))
        #print(f'actor x_4: {x}')
        action = F.tanh(self.ln5(x))
        #print(f'actor action output: {action}')

        #action = action * self.action_bound
        #print(f'Final action output:{action}')
        #action = action.squeeze(1)

        #print(f'action:{action}')
        return action
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state(int)
        :param action_dim: Dimension of input action(int)
        """
        super().__init__()
        self.state_dim=state_dim
        self.action_dim=action_dim

        self.fcs1 = nn.Linear(self.state_dim, 16)
        self.fcs2 = nn.Linear(16, 32)
        self.fcs3 = nn.Linear(32, 16)
        self.fca = nn.Linear(self.action_dim, 16)
        #print(f'self.fca data type:{self.fca.weight.dtype}')
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)


    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Tensor)
        :param action: Input action (int)
        :return: Value function : Q(s,a) (Torch Variable : [n,1])
        """

        #action = torch.tensor(action, dtype=torch.float32)
        #print(f'state:{state}', f'action:{action}', f'data_types:{state.type(), action.type()}')
        s1 = F.tanh(self.fcs1(state))
        #print(f'critic s1:{s1}')
        s2 = F.tanh(self.fcs2(s1))
        #print(f"critic s2: {s2}")
        s3 = F.tanh(self.fcs3(s2))
        #print(f'critic s3:{s3}')
        s3 = s3.unsqueeze(0)
        s3 = s3.view(1,-1)
        #print(f'shape of critic latent state: {s3.shape}')
        
        #unbound action
        a = F.tanh(self.fca(action))
        #print(f'critic action: {a}')
        a = a.unsqueeze(0)
        a = a.view(1,-1)
        #print(f'postprocessed critic action: {a}')
        #print(f'critic action:{a}')
        #print(f'shape of critic latent action: {a.shape}')
        
        x = torch.cat((s3,a),dim=1)
        #print(f'concatenated critic latent vector: {x}')
        x = F.tanh(self.fc1(x))
        #print(f'latent state before value: {x}')
        value = F.tanh(self.fc2(x))
        #print(f'critic value:{value}')
        return value
