import math
import numpy as np
import pandas as pd

#swap following two lines' comments after testing
#from utils import *
from env.utils import *


class Market:
    """
    cover_upper_bound : take action "cover_" when accumulative quantatiy hits cover_upper_bound
    cover_lower_bound : take action "cover_" when accumulative quantity hits cover_lower_bound
    self.pl_client : profit of customer at each time step
    self.pl_cover : amount of profit when agent(dealer) takes cover action
    """
    
    def __init__(self, filepath):
        self.df = read_csv_to_df(filepath)
        self.data = df_to_data(self.df)
        self.exposure = list(self.data[0])#/100000000)
        self.rate = list(self.data[1])#-139.531634539)
        self.sell_rate = list(self.data[2])#-139.526564539)
        self.buy_rate = list(self.data[3])#-139.53555673759)
        self.episode_length = len(self.data[0]-1)
        #print(f'self.episode_length: {len(self.data[0]-1)}')
        self.reward = [0]
        self.cover = [0]
        self.clock = 0
        #self.done = False
        
    #agent takes the step, i.e. take action to interact with environment
    def step(self, action):
        """
        takes action and returns obs, reward, terminated
        action: float, positive -> sell, negative -> buy, zero -> hold
        obs: list containing [exposure, rate, sell_rate, buy_rate]
        return : Total_pl(reward) at time(num_step)
        terminated: boolean value, when num_steps==max_episode_length, return True
        """ 
        #print(f'self.clock: {self.clock}')
        #if self.clock ==1410:
        #    print(f'self.exposure[1410]: {self.exposure[1410]}')
        exposure = self.exposure[self.clock]
        rate = self.rate[self.clock]
        sell_rate = self.sell_rate[self.clock]
        buy_rate = self.buy_rate[self.clock]
        obs = [exposure, rate, sell_rate, buy_rate]  
        reward = 0
        #step >= 1
        if self.clock > 0:
            #"cover_buy"
            if action < 0:      
                reward = (self.reward[(self.clock -1)] + 
                      (rate - self.rate[self.clock -1]) * (self.exposure[(self.clock -1)] + sum(self.cover)) +
                      (rate - buy_rate) * action)
                self.cover.append(action)
                self.reward.append(reward)
                #print('case 1 executed')
            #"cover_sell"
            elif action > 0:    
                reward =  (self.reward[(self.clock -1)] + 
                      (rate - self.rate[self.clock -1]) * (self.exposure[(self.clock -1)] + sum(self.cover)) +
                      (rate - sell_rate) * action)
                self.cover.append(action)
                self.reward.append(reward)
                #print('case 2 executed')
            #"hold"
            else:               
                reward = (self.reward[(self.clock -1)] + 
                      (rate - self.rate[self.clock -1]) * (self.exposure[(self.clock -1)] + sum(self.cover)))
                self.cover.append(action)
                self.reward.append(reward)
                #print('case 3 executed')
        #first step        
        else:
            reward = self.reward[0]
            print('case 4 executed')
            #dont append, self.reward is initialized as =[0]
            #self.reward.append(reward)


        #check for termination
        terminated = bool(self.clock +1 == self.episode_length)
        #update clock and move to next time step
        self.clock += 1

        return obs, reward, terminated, self.reward


    def reset(self):
        """
        reset env at start of new episode
        """
        #self.done = False
        self.reward = [0]
        self.cover = [0]
        self.clock = 0
        obs = [self.exposure[0],self.rate[0],self.sell_rate[0],self.buy_rate[0]]
        print(f'------------reset obs----------------:{obs}')
        return obs
    
    def sample_action(self):
        action = np.random.randint(-5000,5000)
        return action