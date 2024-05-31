# %%
# General
import numpy as np
import torch
import torchvision

# Gym
import gym

# Saving images
from collections import deque
from os import listdir
from os.path import isfile, join
import os
import sys
import gc
from tqdm import tqdm
import math
import argparse
from torchvision.transforms import InterpolationMode

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--data_amount", type=int, default=10000)
args = parser.parse_args()
max_t = args.data_amount

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
print("using:",device)

# %%
env_id = "ALE/Pong-v5"
# Create the env
env = gym.make(env_id,obs_type="grayscale",full_action_space=False)
a_size = env.action_space.n


# %%


class Saver():
    """
    Class to save the output and input to an environment as tensors which can be used to train the nn
    """
    def __init__(self,save_dir,in_transform = None, out_transform = None, small_transform = None,file_limit = 1000) -> None:
        self.save_dir = save_dir

        self.observations = deque()#maxsize=3)
        self.actions = deque()#maxsize=1)
        self.rewards = deque()#maxsize=1)
        self.dones = deque()#maxsize=1)
        self.h=210
        self.w=160
        self.file_limit = file_limit
        self.in_transform = in_transform
        self.out_transform = out_transform
        self.small_transform = small_transform

        self.j = None


        self.positional_encoding = self._make_pe()

        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    def add_transition(self,observation, action, reward, done):
        """
        Observations: What the agent sees (dimensions are 210 x 160)
        Actions: What the agent does (dimensions are 6 x 1)
        Rewards: The reward the agent gets (range is -1 to 1)?
        Dones: If the episode is over (True or False)
        """
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if len(self.observations) == 3:
            self._save_transitions()
            #clear screen of 3 screens ago.
            garbo = self.observations.popleft()
            del garbo
            
        #clear old values
        if len(self.actions) == 1:
            garbo = self.actions.popleft()
            del garbo
        if len(self.rewards) == 1:
            garbo = self.rewards.popleft()
            del garbo
        if len(self.dones) == 1:
            garbo = self.dones.popleft()
            del garbo

        #get rid of it!
        gc.collect()

    def _transform(self, input,truth):
        if self.in_transform == None or self.out_transform == None:
            return(input,truth)
        input = self.in_transform(input)
        #make the pong image bigger
        truth[0] = self.out_transform(truth[0])
        #add a small version of the pong image into index 0.
        truth = [self.small_transform(truth[0])] + truth

        return input,truth

    def _make_pe(self):
        #make a positional encoding matrix of size a_size x h x w. see https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        position = torch.arange(a_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.h, 2) * (-math.log(10000.0) / self.h))
        pe = torch.zeros(a_size, self.h)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.repeat((self.w,1,1)).permute(1,2,0)
        return pe
    
    def _encode_action(self,action):
        #expand action to a matrix the size of a pong image, and add it as a seperate channel in a tensor image later
        
        #grab the positional encoding for the action, delete the first dimension
        act = self.positional_encoding[action,:,:].squeeze(0)
        return act

    def _convert_to_tensors(self):
        if self.h == None or self.w == None:
            self.h = self.observations[0].shape[0] #height of a pong image
            self.w = self.observations[0].shape[1] #width of a pong image

        #take 2 images as input, return one image (the model should need two input images to determine ball velocity)
        #expand action to a matrix the size of a pong image, and add it as a seperate channel in a tensor image later
        act = self._encode_action(self.actions[0])
        rew = self.rewards[0]
        don = self.dones[0]


        input = np.stack((self.observations[0], self.observations[1], act), axis=0) #put the channels together so my life is easier and the nn pays a lot of attention to act
        truth = (self.observations[2],rew,don) #tuple of outputs
        
        

        input = torch.tensor(input,dtype=torch.float)
        truth = [
            torch.tensor(np.expand_dims(truth[0],axis=0),dtype=torch.float).repeat((3,1,1)),
            torch.unsqueeze(torch.tensor(truth[1],dtype=torch.float),dim=0),
            torch.unsqueeze(torch.tensor(truth[2],dtype=torch.float),dim=0),
        ]
        assert input.shape == (3,self.h,self.w)
        assert truth[0].shape == (3,self.h,self.w)
        assert truth[1].shape == (1,), "shape is {}".format(truth[1].shape)
        assert truth[2].shape == (1,)

        return input,truth

    def _save_transitions(self):
        #change bool in dones to floats
        for i in range(len(self.dones)):
            self.dones[i] = float(self.dones[i])

        if self.j == None:
            existing_files = [f for f in listdir(self.save_dir) if isfile(join(self.save_dir, f))]
            if existing_files:
                #This grabs the largest integer out of all the filenames (filter the string for digit chars, convert those chars to an int)
                self.j = max([int(''.join([c for c in f if c.isdigit()])) for f in existing_files])
            else:
                self.j = -1
        self.j += 1

        if self.j >= self.file_limit:
            print("There are at least {} files. Stopping data generator.".format(self.file_limit))
            sys.exit(0)

        input,truth = self._convert_to_tensors()

        input,truth = self._transform(input,truth)

        torch.save(input,self.save_dir+"input{i}.pt".format(i=self.j))
        torch.save((truth),self.save_dir+"truth{i}.pt".format(i=self.j))

    

# %%

save_dir = "diffusion_training_data/"
imagenet_stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
diffusion_stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

in_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
    torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    torchvision.transforms.Normalize(*imagenet_stats)
                                      ])
out_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
    torchvision.transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
    torchvision.transforms.Normalize(diffusion_stats[0], diffusion_stats[1])
                                      ])
small_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
    torchvision.transforms.Normalize(diffusion_stats[0], diffusion_stats[1])
                                      ])

# %%
def run_pong(n_episodes, max_t,saver):
    for i_episode in tqdm(range(1, n_episodes+1), position=0,leave=True, ascii=True):
        state = env.reset()[0]
        for t in range(max_t):
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            saver.add_transition(state, action, reward, terminated or truncated)
            if terminated or truncated:
                break 
        

# %%
#max_t = 1000 #maximum number of files you want
n_episodes = max(max_t//100,1) #number of episodes you want to run
saver = Saver(save_dir,in_transform=in_transform,out_transform=out_transform, small_transform=small_transform, file_limit=max_t)
run_pong(n_episodes,max_t,saver)

# %%



