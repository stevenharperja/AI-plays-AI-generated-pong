#general
import torch
import torch.nn as nn
import numpy as np

import conditional_diffusion.utils as utils

# for model construction
import conditional_diffusion.modules as modules 
from torchvision.models import resnet18
# for using the model
from tqdm import tqdm 
from conditional_diffusion.ddpm_conditional import Diffusion as Diffusion 
# For dataloader
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
#for training
import torch.optim as optim
import torchvision
from conditional_diffusion.ddpm_conditional import Diffusion
from conditional_diffusion.ddpm_conditional import SummaryWriter
from conditional_diffusion.ddpm_conditional import plot_images
from conditional_diffusion.ddpm_conditional import save_images  
from conditional_diffusion.modules import EMA
import os
import logging
import copy
import argparse

# pass argument to enable ema during training (costs more memory but may improve training)
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--use_ema", type=bool, default=False)
parser.add_argument("-b", "--batch_size", type=int, default=8)
parser.add_argument("-e", "--epochs", type=int, default=300)
args = parser.parse_args()
use_ema = args.use_ema
batch_size = args.batch_size
num_epochs = args.epochs

assert(torch.cuda.is_available(), "no cuda found! you cant run this model without a graphics card connected!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "diffusion_training_data/"
# %%
class Net(nn.Module):
    def __init__(self, device, embedding_scale = 3):
        super(Net, self).__init__()
        self.embedding_scale = embedding_scale

        resnet = resnet18(weights="IMAGENET1K_V1")
        resnet.fc = nn.Identity()
        #freeze the weights for resnet
        for param in resnet.parameters():
            param.requires_grad = False

        ###init and modify diffusion model
        diffusion_model = modules.UNet_conditional(num_classes=10).to(device)
        ckpt = torch.load("models/conditional_ema_ckpt.pt", map_location=device)
        diffusion_model.load_state_dict(ckpt)
        #remove nn.embedder and just send an embedding instead
        diffusion_model.label_emb = nn.Identity()
        #freeze the weights for diffusion model
        for param in diffusion_model.parameters():
            param.requires_grad = False
        #unfreeze all of the diffusion model's decoder layers
        diffusion_model.up1.requires_grad = True
        diffusion_model.sa4.requires_grad = True
        diffusion_model.up2.requires_grad = True
        diffusion_model.sa5.requires_grad = True
        diffusion_model.up3.requires_grad = True #unfreeze just the last 3 layers of diffusion model
        diffusion_model.sa6.requires_grad = True
        diffusion_model.outc.requires_grad = True

        ###init upscaler model
        upscaler = nn.Sequential(
            nn.Upsample(size=(224,224), mode="nearest"),
            # modules.DoubleConv(3,3,residual=True),
            # modules.DoubleConv(3,3,residual=True)
        )

        #input of (N, 3, 224, 224), output of (N, 256)
        self.encoder = nn.Sequential(
            resnet,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0), #Using a layer which isnt trainable so I can reduce the size while keeping it frozen.
        )
        
        self.diffusion_model = diffusion_model #input of (N,256) output of (N,3,64,64)
        self.upscaler = upscaler
        self.final_layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0) # reduce channel size to 1 for black and white
        #input of (N, 256), output of (N, 1)
        self.reward_maker = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
        )
        #input of (N, 256), output of (N, 1)
        self.done_maker = nn.Sequential(
            nn.Linear(in_features=256, out_features=1),
        )


        ###variables for the diffusion model
        self.noise_steps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.device = device
        self.zero_embedding = torch.zeros((1,256)).to(self.device) 


    #prediction function. this may not work for training. but maybe since its pretrained it might? work
    def diffusion_sample(self, embedding): # see https://github.com/stevenharperja/conditional_diffusion/blob/main/conditional_diffusion/ddpm_conditional.py#L41
        n = embedding.size()[0]
        x = torch.randn((n, 3, 64, 64)).to(self.device)
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0, ascii=True):
            t = (torch.ones(n) * i).long().to(self.device)
            predicted_noise = self.diffusion_model(x, t, embedding)
            if self.embedding_scale > 0:
                uncond_predicted_noise = self.diffusion_model(x, t, None)
                predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, self.embedding_scale)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x


    def forward(self, x, t = None, noised_truth = None, use_embedding = True):
        """
        
        x: The input image (n,3,224,224)
        t: The timesteps vector (n)
        noised_truth: a noised version of the target image, used for training (n,3,64,64)
        use_embedding: Whether to use the embedding or not, useful for training faster?

        
        returns:
        if training:
            it returns a predicted noise, and an upscaled predicted noise, along with a predicted reward and done
        if not training
            it returns a predicted image, along with a reward and done.
        
        """
        n = x.size()[0]
        embedding = self.zero_embedding.repeat((n,1))
        if (not self.training) or use_embedding:
            embedding = self.encoder(x) #(n,256)       
        assert embedding.size() == (n,256)
        unscaled_image = None
        if self.training:
            if noised_truth == None:
                raise Exception("Cannot train without a provided noised target image")
            if t == None:
                raise Exception("Cannot train without a provided timesteps vector")
            unscaled_image = self.diffusion_model(noised_truth,t,embedding) #(n,3,64,64)
        else:
            unscaled_image = self.diffusion_sample(embedding) #(n,3,64,64) 
        
        image = self.upscaler(unscaled_image)
        # print(image)
        # assert(False)
        image = self.final_layer(image)#(n,1,224,224)
        if not self.training:
            image = (image.clamp(-1, 1) + 1) / 2
            image = (image * 255).type(torch.uint8)
        rew = self.done_maker(embedding) #(n,1)
        don = self.reward_maker(embedding) #(n,1)
        if self.training:
            return unscaled_image,image,rew,don
        else:
            return image,rew,don




# %%

class PongDataset(Dataset):
    def __init__(self, dir, device):
        self.dir = dir
        self.device = device
    def __len__(self):
        existing_files = [f for f in listdir(self.dir) if isfile(join(self.dir, f))]
        if existing_files:
            #This grabs the largest integer out of all the filenames (filter the string for digit chars, convert those chars to an int)
            i = max(*[int(''.join([i for i in f if i.isdigit()])) for f in existing_files])
            return i
        else:
            return 0
    def __getitem__(self, index):

        #use data which was preformatted for the 'trivial' model
        input = torch.load(self.dir+"input{i}.pt".format(i=index)).to(self.device)
        truth = torch.load(self.dir+"truth{i}.pt".format(i=index))
        truth = tuple([t.to(self.device)for t in truth])
        assert len(truth) == 4, "length is {}".format(len(truth))
        assert input.shape == (3,224,224)
        assert truth[0].shape == (3,64,64)
        assert truth[1].shape == (3,224,224), "shape is {}".format(truth[1].shape)
        assert truth[2].shape == (1,)
        assert truth[3].shape == (1,)

        return input, truth 

# %%
trainset = PongDataset(save_dir,device)
if __name__ == '__main__':
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)#, num_workers=1)
    testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

# %%
# Create an instance of the network
print("Creating model")
net = Net(device).to(device)
if os.path.exists("models/Pong_Generator/ckpt.pt"):
    print("Loading model from file")
    ckpt = torch.load("models/Pong_Generator/ckpt.pt", map_location=device)
    net.load_state_dict(ckpt)
    # print("removing conv layers")
    # net.upscaler = nn.Upsample(size=(224,224), mode="bilinear", align_corners=True)

ema = None
ema_model = None
if use_ema:
    ema = EMA(0.995)
    if os.path.exists("models/Pong_Generator/ema_ckpt.pt"):
        print("Loading EMA model from file")
        ckpt = torch.load("models/Pong_Generator/ema_ckpt.pt", map_location=device)
        ema_model = Net(device).to(device)
        ema_model.load_state_dict(ckpt)
        # print("removing conv layers")
        # net.upscaler = nn.Upsample(size=(224,224), mode="bilinear", align_corners=True)
    else:
        ema_model = copy.deepcopy(net).eval().requires_grad_(False)
    
# %%
#see https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# define loss function and optimizer


small_img_criterion = nn.MSELoss()
img_criterion = nn.MSELoss()
rew_criterion = nn.MSELoss()
don_criterion = nn.MSELoss()
optimizer = optim.AdamW(net.parameters(), lr=3e-4)

if os.path.exists("models/Pong_Generator/optim.pt"):
    print("loading optimizer from file")
    optimizer.load_state_dict(torch.load("models/Pong_Generator/optim.pt"))

image_importance = 10 #hyperparameter for weighting how important the image is in the loss function.


# %%


beta_start=1e-4
beta_end=0.02
noise_steps = 1000
beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)
def noise_given_noise(x,t,noise):
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    Ɛ = noise
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


small_transform = torchvision.transforms.Resize((64, 64))
big_transform = torchvision.transforms.Resize((224,224))


# %%


run_name = "Pong_Generator"


diffusion = Diffusion(img_size=64, device=device)
logger = SummaryWriter(os.path.join("runs", run_name))


print("checking most recent epoch for starting point")

epoch_offset = 0
image_dir = os.path.join("results", run_name)
if os.path.exists(image_dir):
    existing_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    if existing_files:
        #This grabs the largest integer out of all the filenames (filter the string for digit chars, convert those chars to an int)
        epoch_offset = max([int(''.join([c for c in f if c.isdigit()])) for f in existing_files]) + 1


def sample(input,epoch):
    net.eval()
    #make only 3 images or a max of batch size
    num_samples = min(3,batch_size)
    sampled_images = net.forward(input[:num_samples,:,:,:])[0]
    print(sampled_images)
    #plot_images(sampled_images)
    os.makedirs("results/{}".format(run_name), exist_ok = True)
    save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))
    os.makedirs("models/{}".format(run_name), exist_ok = True)
    torch.save(net.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
    torch.save(optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
    if use_ema:
        ema_sampled_images = ema_model(input[0].unsqueeze(0))[0]
        save_images(ema_sampled_images, os.path.join("results", run_name, f"{epoch}_ema.jpg"))
        torch.save(ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
    net.train()

l = len(trainloader)
print("Starting training")
net.train()
ema_model.eval() #always have the ema model in eval mode
for epoch in range(epoch_offset,num_epochs+epoch_offset):  # loop over the dataset multiple times

    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(trainloader, position=0,leave=True, ascii=True)

    input = None
    for i, data in enumerate(pbar, 0):
        #print(i)

        input, truth = data
        if i ==0:
            sample(input,epoch-1)
        small_images = truth[0] #64 by 64 image
        images = truth[1] #224 by 224 image
        assert(small_images.size() == (batch_size,3,64,64), "size is " + str(small_images.size()))


        
        t = diffusion.sample_timesteps(small_images.shape[0]).to(device) #get <batch size> number of t's
        x_t, small_noise = diffusion.noise_images(small_images, t) #TODO change the random noise to not be reallocated in memory so that the time complexity is less.

        noise = big_transform(small_noise)#sadly i dont think this can be quickened

        use_embedding = True
        if np.random.random() < 0.1: #randomly dont use embedding to generate an image.
            use_embedding = False
        #forward
        small_predicted_noise, predicted_noise, rew, don = net.forward(input, t = t, noised_truth = x_t, use_embedding = use_embedding)

        # print(predicted_noise)
        # assert False
        loss0 = small_img_criterion(small_noise,small_predicted_noise)
        #loss1 = img_criterion(noise,predicted_noise)
        loss2 = rew_criterion(rew,truth[2])
        loss3 = don_criterion(don,truth[3])
        loss = loss0 + loss2 + loss3

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_ema:
            ema.step_ema(ema_model, net)

        pbar.set_postfix(MSE=loss.item())
        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

    if epoch % 10 == 0 or epoch == num_epochs - 1+epoch_offset:
        sample(input,epoch)

print('Finished Training')
