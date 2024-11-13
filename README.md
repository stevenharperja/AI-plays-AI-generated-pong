
> TLDR: 
<br> I tried to a fake version of Pong using an image generator model's output.
It didn't produce results but it was a good learning experience.
## 1. Introduction
### Overview

The goal of the project was to create an emulated version of Pong using an image generator neural network. Then run a reinforcement learning agent on those generated images.  
The idea being that a reinforcement learning agent could be trained without ever touching the original source environment. 

## 2. Background
### Pong
Pong is a two-dimensional sports game that simulates table tennis.  
The player controls an in-game paddle by moving it vertically across the left or right side of the screen.  
They can compete against another player controlling a second paddle on the opposing side.  
Players use the paddles to hit a ball back and forth.  
The goal is for each player to reach eleven points before the opponent;  
points are earned when one fails to return the ball to the other.  
[Wikipedia](https://en.wikipedia.org/wiki/Pong)   

![Gif of a Pong game](https://upload.wikimedia.org/wikipedia/commons/6/62/Pong_Game_Test2.gif)  


### Reinforcement Learning (RL)
Reinforcement learning is the general term used to describe using machine learning to make a model which deals with time-based situations while making decisions to navigate an environment.
Examples are self-driving cars, robotics, playing video games, etc.
The main difficulties of reinforcement learning are making a "good" model from a generally lacking amount of data, making models which are safe around humans, and can generalize to multiple tasks, etc.

### Diffusion Networks
A diffusion network takes a set of random noise and tries to predict what an image must have looked like before noise was added to it.
It is trained by adding noise to images from the target dataset.
But at runtime, it is given completely random noise which it uses to make completely new images.
It has been a popular type of model for image generation in recent years. So I thought it would be good to use it.

## 3. Implementation

For simplicity, and in order to learn a famous architecture, I chose to implement the generator network using a Diffusion network. This consisted of a UNet and adapted the model code from [here](https://github.com/dome272/Diffusion-Models-pytorch), and also retrained weights provided there as well. the model uses 64 by 64 images, I chose it because it was the only pretrained one I could find that was small enough to fit on my GPU reliably.

The agent was never implemented because I was unable to get a good generator working, and ran out of time to finish the project by my self-imposed deadline.

## 4. Conclusion
The project had many hiccups and every time I thought I solved the problems with the generator, it still didn't work. I had to fix the data a lot.
The best results I got were a white jpg.
Since 

## 5. Bibliography
- About diffusion networks
    - https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/ (mathy explanation along with code walk through) <- very nice
        - https://kikaben.com/up-sampling-with-transposed-convolution/ (what is up-convolution)
        - https://www.analyticsvidhya.com/blog/2022/10/image-segmentation-with-u-net/ (Unet?)
    - https://github.com/lucidrains/denoising-diffusion-pytorch (pytorch implementation)(has links to youtube videos along with lots more info) <- very nice
        - https://www.youtube.com/watch?v=344w5h24-h8 (diffusion overview)
            - https://arxiv.org/abs/2112.10741 (text embedding)
            - https://huggingface.co/spaces/valhalla/glide-text2im (text embedding)
    - https://github.com/hojonathanho/diffusion (tensorflow implemenation)
    - https://medium.com/@kemalpiro/step-by-step-visual-introduction-to-diffusion-models-235942d2f15c (more info on Diffusion models)
    - https://m.youtube.com/watch?v=NhdzGfB1q74 (U-net)
    - https://keras.io/examples/generative/ddim/ (more explanation on ddim)
- Pong RL
    - http://karpathy.github.io/2016/05/31/rl/ (full code implementation of Pong AI)
    - https://huggingface.co/blog/deep-rl-pg (policy gradient implementation in PyTorch)
    - https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit4/unit4.ipynb#scrollTo=gfGJNZBUP7Vn (notebook file with policy model introduction)
- PyTorch
    - https://pytorch.org/tutorials/beginner/basics/intro.html (beginner tutorial)
- Video generation model
    - https://blog.marvik.ai/2024/01/30/diffusion-models-for-video-generation/ (overview of recent video generation models)
        - https://paperswithcode.com/method/vision-transformer (what is it?)
        - https://doubiiu.github.io/projects/Make-Your-Video/ (might use)
    - https://machine-learning-made-simple.medium.com/what-are-the-different-types-of-transformers-in-ai-5085275664e8 (Whats an autoregressive transformer?)

- Unused:
    - https://video-diffusion.github.io/ (they use an approach which requires processing set amounts of frames. which probably doesnt work here)
    
- Embedder model implementation
    - https://pytorch.org/vision/0.18/models/generated/torchvision.models.resnet50.html (code)
- diffusion model implementation
    - https://github.com/explainingai-code/StableDiffusion-PyTorch (potential explanation of how to do embeddings for diffusion models?)
    - https://wandb.ai/capecape/train_sd/reports/How-To-Train-a-Conditional-Diffusion-Model-From-Scratch--VmlldzoyNzIzNTQ1 (paper explaining conditional diffusion models VERY GOOD)
        - https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/modules.py (code for the modules used in the nn)
            - https://github.com/dome272/Diffusion-Models-pytorch (source of that code)
            - https://www.baeldung.com/cs/gelu-activation-function (Gelu explanation)
    - https://github.com/VSehwag/minimal-diffusion (smaller model?)
- https://nightcafe.studio/blogs/info/understanding-ai-model-checkpoints-a-simplified-guide (checkpoints)
- Upscaler model
    - https://paperswithcode.com/paper/image-super-resolution-using-deep
    - https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py

- really understanding diffusion models ddpm
    - [explains how variance works](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
