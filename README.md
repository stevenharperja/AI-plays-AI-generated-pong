# Project Outline - AI Plays: AI-Generated Pong

## 1. Introduction
### Overview
TLDR: Make a fake version of pong using an image generator network's output. Run an AI agent on that fake pong.

This project is to test how well an image generating model can help an AI actor play Pong in as few sessions of real Pong as possible.
 
Keywords: Few-shot Learning, Diffusion Networks, Reinforcement Learning

### Process
We start by having an AI agent play a few games of pong. Afterward, we use the screen data, input data, and reward data seen during those games to train a diffusion model.
This diffusion model then creates scenes of "virtual pong" which the AI agent then trains on.
We repeat this a number of times and compare the results of the training vs an AI agent which only trained on "real pong".

## 2. Goals and Objectives
### Project Goals
The goals of this project are to show how well this technique works to improve few-shot learning, what the drawbacks are, and roughly how many training sessions are needed to compare to a model trained on many shots.

### Why I chose this project
I hope to show a proof of concept that could be extended to reinforcement learning in other fields such as robotics or other more complex games than Pong.
This technique can be and already is applied in other forms of machine learning, and I want to demonstrate it with Pong because I like video games.  
I hope to learn more about diffusion models and Reinforcement learning techniques by doing this project.

## 3. Background
### Pong
Pong is a two-dimensional sports game that simulates table tennis.
The player controls an in-game paddle by moving it vertically across the left or right side of the screen.
They can compete against another player controlling a second paddle on the opposing side.
Players use the paddles to hit a ball back and forth.
The goal is for each player to reach eleven points before the opponent;
points are earned when one fails to return the ball to the other.
[Wikipedia](https://en.wikipedia.org/wiki/Pong)  

![Pong Game](https://upload.wikimedia.org/wikipedia/commons/6/62/Pong_Game_Test2.gif)

We will be using Pong and assigning a +1 reward to the agent whenever it scores a point, and -1 reward when the opponent scores a point.
We will use OpenAI's Gym library to do this.

### Reinforcement Learning (RL)
Reinforcement learning is the general term used to describe using machine learning to make a model which deals with time-based situations while making decisions to navigate an environment.
Examples are self-driving cars, robotics, playing video games, etc.
The main difficulties of reinforcement learning are making a "good" model from a generally lacking amount of data, making models which are safe around humans, and can generalize to multiple tasks, etc.

### Diffusion Networks
A diffusion network takes a set of random noise and tries to predict what an image must have looked like before noise was added to it.
It is trained by adding noise to images from the target dataset.
But at runtime, it is given completely random noise which it uses to make completely new images.
It has been a popular type of model for image generation in recent years. So I thought it would be good to use it.

## 4. Methodology
### Approach
We will start by making the Pong agent and testing it playing Pong. 
Then we will make the diffusion model. 
The diffusion model will be trained on the Pong scenes and will act as a recurrent neural network, taking the previous image it generated as an embedding for the next image. 
The embedding which trackes image history will use exponential decay to record information from the last few frames.:
    To do it, divide the previous embedding by 2, and add the last image we produced to it (with a max value of 1 in the matrix). this will allow the model to have information beyond just the last frame
The diffusion model will need to output a displayed Pong image and rewards to go alongside it. 
The diffusion model will take an additional embedding conveying what buttons are being pushed in each image. The agent can then supply the button presses or they can be picked randomly. 
The agent will train on the data from the diffusion model, as well as its own real games of Pong. 
The agent will need to take in a Pong image, rewards, and output its own button inputs to send to the Pong environment in OpenAI's Gym. 
The agent will just see the difference between one frame and the next, and diffusion will only create the difference between one frame and the next like in http://karpathy.github.io/2016/05/31/rl/ 

### RL Algorithm and Diffusion Model Architecture
We will use a fully connected neural network for the Pong agent. Or maybe I'll find some more complex one and use that.
The diffusion model will use a diffusion model from a diffusion tutorial.
All models will be implemented using PyTorch or TensorFlow, but I haven't decided which yet.

## 5. Implementation
### Steps
1.
Create an agent model using TensorFlow or Pytorch and have it play pong using OpenAI Gym.
    i. Use a convolutional layer followed by 2 fully connected layers, output a positive or negative number for up/down on the controller.
        a. Theres not much particular thought behind this architecture its just off the top of my head.
2.
Create graph functions to see how well the model does as we train it.
    i. Use seaborn 
3.
Create a diffusion model from a prebuilt implementation. figure out how to implement the embeddings for controls and recurrent image generation.
    i. 
    notes:
    use pytorch implementation?
    throw embeddings as diffusion input?
    ii. Embeddings
        a. Instead of gaussian noise put the previous frame + the input. or maybe concatinate?
        b. Show controller input by adding a +1 or -1 to the whole input image.
    iii. Structure
        a. Run the model as a recurrent neural net, putting its output frames into itself as input frames.
        b. Use the distance from each sequential frame of pong with the same controller inputs, as the error. 
        c. The model will need to output rewards for the agent as well. 
            i. take the embedding created in the middle of the U-net, and feed it into some layers which outputs a +1, 0, or -1
4.
Train the diffusion model using the same few games of pong.
    i. Recurrency
        a. Use the distance from each sequential frame of pong with the same controller inputs, as the error. 
    ii. How much training data is needed?
        a. no idea. but I'll try to use as little as possible, but we can compare how much is needed.
    iii. Use transfer learning?
5.
Hook up the agent to play with the diffusion model as if it were OpenAI Gym.
    i. The diffusion model will need to be able to generate a new frame when given a new input from the agent.


### Programming Languages, Frameworks, and Libraries
Python, OpenAI Gym, TensorFlow or PyTorch.

## 6. Evaluation
### Performance Evaluation
We will compare how many rounds of training it takes to create an agent on real Pong with the same average number of game wins as one trained on both the diffusion model and real Pong.
We will also compare this with an agent trained by overfitting (training on game results multiple times without playing new games) on the same number of real pong games.

### Metrics and Criteria for Success
Making an agent model trained on both the diffusion model and real Pong which performs better than a model trained using the same amount of real Pong iterations but no diffusion model interactions.
Even if the diffusion model ends up taking longer to run than an instance of Pong, it is still worthwhile because when this technique is extended to more difficult games or to real-life scenarios, running a diffusion model can be cheaper/faster than running that game or potentially losing a robot. 

## 7. Results and Analysis
### Results
Present the results obtained

### Analysis
Analyze and interpret the results

## 8. Conclusion
### Summary
Summarize the project and its outcomes

### Limitations and Future Improvements
Discuss any limitations or future improvements

The techniques used with both the Diffusion and AI agent likely don't let the models have a lot of "memory" of previous frames, so when extended to more complex environments it would be better to change the model architectures. The Diffusion model architecture should be changed away from a recurrent architecture in favor of something like a transformer. This would make it so the agent couldn't play in real time but i think there is likely some work around possible. 

## 9. References
List any references or resources used in the project
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
    -https://m.youtube.com/watch?v=NhdzGfB1q74 (U-net)
- Pong RL
    - http://karpathy.github.io/2016/05/31/rl/ (full code implementation of Pong AI)
    - https://huggingface.co/blog/deep-rl-pg (policy gradient implementation in PyTorch)
    - https://colab.research.google.com/github/huggingface/deep-rl-class/blob/master/notebooks/unit4/unit4.ipynb#scrollTo=gfGJNZBUP7Vn (notebook file with policy model introduction)
- PyTorch
    - https://pytorch.org/tutorials/beginner/basics/intro.html (beginner tutorial)
    
