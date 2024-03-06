# Project Outline - Pong RL-Diffusion

## 1. Introduction
### Overview
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
The diffusion model will need to output a displayed Pong image and rewards to go alongside it.
The diffusion model will take an additional embedding conveying what buttons are being pushed in each image. The agent can then supply the button presses or they can be picked randomly.
The agent will train on the data from the diffusion model, as well as its own real games of Pong.
The agent will need to take in a Pong image, rewards, and output its own button inputs to send to the Pong environment in OpenAI's Gym.

### RL Algorithm and Diffusion Model Architecture
We will use a fully connected neural network for the Pong agent. Or maybe I'll find some more complex one and use that.
The diffusion model will use a diffusion model from a diffusion tutorial.
All models will be implemented using PyTorch or TensorFlow, but I haven't decided which yet.

## 5. Implementation
### Steps
???

### Programming Languages, Frameworks, and Libraries
Python, OpenAI Gym, TensorFlow or PyTorch.

## 6. Evaluation
### Performance Evaluation
We will compare how many rounds of training it takes to create an agent on real Pong with the same average number of game wins as one trained on both the diffusion model and real Pong.
We will also compare this with an agent trained by overfitting (training on game results multiple times without playing new games) on the same number of real pong games.

### Metrics and Criteria for Success
Making an agent model trained on both the diffusion model and real Pong which performs better than a model trained using the same amount of real Pong iterations but no diffusion model interactions.

## 7. Results and Analysis
### Results
Present the results obtained from the Pong RL-Diffusion implementation

### Analysis
Analyze and interpret the results

## 8. Conclusion
### Summary
Summarize the project and its outcomes

### Limitations and Future Improvements
Discuss any limitations or future improvements

## 9. References
List any references or resources used in the project
