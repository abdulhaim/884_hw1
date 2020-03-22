import gym
from PPO import PPO, Memory
from PIL import Image
import torch
from reacher import ReacherEnv
from pusher import PusherEnv
from reacher_wall import ReacherWallEnv
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
import imageio
import argparse

parser = argparse.ArgumentParser(description="Evaluate")
parser.add_argument('--env', default='ReacherEnv')

def make_env(env_name):
    if(env_name=="ReacherEnv"):
        env = ReacherEnv(render=False)
    elif(env_name=="ReacherWallEnv"):
        env = ReacherWallEnv(render=False)
    else:
        env = PusherEnv(render=False)
    return env

def createGif(env_name):
    png_dir = "./gif/" + env_name + "/"

    images = []
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(env_name +'.gif', images)

def test(env_name):
    ############## Hyperparameters ##############
    env = make_env(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    n_episodes = 3          # num of episodes to run
    max_timesteps = 1500    # max timesteps in one episode
    render = True           # render the environment
    save_gif = True        # png images are saved in gif folder
    
    # filename and directory to load model from
    filename = "PPO_continuous_" +env_name+ ".pth"
    directory = "./preTrained/"

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    ppo.policy_old.load_state_dict(torch.load(directory + filename))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray((img * 255).astype(np.uint8))
                 img.save('./gif/' + env_name + '/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()

    createGif(env_name)
    
if __name__ == '__main__':
    args = parser.parse_args()
    test(args.env)
    
    