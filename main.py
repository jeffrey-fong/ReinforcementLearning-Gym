import argparse

import numpy as np
import torch
import gym

from algos.dqn_algo import Algorithm
from models.dqn_model import Model
from utils.replay_buffer import ReplayBuffer


def game_loop(args):
    env = gym.make(args.env)
    replay_buffer = ReplayBuffer(obs_shape=env.observation_space.shape,
                                     action_shape=(env.action_space.n,),
                                     reward_shape=(1,),
                                     capacity=10000,
                                     batch_size=50,
                                     device=args.device)
    model = Model(action_shape=(env.action_space.n,)).to(args.device)
    model.eval()
    algo = Algorithm()

    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        # Convert action to one-hot
        action_one_hot = np.zeros((env.action_space.n,))
        action_one_hot[action] = 1
        action = action_one_hot
        replay_buffer.add(observation, action, action, reward, done)

        if done:
            observation = env.reset()
    env.close()


def main():
    argparser = argparse.ArgumentParser(description="Reinforcement Learning Algorithms for Gym")
    argparser.add_argument('--env',
                           default='LunarLander-v2',
                           type=str,
                           help='Specify the environment for the RL algorithm')
    argparser.add_argument('--render',
                           default=True,
                           type=bool,
                           help='Render display of gym environment view')
    argparser.add_argument('--device',
                           help='GPU device',
                           default=0,
                           type=int)
    argparser.add_argument('--max_episode_count',
                           help='Max count for each data collection phase',
                           default=50,
                           type=int)
    args = argparser.parse_args()
    args.device = torch.device('cuda', args.device) if args.device != 'cpu' else torch.device('cpu')

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()