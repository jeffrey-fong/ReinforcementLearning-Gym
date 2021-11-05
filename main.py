import argparse
import random
import numpy as np
import torch
import gym
from collections import deque

from algos.dqn_algo import Algorithm
from models.dqn_model import Model
from utils.replay_buffer import ReplayBuffer


def game_loop(args):
    env = gym.make(args.env)
    env.seed(0)
    replay_buffer = ReplayBuffer(obs_shape=env.observation_space.shape,
                                     action_shape=(env.action_space.n,),
                                     reward_shape=(1,),
                                     capacity=10000,
                                     batch_size=64,
                                     device=args.device)
    model = Model(action_shape=(env.action_space.n,),
                  obs_shape=env.observation_space.shape,
                  device=args.device).to(args.device)
    model.eval()
    algo = Algorithm(model=model, device=args.device)

    eps = 1.0
    eps_decay = 0.99
    episode_count = 0
    count = 0
    train_every_count = 0
    itr = 0
    total_reward = 0.0
    avg_reward = deque(maxlen=100)

    obs = env.reset()
    while True:
        if args.render:
            env.render()

        if random.uniform(0, 1) < eps:
            action_one_hot = np.zeros(shape=(env.action_space.n,))
            action_one_hot[random.randint(0, env.action_space.n - 1)] = 1.0
            action = np.argmax(action_one_hot)
        else:
            with torch.no_grad():
                action_one_hot = model.policy(torch.tensor(obs).to(args.device))
                action_one_hot = action_one_hot.cpu().detach().numpy()
                action = np.argmax(action_one_hot)

                q = []
                q.append(model.qf1_model(torch.cat([torch.tensor(obs).to(args.device),
                                                    torch.tensor([1., 0., 0., 0.]).to(args.device).type(torch.float)],
                                                   dim=-1)).mean.item())
                q.append(model.qf1_model(
                    torch.cat([torch.tensor(obs).to(args.device), torch.tensor([0., 1., 0., 0.]).to(args.device).type(torch.float)],
                              dim=-1)).mean.item())
                q.append(model.qf1_model(
                    torch.cat([torch.tensor(obs).to(args.device), torch.tensor([0., 0., 1., 0.]).to(args.device).type(torch.float)],
                              dim=-1)).mean.item())
                q.append(model.qf1_model(
                    torch.cat([torch.tensor(obs).to(args.device), torch.tensor([0., 0., 0., 1.]).to(args.device).type(torch.float)],
                              dim=-1)).mean.item())
                # print('q_value:', q)

        next_obs, reward, done, info = env.step(action)
        reward, done = np.array([reward]), np.array([done])
        replay_buffer.add(obs, action_one_hot, reward, next_obs, done)

        total_reward += reward
        episode_count += 1
        count += 1
        obs = next_obs

        if count >= args.pretrain and count % args.train_every == 0:
            train_every_count = 0
            model.train()
            model = algo.optimize(model, replay_buffer)
            model.eval()

        if done or episode_count % args.max_episode_count == 0:
            itr += 1
            avg_reward.append(total_reward.item())
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(itr, np.mean(avg_reward)), end="")
            if itr % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(itr, np.mean(avg_reward)))
            train_every_count += 1
            if args.render:
                env.render()
            if eps > 0.01:
                eps *= eps_decay
            obs = env.reset()
            algo.writer.add_scalar('Loss/avg_reward', total_reward, itr)
            total_reward = 0.0
            episode_count = 0
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
                           help='Max count for each episode',
                           default=1000,
                           type=int)
    argparser.add_argument('--pretrain',
                           help='Episodes needed to start pretraining',
                           default=500,
                           type=int)
    argparser.add_argument('--train_every',
                           help='Episodes needed to start every training after pretrain',
                           default=4,
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