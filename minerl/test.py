import gym
import minerl
import logging

env = gym.make('MineRLNavigateDense-v0')

for episode in range(10):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        print(action)
        exit()
        s, r, done, info = env.step(action)
    print("End episode")

