import gym
import numpy as np

def initialize_env(env_name, rd_seed=42, print_spaces=False):

    env = gym.make(env_name)
    env.action_space.seed(rd_seed)

    if print_spaces:
        print("-------------------------")
        print(f"Environment: {env_name}")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        print("-------------------------")

    return env