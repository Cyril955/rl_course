import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym

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

def evaluate_random_policy(env_name, num_episodes=100):
    """Evaluates a completely random policy to establish a lower bound."""
    env = gym.make(env_name)
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = env.action_space.sample()
            try:
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                state, reward, done, _ = env.step(action)
                
            # Handle Gym vs Gymnasium done flag
            if isinstance(done, tuple): done = done[0] or done[1] 
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)

def evaluate_expert_policy(env_name, model_path, num_episodes=100):
    """Loads an SB3 PPO zip file and evaluates it."""
    env = gym.make(env_name)
    model = PPO.load(model_path)
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        # Handle Gymnasium reset
        if isinstance(state, tuple): state = state[0]
        done = False
        ep_reward = 0
        while not done:
            action, _states = model.predict(state, deterministic=True)
            try:
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            except ValueError:
                state, reward, done, _ = env.step(action)
            if isinstance(done, tuple): done = done[0] or done[1]
            ep_reward += reward
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)