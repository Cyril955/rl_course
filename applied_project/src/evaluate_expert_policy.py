import argparse
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate_ppo(
    env_id: str,
    model_path: str,
    n_episodes: int = 20,
    render: bool = False,
    sleep: float = 0.0,
) -> tuple[float, float]:
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    model = PPO.load(model_path)

    returns = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            steps += 1

            if render and sleep > 0:
                time.sleep(sleep)
                
        returns.append(episode_return)
        status = "success" if terminated else "time_limit"

        print(
            f"Episode {episode + 1}: "
            f"return = {episode_return:.2f}, "
            f"steps = {steps}, "
            f"status = {status}"
        )

    env.close()

    return float(np.mean(returns)), float(np.std(returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)

    args = parser.parse_args()

    mean_return, std_return = evaluate_ppo(
        env_id=args.env_id,
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render,
        sleep=args.sleep,
    )

    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")

# Without rendering:
# python evaluate_expert_policy.py --env-id CartPole-v1 --model-path models/experts/CartPole-v1/ppo_expert_seed_0.zip
# python evaluate_expert_policy.py --env-id Acrobot-v1 --model-path models/experts/Acrobot-v1/ppo_expert_seed_0.zip

# With rendering:
# python evaluate_expert_policy.py --env-id CartPole-v1 --model-path models/experts/CartPole-v1/ppo_expert_seed_0.zip --episodes 3 --render --sleep 0.02
# python evaluate_expert_policy.py --env-id Acrobot-v1 --model-path models/experts/Acrobot-v1/ppo_expert_seed_0.zip --episodes 3 --render --sleep 0.02