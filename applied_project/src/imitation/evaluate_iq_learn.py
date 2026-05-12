import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from src.imitation.q_network import QNetwork


def evaluate_iq_policy(
    env_id: str,
    model_path: str,
    n_episodes: int = 20,
    render: bool = False,
    sleep: float = 0.0,
) -> tuple[float, float]:
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim)
    q_net.load_state_dict(torch.load(model_path, map_location="cpu"))
    q_net.eval()

    returns = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            obs_tensor = torch.tensor(
                obs,
                dtype=torch.float32,
            ).unsqueeze(0)

            action = q_net.act(obs_tensor, greedy=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ep_return += reward
            steps += 1

            if render and sleep > 0:
                time.sleep(sleep)

        returns.append(ep_return)

        print(
            f"Episode {episode + 1}: "
            f"return={ep_return:.2f}, steps={steps}"
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

    mean_return, std_return = evaluate_iq_policy(
        env_id=args.env_id,
        model_path=args.model_path,
        n_episodes=args.episodes,
        render=args.render,
        sleep=args.sleep,
    )

    print(f"Mean return: {mean_return:.2f} ± {std_return:.2f}")


# python scripts/06_evaluate_iq_learn.py \
#   --env-id CartPole-v1 \
#   --model-path models/iq_learn/CartPole-v1/iq_K10_seed0.pt \
#   --episodes 20


# python scripts/06_evaluate_iq_learn.py \
#   --env-id CartPole-v1 \
#   --model-path models/iq_learn/CartPole-v1/iq_K10_seed0.pt \
#   --episodes 3 \
#   --render \
#   --sleep 0.02