import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


def train_ppo_expert(
    env_id: str,
    seed: int,
    total_timesteps: int,
    save_dir: str,
) -> None:
    set_random_seed(seed)

    env = gym.make(env_id)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)

    save_path = Path(save_dir) / env_id / f"ppo_expert_seed_{seed}"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(save_path)
    env.close()

    print(f"Saved expert to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--save-dir", type=str, default="models/experts")

    args = parser.parse_args()

    train_ppo_expert(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
    )

# python train_expert_PPO.py --env-id CartPole-v1 --seed 0 --timesteps 100000
# python train_expert_PPO.py --env-id Acrobot-v1 --seed 0 --timesteps 300000
