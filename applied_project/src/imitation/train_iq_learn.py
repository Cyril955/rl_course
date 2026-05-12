import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.buffers.replay_buffer import ReplayBuffer
from src.datasets.expert_transition_dataset import ExpertTransitionDataset
from src.imitation.iq_learn import compute_iq_loss, compute_regularizer
from src.imitation.q_network import QNetwork


def collect_steps(
    env,
    q_net: QNetwork,
    replay_buffer: ReplayBuffer,
    n_steps: int,
    greedy: bool = False,
) -> None:
    obs, _ = env.reset()

    for _ in range(n_steps):
        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
        ).unsqueeze(0)

        action = q_net.act(obs_tensor, greedy=greedy)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.add(
            state=obs,
            action=action,
            reward=reward,
            next_state=next_obs,
            done=done,
        )

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs


def sample_initial_states(
    env_id: str,
    batch_size: int,
) -> torch.Tensor:
    states = []

    for _ in range(batch_size):
        env = gym.make(env_id)
        obs, _ = env.reset()
        states.append(obs)
        env.close()

    return torch.tensor(np.asarray(states), dtype=torch.float32)


def train_iq_learn(
    env_id: str,
    expert_dataset_path: str,
    save_path: str,
    seed: int = 0,
    total_updates: int = 20_000,
    start_steps: int = 1_000,
    collect_per_update: int = 1,
    batch_size: int = 256,
    gamma: float = 0.99,
    lr: float = 3e-4,
    lambda_reg: float = 0.1,
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_id)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    expert_dataset = ExpertTransitionDataset(expert_dataset_path)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim)

    collect_steps(
        env=env,
        q_net=q_net,
        replay_buffer=replay_buffer,
        n_steps=start_steps,
    )

    for update in range(total_updates):
        collect_steps(
            env=env,
            q_net=q_net,
            replay_buffer=replay_buffer,
            n_steps=collect_per_update,
        )

        expert_batch = expert_dataset.sample(batch_size)
        agent_batch = replay_buffer.sample(batch_size)
        init_states = sample_initial_states(env_id, batch_size)

        iq_loss = compute_iq_loss(
            q_net=q_net,
            expert_batch=expert_batch,
            init_states=init_states,
            gamma=gamma,
        )

        reg_loss = compute_regularizer(
            q_net=q_net,
            agent_batch=agent_batch,
            gamma=gamma,
        )

        loss = iq_loss + lambda_reg * reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()

        if update % 500 == 0:
            print(
                f"Update {update:06d} | "
                f"loss={loss.item():.4f} | "
                f"iq={iq_loss.item():.4f} | "
                f"reg={reg_loss.item():.4f}"
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(q_net.state_dict(), save_path)

    env.close()
    print(f"Saved IQ-Learn Q-network to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--expert-dataset", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)

    args = parser.parse_args()

    train_iq_learn(
        env_id=args.env_id,
        expert_dataset_path=args.expert_dataset,
        save_path=args.save_path,
        seed=args.seed,
        total_updates=args.updates,
        batch_size=args.batch_size,
    )


# python scripts/05_train_iq_learn.py \
#   --env-id CartPole-v1 \
#   --expert-dataset data/expert/CartPole-v1/expert_K10_seed0.npz \
#   --save-path models/iq_learn/CartPole-v1/iq_K10_seed0.pt \
#   --seed 0 \
#   --updates 20000