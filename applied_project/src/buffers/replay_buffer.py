import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        capacity: int = 100_000,
    ) -> None:
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)

        return {
            "states": torch.tensor(self.states[idx], dtype=torch.float32),
            "actions": torch.tensor(self.actions[idx], dtype=torch.long),
            "rewards": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "next_states": torch.tensor(
                self.next_states[idx],
                dtype=torch.float32,
            ),
            "dones": torch.tensor(self.dones[idx], dtype=torch.float32),
        }

    def __len__(self) -> int:
        return self.size