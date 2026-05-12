import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def q_value(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.forward(obs)
        return q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    def soft_value(self, obs: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(obs)
        return torch.logsumexp(q_values, dim=-1)

    def act(
        self,
        obs: torch.Tensor,
        greedy: bool = False,
    ) -> int:
        with torch.no_grad():
            q_values = self.forward(obs)

            if greedy:
                action = torch.argmax(q_values, dim=-1)
            else:
                probs = torch.softmax(q_values, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

        return int(action.item())