import torch


def compute_iq_loss(
    q_net,
    expert_batch: dict[str, torch.Tensor],
    init_states: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    states = expert_batch["states"]
    actions = expert_batch["actions"]
    next_states = expert_batch["next_states"]
    dones = expert_batch["dones"]

    q_exp = q_net.q_value(states, actions)

    with torch.no_grad():
        v_next = q_net.soft_value(next_states)

    bellman_exp = q_exp - gamma * (1.0 - dones) * v_next
    v_init = q_net.soft_value(init_states)

    iq_objective = bellman_exp.mean() - (1.0 - gamma) * v_init.mean()

    return -iq_objective


def compute_regularizer(
    q_net,
    agent_batch: dict[str, torch.Tensor],
    gamma: float,
) -> torch.Tensor:
    states = agent_batch["states"]
    actions = agent_batch["actions"]
    next_states = agent_batch["next_states"]
    dones = agent_batch["dones"]

    q_agent = q_net.q_value(states, actions)

    with torch.no_grad():
        v_next = q_net.soft_value(next_states)

    bellman_agent = q_agent - gamma * (1.0 - dones) * v_next

    return bellman_agent.pow(2).mean()