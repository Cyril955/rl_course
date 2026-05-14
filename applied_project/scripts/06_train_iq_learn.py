import argparse
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np


def split_flat_transitions_into_trajectories(data: np.lib.npyio.NpzFile):
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_states = data["next_states"]
    dones = data["dones"]

    trajectories = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": [],
        "lengths": [],
    }

    start = 0

    for i, done in enumerate(dones):
        if bool(done):
            end = i + 1

            trajectories["states"].append(states[start:end])
            trajectories["actions"].append(actions[start:end])
            trajectories["rewards"].append(rewards[start:end])
            trajectories["next_states"].append(next_states[start:end])
            trajectories["dones"].append(dones[start:end])
            trajectories["lengths"].append(end - start)

            start = end

    if start < len(states):
        end = len(states)

        trajectories["states"].append(states[start:end])
        trajectories["actions"].append(actions[start:end])
        trajectories["rewards"].append(rewards[start:end])
        trajectories["next_states"].append(next_states[start:end])
        trajectories["dones"].append(dones[start:end])
        trajectories["lengths"].append(end - start)

    trajectories["lengths"] = np.asarray(
        trajectories["lengths"],
        dtype=np.int64,
    )

    return trajectories


def convert_npz_to_iq_pkl(
    input_npz: Path,
    output_pkl: Path,
) -> None:
    data = np.load(input_npz)
    trajectories = split_flat_transitions_into_trajectories(data)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pkl, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"[OK] Converted expert dataset:")
    print(f"     input:  {input_npz}")
    print(f"     output: {output_pkl}")
    print(f"     trajectories: {len(trajectories['states'])}")
    print(f"     transitions:  {sum(trajectories['lengths'])}")


def copy_dataset_to_iq_repo(
    converted_pkl: Path,
    iq_repo_dir: Path,
) -> Path:
    experts_dir = iq_repo_dir / "iq_learn" / "experts"
    experts_dir.mkdir(parents=True, exist_ok=True)

    destination = experts_dir / converted_pkl.name
    shutil.copyfile(converted_pkl, destination)

    print(f"[OK] Copied dataset into IQ-Learn repo:")
    print(f"     {destination}")

    return destination


def env_id_to_iq_env_name(env_id: str) -> str:
    mapping = {
        "CartPole-v1": "cartpole",
        "Acrobot-v1": "acrobot",
    }

    if env_id not in mapping:
        raise ValueError(
            f"No IQ-Learn env mapping for {env_id}. "
            f"Known environments: {list(mapping.keys())}"
        )

    return mapping[env_id]


def run_iq_learn_training(
    iq_repo_dir: Path,
    env_id: str,
    demo_filename: str,
    n_demos: int,
    seed: int,
    learn_steps: int,
    subsample_freq: int,
    use_wandb: bool,
) -> None:
    iq_workdir = iq_repo_dir / "iq_learn"
    iq_env_name = env_id_to_iq_env_name(env_id)

    command = [
        "python",
        "train_iq.py",
        "agent=softq",
        "method=iq",
        f"env={iq_env_name}",
        f"env.demo={demo_filename}",
        f"expert.demos={n_demos}",
        f"expert.subsample_freq={subsample_freq}",
        f"seed={seed}",
        f"env.learn_steps={learn_steps}",
        "method.chi=True",
        "method.loss=value_expert",
        "agent.init_temp=0.001",
    ]

    if not use_wandb:
        command.append("project_name=disabled")

    print("[INFO] Launching IQ-Learn:")
    print("       " + " ".join(command))
    print(f"[INFO] Working directory: {iq_workdir}")

    subprocess.run(
        command,
        cwd=iq_workdir,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--expert-npz", type=str, required=True)
    parser.add_argument("--iq-repo-dir", type=str, default="external/IQ-Learn")
    parser.add_argument("--n-demos", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learn-steps", type=int, default=100_000)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--use-wandb", action="store_true")

    args = parser.parse_args()

    expert_npz = Path(args.expert_npz).resolve()
    iq_repo_dir = Path(args.iq_repo_dir).resolve()

    if not expert_npz.exists():
        raise FileNotFoundError(f"Expert dataset not found: {expert_npz}")

    if not iq_repo_dir.exists():
        raise FileNotFoundError(f"IQ-Learn repo not found: {iq_repo_dir}")

    converted_dir = Path("data/iq_format") / args.env_id
    converted_name = (
        f"{args.env_id}_K{args.n_demos}_seed{args.seed}.pkl"
    )
    converted_pkl = converted_dir / converted_name

    convert_npz_to_iq_pkl(
        input_npz=expert_npz,
        output_pkl=converted_pkl,
    )

    repo_dataset_path = copy_dataset_to_iq_repo(
        converted_pkl=converted_pkl,
        iq_repo_dir=iq_repo_dir,
    )

    run_iq_learn_training(
        iq_repo_dir=iq_repo_dir,
        env_id=args.env_id,
        demo_filename=repo_dataset_path.name,
        n_demos=args.n_demos,
        seed=args.seed,
        learn_steps=args.learn_steps,
        subsample_freq=args.subsample_freq,
        use_wandb=args.use_wandb,
    )


if __name__ == "__main__":
    main()

# python scripts/06_train_iq_learn.py --env-id CartPole-v1 --expert-npz data/expert/CartPole-v1/expert_K10_seed0.npz --iq-repo-dir external/IQ-Learn --n-demos 10 --seed 0 --learn-steps 100000 --subsample-freq 1