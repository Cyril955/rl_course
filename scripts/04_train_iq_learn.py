import argparse
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# Pool size: we always convert the full expert NPZ and let IQ-Learn subsample.
# The PKL is named with _pool suffix to clarify it's the full pool.
POOL_SUFFIX = "pool"


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
    if output_pkl.exists():
        n_traj = len(pickle.load(open(output_pkl, "rb"))["states"])
        print(f"[SKIP] PKL already exists ({n_traj} trajectories): {output_pkl}")
        return

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
        sys.executable,
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


def copy_training_curve(
    iq_repo_dir: Path,
    env_id: str,
    n_demos: int,
    seed: int,
    output_path: Path,
) -> bool:
    """Find training_curve.json from the most recent Hydra run and save it."""
    import json
    iq_workdir = iq_repo_dir / "iq_learn"
    outputs_root = iq_workdir / "outputs"
    candidates: list[Path] = []
    if outputs_root.exists():
        run_dirs = sorted(outputs_root.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            candidates.append(run_dir / "training_curve.json")

    for src in candidates:
        if src.exists():
            raw = json.loads(src.read_text())
            out = {
                "env":             env_id,
                "method":          "iq_learn",
                "K":               n_demos,
                "train_seed":      seed,
                "episode_returns": raw.get("episode_returns", []),
                "eval_history":    raw.get("eval_history", []),
                "n_episodes":      len(raw.get("episode_returns", [])),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(out))
            print(f"[OK] Training curve saved → {output_path}")
            return True

    print("[WARNING] training_curve.json not found in Hydra output dirs.")
    return False


def copy_trained_model(
    iq_repo_dir: Path,
    env_id: str,
    output_model_path: Path,
) -> bool:
    """Copy the best trained IQ-Learn model to a unique path.

    Hydra changes cwd to outputs/{date}/{time}/, so models land there, not in
    the top-level iq_learn/results/. We find the most recently modified run dir.
    """
    iq_workdir = iq_repo_dir / "iq_learn"
    model_name = f"softq_iq_{env_id}"

    # Collect all candidate paths from Hydra output dirs (most recent first)
    # and also the legacy top-level results dirs (fallback)
    candidates: list[Path] = []

    outputs_root = iq_workdir / "outputs"
    if outputs_root.exists():
        run_dirs = sorted(outputs_root.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs:
            # "results" is the most recent periodic save (≈ end of training).
            # "results_best" is only saved at eval + epoch%5==0, so epoch 0
            # (before training) satisfies this first — giving the initial bad model.
            # Using "results" first avoids this trap.
            for results_subdir in ("results", "results_best"):
                candidates.append(run_dir / results_subdir / model_name)

    for results_subdir in ("results", "results_best"):
        candidates.append(iq_workdir / results_subdir / model_name)

    for src in candidates:
        if src.exists():
            output_model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, output_model_path)
            print(f"[OK] Saved model: {src} → {output_model_path}")
            return True

    print(f"[WARNING] No trained model found in {outputs_root} or {iq_workdir}/results*/")
    return False


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--env-id", type=str, required=True)
    # Should be the full pool NPZ (e.g. expert_K10_seed0.npz).
    # IQ-Learn will subsample --n-demos trajectories from it.
    parser.add_argument("--expert-npz", type=str, required=True)
    parser.add_argument("--iq-repo-dir", type=str, default="external/IQ-Learn")
    parser.add_argument("--n-demos", type=int, required=True,
                        help="Number of expert trajectories IQ-Learn will use (K).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learn-steps", type=int, default=100_000)
    parser.add_argument("--subsample-freq", type=int, default=1)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument(
        "--output-model", type=str, default=None,
        help="Where to save the trained Q-net weights. "
             "Defaults to models/iq_learn/{env}/K{n_demos}_seed{seed}.pt",
    )

    args = parser.parse_args()

    expert_npz = Path(args.expert_npz).resolve()
    iq_repo_dir = Path(args.iq_repo_dir).resolve()

    if not expert_npz.exists():
        raise FileNotFoundError(f"Expert dataset not found: {expert_npz}")

    if not iq_repo_dir.exists():
        raise FileNotFoundError(f"IQ-Learn repo not found: {iq_repo_dir}")

    # Determine the seed suffix from the NPZ filename (e.g. expert_K10_seed0.npz → seed0)
    npz_stem = expert_npz.stem  # e.g. "expert_K10_seed0"
    pool_seed = args.seed  # reuse --seed as dataset seed by convention

    # Always convert the full pool PKL (named after pool size, not K).
    # Multiple K values reuse the same PKL; IQ-Learn subsamples via expert.demos=K.
    converted_dir = Path("data/iq_learn") / args.env_id
    pool_name = f"{args.env_id}_{npz_stem}.pkl"  # e.g. CartPole-v1_expert_K10_seed0.pkl
    converted_pkl = converted_dir / pool_name

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

    output_model = Path(
        args.output_model
        or f"models/iq_learn/{args.env_id}/K{args.n_demos}_seed{args.seed}.pt"
    )
    copy_trained_model(iq_repo_dir, args.env_id, output_model)

    training_json = Path(
        f"results/training/iq_learn/{args.env_id}/"
        f"iq_learn_K{args.n_demos}_seed{args.seed}.json"
    )
    copy_training_curve(iq_repo_dir, args.env_id, args.n_demos, args.seed, training_json)


if __name__ == "__main__":
    main()

# Example – single run:
# python scripts/04_train_iq_learn.py \
#   --env-id CartPole-v1 \
#   --expert-npz data/expert/CartPole-v1/expert_K15_seed0.npz \
#   --n-demos 10 --seed 0 --learn-steps 100000 --subsample-freq 20
#
# Use 08_run_sweep.sh to run the full K × seed grid.