"""Train a DQN agent on an Atari environment with Stable Baselines3.

Every group member runs this same script with their own hyperparameter
flags so results stay comparable. Each run appends one row to
experiments_log.csv, which is the raw material for the assignment's
hyperparameter table.

Examples:
  python train.py --env ALE/Pong-v5 --policy cnn --member Alice --name alice_01 --lr 1e-4
  python train.py --env ALE/Pong-v5 --policy mlp --member Bob   --name bob_01   --gamma 0.9
"""
import argparse
import csv
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

LOG_CSV = Path(__file__).parent / "experiments_log.csv"
CSV_FIELDS = [
    "run_name", "member", "env", "policy", "lr", "gamma", "batch_size",
    "eps_start", "eps_end", "eps_fraction", "timesteps",
    "final_ep_rew_mean", "final_ep_len_mean", "train_seconds", "model_path",
]


def build_env(env_id: str, policy: str):
    # ALE/*-v5 envs already frameskip=4 internally; SB3's AtariWrapper (used by
    # make_atari_env) also frameskips by default. Force frameskip=1 on the base
    # env so the wrapper is the only one skipping frames (avoids an effective
    # 16-frame skip instead of the intended 4).
    env = make_atari_env(env_id, n_envs=1, seed=0, env_kwargs={"frameskip": 1})
    if policy == "cnn":
        env = VecFrameStack(env, n_stack=4)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari with SB3.")
    parser.add_argument("--env", default="ALE/Pong-v5", help="Gymnasium Atari env id")
    parser.add_argument("--policy", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--member", default="unassigned", help="Group member name, for the log")
    parser.add_argument("--name", default=None, help="Run name (used in model filename + log row)")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument(
        "--eps-fraction", type=float, default=0.1,
        help="Fraction of training over which epsilon decays (SB3's exploration_fraction)",
    )
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    run_name = args.name or f"{args.policy}_{int(time.time())}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"dqn_model_{run_name}.zip"

    env = build_env(args.env, args.policy)
    sb3_policy = "CnnPolicy" if args.policy == "cnn" else "MlpPolicy"

    model = DQN(
        sb3_policy,
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        exploration_initial_eps=args.eps_start,
        exploration_final_eps=args.eps_end,
        exploration_fraction=args.eps_fraction,
        verbose=1,
        tensorboard_log="./tb_logs",
    )

    start = time.time()
    model.learn(total_timesteps=args.timesteps, tb_log_name=run_name)
    train_seconds = time.time() - start

    if model.ep_info_buffer:
        final_rew_mean = float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
        final_len_mean = float(np.mean([ep["l"] for ep in model.ep_info_buffer]))
    else:
        final_rew_mean = None
        final_len_mean = None

    model.save(model_path)
    # Keep a canonical dqn_model.zip pointing at whichever run was trained most recently.
    model.save(out_dir / "dqn_model.zip")

    is_new = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "run_name": run_name, "member": args.member, "env": args.env,
            "policy": args.policy, "lr": args.lr, "gamma": args.gamma,
            "batch_size": args.batch_size, "eps_start": args.eps_start,
            "eps_end": args.eps_end, "eps_fraction": args.eps_fraction,
            "timesteps": args.timesteps, "final_ep_rew_mean": final_rew_mean,
            "final_ep_len_mean": final_len_mean, "train_seconds": round(train_seconds, 1),
            "model_path": str(model_path),
        })

    print(f"Saved model to {model_path} and logged run '{run_name}' to {LOG_CSV}")


if __name__ == "__main__":
    main()
