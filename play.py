"""Load a trained DQN agent and watch it play its Atari environment.

The assignment asks for GreedyQPolicy evaluation. Stable-Baselines3 does not
use that class name (Keras-RL); the equivalent is:

    model.predict(obs, deterministic=True)

which always selects the action with the highest Q-value (no exploration).

Usage:
  python play.py --model models/dqn_model.zip --env ALE/Pong-v5 --episodes 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)


def build_env(env_id: str, policy: str, seed: int):
    # Match train.py: ALE v5 already frameskips; force frameskip=1 so SB3's
    # AtariWrapper is the only skipper. Use human render for the live demo.
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": "human", "frameskip": 1},
        wrapper_kwargs={"terminal_on_life_loss": False},
    )
    if policy == "cnn":
        env = VecFrameStack(env, n_stack=4)
    return env


def detect_policy(model: DQN) -> str:
    """Infer cnn vs mlp from the loaded network (avoids obs-space mismatches)."""
    name = model.policy.__class__.__name__.lower()
    if "cnn" in name:
        return "cnn"
    if "mlp" in name:
        return "mlp"
    # Fallback: stacked image obs => CNN preprocessing
    shape = model.observation_space.shape
    if shape is not None and len(shape) == 3:
        return "cnn"
    return "mlp"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play an Atari game with a trained DQN agent (greedy evaluation).",
    )
    parser.add_argument(
        "--model",
        default="models/dqn_model.zip",
        help="Path to the saved DQN zip (default: models/dqn_model.zip)",
    )
    parser.add_argument("--env", default="ALE/Pong-v5", help="Gymnasium Atari env id")
    parser.add_argument(
        "--policy",
        choices=["auto", "cnn", "mlp"],
        default="auto",
        help="Must match training. 'auto' reads it from the zip (recommended).",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to play")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="If set, sample actions instead of greedy max-Q (not for graded demo).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path.resolve()}", file=sys.stderr)
        print(
            "Place the best checkpoint at models/dqn_model.zip "
            "(e.g. copy dqn_model_winston_lr3e-05_cnn.zip over it).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model from {model_path} ...")
    model = DQN.load(str(model_path), device="auto")

    policy = detect_policy(model) if args.policy == "auto" else args.policy
    if args.policy != "auto" and args.policy != detect_policy(model):
        print(
            f"WARNING: --policy={args.policy} but zip looks like "
            f"{detect_policy(model)}. Using --policy as requested.",
            file=sys.stderr,
        )

    deterministic = not args.stochastic
    lr = model.learning_rate(1.0) if callable(model.learning_rate) else model.learning_rate
    print(f"Env: {args.env}")
    print(f"Policy: {policy} ({model.policy.__class__.__name__})")
    print(f"Action selection: {'greedy (deterministic=True)' if deterministic else 'stochastic'}")
    print(
        f"Hyperparams in zip: lr={lr}, gamma={model.gamma}, batch={model.batch_size}, "
        f"eps={model.exploration_initial_eps}->{model.exploration_final_eps} "
        f"@ fraction={model.exploration_fraction}"
    )
    print(f"Timesteps trained (in zip): {model.num_timesteps}")
    if model.num_timesteps < 10_000:
        print(
            "WARNING: this zip looks barely trained "
            f"({model.num_timesteps} steps). For the demo, replace it with "
            "dqn_model_winston_lr3e-05_cnn.zip (500k steps).",
            file=sys.stderr,
        )
    print(f"Playing {args.episodes} episode(s). Close the game window or Ctrl+C to stop.\n")

    env = build_env(args.env, policy, args.seed)
    try:
        # Bind env so predict/obs shapes stay consistent with training wrappers.
        model.set_env(env)
    except ValueError as exc:
        env.close()
        print(f"ERROR: model does not match this env/policy setup: {exc}", file=sys.stderr)
        print("Hint: use --policy auto, or match the policy used during training.", file=sys.stderr)
        sys.exit(1)

    rewards_log: list[float] = []

    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            total_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, infos = env.step(action)
                total_reward += float(rewards[0])
                done = bool(np.asarray(dones).reshape(-1)[0])
                steps += 1
                # Live GUI (assignment: env.render / human display)
                env.render("human")

            rewards_log.append(total_reward)
            monitor = infos[0].get("episode") if infos else None
            if monitor is not None:
                print(
                    f"Episode {ep}: reward={monitor['r']:.2f}  "
                    f"length={int(monitor['l'])}  (steps this loop={steps})"
                )
            else:
                print(f"Episode {ep}: reward={total_reward:.2f}  steps={steps}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

    if rewards_log:
        print(
            f"\nDone. Mean reward over {len(rewards_log)} episode(s): "
            f"{float(np.mean(rewards_log)):.2f}"
        )


if __name__ == "__main__":
    main()
