"""Load a trained DQN agent and watch it play its Atari environment.

The assignment calls for a "GreedyQPolicy" evaluation. Stable-Baselines3
doesn't have a class of that name (that's Keras-RL terminology) - the
equivalent here is model.predict(obs, deterministic=True), which always
takes the highest-Q action instead of exploring.

Usage:
  python play.py --model models/dqn_model.zip --env ALE/Pong-v5 --policy cnn --episodes 5
"""
import argparse

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)


def build_env(env_id: str, policy: str):
    # See train.py's build_env for why frameskip=1 is forced here.
    env = make_atari_env(
        env_id, n_envs=1, seed=0,
        env_kwargs={"render_mode": "human", "frameskip": 1},
    )
    if policy == "cnn":
        env = VecFrameStack(env, n_stack=4)
    return env


def main():
    parser = argparse.ArgumentParser(description="Play an Atari game with a trained DQN agent.")
    parser.add_argument("--model", default="models/dqn_model.zip")
    parser.add_argument("--env", default="ALE/Pong-v5")
    parser.add_argument("--policy", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = build_env(args.env, args.policy)
    model = DQN.load(args.model, env=env)

    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            env.render()
        print(f"Episode {ep}: total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
