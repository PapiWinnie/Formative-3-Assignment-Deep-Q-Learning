"""One-off backfill for experiments_log.csv rows written by the buggy
EpisodeStatsCallback (see train.py history): final_ep_rew_mean /
final_ep_len_mean were left blank even though the saved model trained fine.

For every row with a blank final_ep_rew_mean, this loads the matching
model, evaluates it for a few episodes, and writes the real numbers back
into that row. Everything else in the CSV is left untouched.

Usage:
  python scripts/backfill_rewards.py
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from train import build_env, CSV_FIELDS

LOG_CSV = REPO_ROOT / "experiments_log.csv"
N_EVAL_EPISODES = 20


def main():
    with open(LOG_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row["final_ep_rew_mean"].strip():
            continue

        model_path = REPO_ROOT / row["model_path"]
        if not model_path.exists():
            print(f"{row['run_name']}: skipped, model file not found at {model_path}")
            continue

        env = build_env(row["env"], row["policy"])
        model = DQN.load(model_path, env=env)

        episode_rewards, episode_lengths = evaluate_policy(
            model, env, n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True, return_episode_rewards=True,
        )
        env.close()

        rew_mean = round(float(sum(episode_rewards) / len(episode_rewards)), 2)
        len_mean = round(float(sum(episode_lengths) / len(episode_lengths)), 1)

        row["final_ep_rew_mean"] = rew_mean
        row["final_ep_len_mean"] = len_mean
        print(f"{row['run_name']}: final_ep_rew_mean={rew_mean}, final_ep_len_mean={len_mean}")

    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
