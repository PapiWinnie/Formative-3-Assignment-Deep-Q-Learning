# Formative 3 — Deep Q-Learning (Atari)

## Environment

**Chosen Atari environment:** `ALE/Pong-v5`

- Action space: `Discrete(6)` (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
- Observation space: `Box(0, 255, (210, 160, 3), uint8)` (RGB frames)
- Built-in `frameskip=4`, `repeat_action_probability=0.25` (sticky actions)
- Note: `train.py`/`play.py` pass `frameskip=1` to the base env to avoid
  double frame-skipping with SB3's `AtariWrapper` (which also skips 4 frames) —
  see the comment in `build_env()`.

## Setup

```bash
pip install -r requirements.txt
```

## Files

- `train.py` — trains a DQN agent (CNN or MLP policy) and logs each run to `experiments_log.csv`
- `play.py` — loads `models/dqn_model.zip` and plays a few episodes with a greedy (deterministic) policy
- `experiments_log.csv` — auto-generated log of every training run (one row per experiment)

## Roles

| Member | Hyperparameter focus | Experiments |
|---|---|---|
| _Name 1_ | Learning rate (+ MLP vs CNN policy comparison) | 10 |
| _Name 2_ | Gamma, batch size | 10 |
| _Name 3_ | Epsilon schedule (start / end / decay) | 10 |

## Running an experiment

```bash
python train.py --env ALE/Pong-v5 --policy cnn --member "Name 1" --name name1_01 \
    --lr 1e-4 --gamma 0.99 --batch-size 32 --eps-start 1.0 --eps-end 0.05 --eps-fraction 0.1
```

Repeat with different flags for each of your 10 runs — `--member` and `--name` keep the log
organized. After all 30 runs are in `experiments_log.csv`, pick the best-performing
config and retrain/confirm it as the final `models/dqn_model.zip`.

## Hyperparameter Table

_Fill in from `experiments_log.csv` once experiments are done._

| Member | Hyperparameter Set | Noted Behavior |
|---|---|---|
| | lr=, gamma=, batch=, epsilon_start=, epsilon_end=, epsilon_decay= | |

## MLP vs CNN

_Summarize which policy performed better on this environment and why._

## Best Configuration

_Final hyperparameters used for the submitted `dqn_model.zip`, and why it was chosen._

## Gameplay Video

_Link or embed the `play.py` recording here._

## Playing the trained agent

```bash
python play.py --model models/dqn_model.zip --env ALE/Pong-v5 --policy cnn --episodes 5
```
