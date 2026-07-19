# Formative 3 — Deep Q-Learning on Atari

Train and evaluate a Deep Q-Network (DQN) agent on an Atari game using **Stable Baselines3** and **Gymnasium**. This repository contains the training script, evaluation/play script, logged hyperparameter experiments, and the final saved policy.

**Repository:** [https://github.com/PapiWinnie/Formative-3-Assignment-Deep-Q-Learning](https://github.com/PapiWinnie/Formative-3-Assignment-Deep-Q-Learning)

---

## Environment

**Chosen Atari environment:** `ALE/Pong-v5`

| Property | Value |
|---|---|
| Action space | `Discrete(6)` — NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE |
| Observation space | `Box(0, 255, (210, 160, 3), uint8)` — RGB frames |
| Reward | +1 when the agent scores, −1 when the opponent scores |
| Built-in frameskip | 4 (plus sticky actions, `repeat_action_probability=0.25`) |

`train.py` and `play.py` pass `frameskip=1` into the base env so SB3’s `AtariWrapper` is the only layer that skips frames (avoids an effective 16-frame skip). CNN runs also use `VecFrameStack(n_stack=4)`.

Pong is a classic Atari control task: the agent must track the ball and paddle from pixels and learn a long-horizon rally policy. That makes it a strong testbed for DQN, CNN vs MLP policies, and discount / exploration tuning.

---

## Project structure

| File / folder | Purpose |
|---|---|
| `train.py` | Train a DQN agent (CnnPolicy or MlpPolicy), log metrics, save the model |
| `play.py` | Load `models/dqn_model.zip` and play with a greedy (deterministic) policy + GUI render |
| `models/dqn_model.zip` | Final submitted policy network |
| `experiments_log.csv` | Winston learning-rate / policy runs |
| `experiments_log_gamma.csv` | Sougnabe gamma runs |
| `experiments_log_Epsilon.csv` | David epsilon-schedule runs |
| `colab/` | Member Colab notebooks used to run the sweeps |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Task 1 — Training (`train.py`)

`train.py` builds an Atari env, constructs a Stable Baselines3 `DQN` with either **CnnPolicy** or **MlpPolicy**, trains for a configurable number of timesteps, and:

- logs mean episode reward and mean episode length
- saves a per-run checkpoint `models/dqn_model_<run_name>.zip`
- updates the canonical `models/dqn_model.zip`

Example:

```bash
python train.py --env ALE/Pong-v5 --policy cnn --member "Winston" --name winston_lr3e-05_cnn \
    --lr 3e-05 --gamma 0.99 --batch-size 32 \
    --eps-start 1.0 --eps-end 0.05 --eps-fraction 0.1 \
    --timesteps 500000
```

> **Note on ε-decay:** The assignment lists `epsilon_decay`. Stable Baselines3 exposes this as `exploration_fraction` (`--eps-fraction`): the fraction of training over which ε decays linearly from `eps_start` to `eps_end`.

### Group roles (10 experiments each)

| Member | Focus | Timesteps per run |
|---|---|---|
| **Winston** | Learning rate + MLP vs CNN policy comparison | 500,000 |
| **Sougnabe** | Discount factor γ | 150,000 |
| **David** | Epsilon schedule (`eps_start`, `eps_end`, `eps_fraction`) | 500,000 |

---

## Hyperparameter tables

Rewards are **final mean episode reward** from the training buffer (higher / less negative is better on Pong). Episode length rising usually means longer rallies before a point is scored.

### Winston — Learning rate & policy (10 experiments)

Fixed: `gamma=0.99`, `batch=32`, `eps_start=1.0`, `eps_end=0.05`, `eps_fraction=0.1`, `timesteps=500000`.

| Member | Hyperparameter Set | Noted Behavior |
|---|---|---|
| Winston | lr=0.001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=MLP** | Reward −20.88, ep_len ≈3208. High LR; agent stayed near random-play level. |
| Winston | lr=0.001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=CNN** | Reward −20.88, ep_len ≈3207. Same collapse as MLP — LR too aggressive for stable Q updates. |
| Winston | lr=0.0003, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=MLP** | Reward −20.64, ep_len ≈3643. Slightly longer episodes than other MLP runs; still no real learning. |
| Winston | lr=0.0003, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=CNN** | Reward −20.90, ep_len ≈3182. No improvement vs higher LR. |
| Winston | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=MLP** | Reward −20.92, ep_len ≈3172. Default SB3-ish LR; MLP still fails on raw pixels. |
| Winston | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=CNN** | Reward −20.88, ep_len ≈3210. CNN with default LR also stuck near −21. |
| Winston | lr=3e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=MLP** | Reward −20.94, ep_len ≈3226. Lower LR alone does not rescue MLP. |
| Winston | lr=3e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=CNN** | **Best overall run.** Reward **−15.86**, ep_len **≈7599**. Clear learning signal; longer rallies. |
| Winston | lr=1e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=MLP** | Reward −20.76, ep_len ≈3225. Too slow; barely moves off baseline. |
| Winston | lr=1e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1, **policy=CNN** | Reward −20.53, ep_len ≈3518. Better than most runs but far behind `lr=3e-05` CNN — under-fitting / too slow. |

**Winston insights:** CNN beat MLP (avg reward ≈ −19.81 vs ≈ −20.83). The only strong config was **CNN + lr=3e-05**. Higher LRs (1e-4–1e-3) failed; 1e-05 was too conservative.

---

### Sougnabe — Gamma / discount factor (10 experiments)

Fixed: `policy=CNN`, `lr=1e-4`, `batch=32`, `eps_start=1.0`, `eps_end=0.05`, `eps_fraction=0.1`, `timesteps=150000`.

| Member | Hyperparameter Set | Noted Behavior |
|---|---|---|
| Sougnabe | lr=0.0001, gamma=0.80, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.83. Very myopic; ignores long rallies. |
| Sougnabe | lr=0.0001, gamma=0.85, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.69. Slightly better than 0.80; still short-sighted. |
| Sougnabe | lr=0.0001, gamma=0.88, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.66. Marginal gain as γ increases. |
| Sougnabe | lr=0.0001, gamma=0.90, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.84. No clear benefit vs neighboring values. |
| Sougnabe | lr=0.0001, gamma=0.93, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.74, ep_len ≈3542. Mildly longer episodes. |
| Sougnabe | lr=0.0001, gamma=0.95, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.61. Best among mid-range γ values. |
| Sougnabe | lr=0.0001, gamma=0.97, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | **Best in gamma sweep.** Reward **−19.99**, ep_len ≈3779. Stronger long-horizon credit assignment. |
| Sougnabe | lr=0.0001, gamma=0.98, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.68. Slight regression vs 0.97 — high γ can add variance. |
| Sougnabe | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.35. Solid but not best at 150k steps. |
| Sougnabe | lr=0.0001, gamma=0.995, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Reward −20.41. Very high γ; no extra gain at this budget. |

**Sougnabe insights:** Low γ (≤0.90) underperformed. **γ=0.97** was best in this sweep. Values ≥0.98 did not keep improving at 150k timesteps — more foresight helps, but extreme γ needs more training to stabilize. `batch_size` was held at 32 across all group runs for fair comparison on other axes.

---

### David — Epsilon schedule (10 experiments)

Fixed: `policy=CNN`, `lr=1e-4`, `gamma=0.99`, `batch=32`, `timesteps=500000`.

| Member | Hyperparameter Set | Noted Behavior |
|---|---|---|
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.1 | Baseline. Reward ≈ −20.9, ep_len ≈3230. Standard schedule; little learning at this LR. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.05 | Fast decay. Reward **−18.75**, ep_len ≈7021. Earlier exploitation helped vs baseline. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.3 | Slow decay. Reward −20.86. Too much exploration for too long. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.5 | Very slow decay. Reward −20.88. Harmed performance — agent rarely exploits. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | Low final ε. **Best in epsilon sweep.** Reward **−17.00**, ep_len ≈7134. Greedier late play improved score. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.1 | High final ε. Reward −18.53. Still better than baseline, but residual noise hurts vs `eps_end=0.01`. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.1 | Mid start. Reward −18.94. Less early exploration than 1.0; decent mid-tier result. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=0.2, epsilon_end=0.05, epsilon_decay=0.1 | Low start. Reward −17.74. Strong — less wasted early random play on Pong. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.4 | High start + slow decay to low end. Reward −20.95. Worst combo — explores too long. |
| David | lr=0.0001, gamma=0.99, batch=32, epsilon_start=0.5, epsilon_end=0.1, epsilon_decay=0.2 | Narrow band. Reward −20.82. Neither enough early exploration nor enough late greed. |

**David insights:** What helped: **lower `eps_end` (0.01)**, **faster decay (0.05)**, and **lower `eps_start` (0.2–0.5)**. What hurt: very slow decay (0.3–0.5) and combining high start with slow decay. Exploration–exploitation must shift toward exploitation within the training budget.

---

## MLP vs CNN

On pixel-based Atari, **CnnPolicy clearly outperformed MlpPolicy**.

| Policy | Mean reward (Winston sweep) | Best single run |
|---|---|---|
| CNN | −19.81 | **−15.86** (`lr=3e-05`) |
| MLP | −20.83 | −20.64 (`lr=3e-04`) |

**Why:** Pong observations are raw RGB frames. CNNs extract spatial features (paddle, ball, edges). MLPs flatten the image and lose locality, so they struggle to represent the Q-function from pixels in the same budget. After this comparison, all later gamma and epsilon sweeps used **CNN**.

---

## Hyperparameter tuning discussion

Across 30 logged experiments (10 per member):

1. **Learning rate** — Highest impact when paired with CNN. `3e-05` unlocked the best score (−15.86). `1e-4`–`1e-3` were unstable/ineffective; `1e-05` was too slow.
2. **Policy architecture** — CNN ≫ MLP for Atari pixels; architecture choice dominated most LR settings.
3. **Gamma (γ)** — Mid-high values worked best; **0.97** led the gamma sweep (−19.99). Very low γ under-values future points; very high γ (≥0.98) did not help more at 150k steps.
4. **Epsilon schedule** — Lower final ε and reasonably fast decay improved scores (`eps_end=0.01` → −17.00). Prolonged high exploration wasted the timestep budget.
5. **Batch size** — Held at **32** for all reported runs so each member’s axis (lr/policy, γ, ε) stayed comparable. No evidence in this repo that batch size alone drove the largest gains.

Raw logs: `experiments_log.csv`, `experiments_log_gamma.csv`, `experiments_log_Epsilon.csv`.

---

## Best configuration (submitted model)

The strongest measured training result was Winston’s CNN run:

| Hyperparameter | Value |
|---|---|
| Environment | `ALE/Pong-v5` |
| Policy | **CnnPolicy** |
| Learning rate | **3e-05** |
| Gamma | 0.99 |
| Batch size | 32 |
| Epsilon start | 1.0 |
| Epsilon end | 0.05 |
| Epsilon decay (`exploration_fraction`) | 0.1 |
| Timesteps | 500,000 |
| Final mean episode reward | **−15.86** |
| Final mean episode length | **≈7599** |

**Saved policy:** [`models/dqn_model.zip`](models/dqn_model.zip)

This config was chosen because it is the only run that clearly left the ≈−21 random-play band and produced substantially longer rallies. Complementary findings (γ ≈ 0.97, lower `eps_end`) suggest further gains if combined in a follow-up train, but the submitted zip is anchored to the best completed experiment.

---

## Task 2 — Playing (`play.py`)

`play.py` loads the trained DQN and evaluates with a **greedy policy**: Stable Baselines3 has no `GreedyQPolicy` class name (that term is from Keras-RL); the equivalent is `model.predict(obs, deterministic=True)`, which always picks the max-Q action.

```bash
python play.py --model models/dqn_model.zip --env ALE/Pong-v5 --policy cnn --episodes 5
```

The script uses `render_mode="human"` and `env.render()` so gameplay is visible in a GUI window.

---

## Gameplay video

> **Add the recording here before final submission.**

Record `play.py` running so the agent is visible interacting with Pong (screen capture of the GUI window), then link or embed it below.

**Gameplay video:** _[paste Drive / YouTube / GitHub release link here]_

Suggested capture:

```bash
python play.py --model models/dqn_model.zip --env ALE/Pong-v5 --policy cnn --episodes 3
```

---

## Presentation notes (group)

Each member (≈2 minutes): cover your 10 rows, what helped vs hurt, and your best setting.

| Member | What improved performance | What harmed performance | Best local config |
|---|---|---|---|
| Winston | CNN + lower LR (`3e-05`) | MLP; LR ≥ 1e-4 or LR = 1e-05 | CNN, lr=3e-05 (−15.86) |
| Sougnabe | γ around 0.97 | γ ≤ 0.90 | γ=0.97 (−19.99) |
| David | low `eps_end`, faster decay, lower start | very slow decay / high start + slow decay | eps_end=0.01 (−17.00) |

Be ready for Q&A on exploration–exploitation trade-offs, why the final agent behaves as it does, and why CNN beat MLP.

---

## Submission checklist

- [x] `train.py` — DQN training, MLP/CNN, hyperparameter flags, logging, `dqn_model.zip`
- [x] `play.py` — load model, greedy evaluation, GUI render
- [x] 10 experiments per member documented in tables above
- [x] `models/dqn_model.zip` in the repository
- [ ] Gameplay video linked in this README
- [ ] Coach hyperparameter sheet filled, saved as PDF, and added to the repo
- [ ] Presentation slot booked; cameras on; gameplay clip ready
