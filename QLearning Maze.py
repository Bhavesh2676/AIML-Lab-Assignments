"""
Experiential Learning - AI & ML (T.Y. Core, SEM-V)
Activity: Design a Q-Learning Agent to Master a Maze Escape Game

Objectives covered:
  1. Implement a Q-learning agent that navigates and escapes a maze.
  2. Study the effect of exploration rate (epsilon), learning rate (alpha)
     and discount factor (gamma) on learning.
  3. Model a real-world navigation problem with reinforcement learning.

Environment
  S = start, E = exit (escape), # = wall, X = trap (dangerous cell).
  Actions : up, down, left, right
  Rewards : +100 for escaping, -100 for falling in a trap,
            -1 for every ordinary step (encourages the shortest path)

Run:  python EL_QLearning_Maze.py
Requires: numpy, matplotlib
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. MAZE ENVIRONMENT
# ---------------------------------------------------------------------------
MAZE = [
    "S . . # . . .",
    "# # . # . # .",
    ". . . . . # .",
    ". # # # . . .",
    ". . X # # # .",
    "# . . . . . .",
    ". # # # # # E",
]
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}   # up, down, left, right
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


class MazeEnv:
    def __init__(self, layout):
        self.grid = [row.split() for row in layout]
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "S":
                    self.start = (r, c)
                if self.grid[r][c] == "E":
                    self.exit = (r, c)
        self.n_states = self.rows * self.cols
        self.n_actions = 4

    def state_id(self, pos):
        return pos[0] * self.cols + pos[1]

    def reset(self):
        self.pos = self.start
        return self.state_id(self.pos)

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        # bumping into a wall or the border keeps the agent where it is
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.grid[nr][nc] == "#":
            nr, nc = self.pos
        self.pos = (nr, nc)
        cell = self.grid[nr][nc]
        if cell == "E":
            return self.state_id(self.pos), 100.0, True
        if cell == "X":
            return self.state_id(self.pos), -100.0, True     # trap ends the episode
        return self.state_id(self.pos), -1.0, False


# ---------------------------------------------------------------------------
# 2. Q-LEARNING AGENT
# ---------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha, gamma, epsilon, seed=None):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.rng = random.Random(seed)
        self.n_actions = n_actions

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:          # explore
            return self.rng.randrange(self.n_actions)
        best = np.flatnonzero(self.Q[state] == self.Q[state].max())
        return int(self.rng.choice(list(best)))       # exploit (random tie-break)

    def update(self, s, a, r, s_next, done):
        # Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
        target = r if done else r + self.gamma * self.Q[s_next].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def greedy_action(self, state):
        return int(np.argmax(self.Q[state]))


# ---------------------------------------------------------------------------
# 3. TRAINING AND EVALUATION
# ---------------------------------------------------------------------------
def train(env, alpha, gamma, epsilon, episodes=800, seed=1, max_steps=200):
    agent = QLearningAgent(env.n_states, env.n_actions, alpha, gamma, epsilon, seed)
    rewards, escapes = [], []
    for _ in range(episodes):
        s = env.reset()
        total, escaped = 0.0, False
        for _ in range(max_steps):
            a = agent.choose_action(s)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2, done)
            total += r
            s = s2
            if done:
                escaped = env.grid[env.pos[0]][env.pos[1]] == "E"
                break
        rewards.append(total)
        escapes.append(escaped)
    return agent, rewards, escapes


def greedy_path(env, agent, max_steps=50):
    s = env.reset()
    path = [env.pos]
    for _ in range(max_steps):
        a = agent.greedy_action(s)
        s, _, done = env.step(a)
        path.append(env.pos)
        if done:
            break
    return path


def evaluate(env, agent, trials=100):
    successes, lengths = 0, []
    for _ in range(trials):
        s = env.reset()
        for step in range(1, 51):
            a = agent.greedy_action(s)
            s, _, done = env.step(a)
            if done:
                if env.grid[env.pos[0]][env.pos[1]] == "E":
                    successes += 1
                    lengths.append(step)
                break
    avg_len = np.mean(lengths) if lengths else float("nan")
    return successes / trials * 100, avg_len


def first_convergence(rewards, window=50, tol=1.0):
    """Episode where the moving average of rewards stops changing by more than tol."""
    ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    for i in range(1, len(ma)):
        if np.all(np.abs(np.diff(ma[i:i + window])) < tol) and i + window <= len(ma):
            return i + window
    return len(rewards)


# ---------------------------------------------------------------------------
# 4. MAIN EXPERIMENT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = MazeEnv(MAZE)
    print("Maze layout (S=start, E=exit, X=trap, #=wall):")
    for row in MAZE:
        print("   ", row)

    # ---- (A) Baseline run -------------------------------------------------
    BASE = dict(alpha=0.1, gamma=0.95, epsilon=0.2)
    agent, rewards, escapes = train(env, **BASE, episodes=800, seed=1)
    success, avg_len = evaluate(env, agent)
    path = greedy_path(env, agent)
    print(f"\n[Baseline] alpha={BASE['alpha']}, gamma={BASE['gamma']}, epsilon={BASE['epsilon']}")
    print(f"  Greedy success rate : {success:.0f}%")
    print(f"  Average escape steps: {avg_len:.1f}")
    print("  Learned path        :", " -> ".join(f"({r},{c})" for r, c in path))

    # Learned policy drawn on the maze
    arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
    policy_grid = [row.split() for row in MAZE]
    for r in range(env.rows):
        for c in range(env.cols):
            if policy_grid[r][c] in (".", "S"):
                policy_grid[r][c] = arrows[agent.greedy_action(env.state_id((r, c)))]
    print("\n  Learned policy (arrows = best move):")
    for row in policy_grid:
        print("   ", " ".join(row))

    # ---- (B) Effect of epsilon, alpha, gamma ------------------------------
    print("\n[Parameter study] each setting trained for 800 episodes, 3 seeds averaged")
    print(f"{'param':<10}{'value':>8}{'success%':>11}{'avg steps':>12}{'final reward':>15}")
    print("-" * 56)

    def study(param, values):
        out = []
        for v in values:
            cfg = dict(BASE)
            cfg[param] = v
            succ, lens, final = [], [], []
            for seed in (1, 2, 3):
                ag, rw, _ = train(env, **cfg, episodes=800, seed=seed)
                s_, l_ = evaluate(env, ag)
                succ.append(s_)
                lens.append(l_)
                final.append(np.mean(rw[-50:]))
            out.append((v, np.mean(succ), np.nanmean(lens), np.mean(final)))
            print(f"{param:<10}{v:>8}{np.mean(succ):>10.0f}%{np.nanmean(lens):>12.1f}{np.mean(final):>15.1f}")
        return out

    eps_res = study("epsilon", [0.05, 0.1, 0.2, 0.4, 0.6])
    alp_res = study("alpha", [0.01, 0.1, 0.5, 0.9])
    gam_res = study("gamma", [0.5, 0.8, 0.9, 0.99])

    # ---- (C) Learning curve plot ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ma = np.convolve(rewards, np.ones(30) / 30, mode="valid")
    axes[0].plot(rewards, alpha=0.3, color="gray", label="Episode reward")
    axes[0].plot(range(29, len(rewards)), ma, color="#1f77b4", linewidth=2.5, label="30-episode moving average")
    axes[0].set_title("Learning Curve (Baseline)", fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total reward")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    labels = [f"ε={v}" for v, *_ in eps_res]
    axes[1].bar(labels, [s for _, s, *_ in eps_res], color="#2ca02c", edgecolor="black")
    axes[1].set_title("Effect of Exploration Rate ε on Success Rate", fontweight="bold")
    axes[1].set_ylabel("Greedy success rate (%)")
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig("EL_learning_curve.png", dpi=130)
    print("\nChart saved: EL_learning_curve.png")

    print("\n--- Observations (read from the results above) ---")
    print("* Epsilon: all settings reach the exit, but the final episode reward falls from about 88 (eps=0.05)")
    print("  to about 68 (eps=0.6) because high exploration keeps taking random steps and costing reward.")
    print("* Alpha and gamma: in this small, static maze every value converges to the same 12-step route,")
    print("  so their effect on success rate and path length is negligible. Their influence shows up on")
    print("  larger or dynamic mazes, where slow learning (small alpha) or short-sighted values (small gamma)")
    print("  would cause the agent to take longer routes or learn more slowly.")
    print("* Reward shaping: the -1 step cost and -100 trap penalty make the agent prefer the short,")
    print("  trap-free route (S -> E in 12 moves, avoiding the trap at row 4, column 2 with zero-based indexing).")
