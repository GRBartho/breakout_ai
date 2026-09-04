"""
Deep Q-Network (DQN) agent for Atari Breakout, using PyTorch + CUDA.

Requirements (install into your venv first):
    pip install torch numpy gymnasium "gymnasium[atari]" "gymnasium[accept-rom-license]" ale-py
    pip install moviepy   # optional, only needed for --video-dir

Usage:
    python breakout_dqn.py --train                        # train from scratch (headless, fastest)
    python breakout_dqn.py --train --render                # train with a live window (slow, real-time viewing)
    python breakout_dqn.py --train --video-dir videos/     # periodically save mp4 clips of the agent's progress
    python breakout_dqn.py --play checkpoint.pt             # watch a trained agent play

The script will automatically use CUDA if available (checks torch.cuda.is_available()),
otherwise falls back to CPU.
"""

import argparse
import collections
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordVideo,
)

# ale-py's Atari environments (the "ALE/..." namespace, e.g. ALE/Breakout-v5)
# are registered with gymnasium as a side effect of this call. Without it,
# gym.make(ENV_ID) raises `NamespaceNotFound: Namespace ALE not found` even
# when ale-py is installed, depending on how entry-point plugin discovery
# resolved in your environment.
gym.register_envs(ale_py)

# --------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------

ENV_ID = "ALE/Breakout-v5"


def make_env(render_mode=None):
    """Create a preprocessed Breakout environment.

    AtariPreprocessing handles: grayscale conversion, 84x84 resize, frame
    skipping (4), max-pooling over consecutive frames, and terminal-on-life-loss.
    FrameStackObservation stacks the last 4 frames so the agent can perceive motion.
    """
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


class DQN(nn.Module):
    """Convolutional network mapping stacked 84x84 grayscale frames to Q-values."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        conv_out_size = self._get_conv_out((4, 84, 84))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x: (batch, 4, 84, 84) uint8 -> normalize to [0, 1]
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch], dtype=np.uint8)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.uint8)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


class DQNAgent:
    def __init__(
        self,
        n_actions: int,
        device: torch.device,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 100_000,
        batch_size: int = 32,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay_steps: int = 1_000_000,
        target_update_freq: int = 10_000,
    ):
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq

        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.steps_done = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps_done / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.from_numpy(np.array(state, dtype=np.uint8)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["policy_net"])
        self.steps_done = checkpoint.get("steps_done", 0)


# --------------------------------------------------------------------------
# Training / evaluation loops
# --------------------------------------------------------------------------


def record_eval_video(agent: DQNAgent, video_dir: str, name_prefix: str):
    """Run one greedy episode in a fresh env and save it as an mp4.

    Uses a separate env instance so recording never slows down the main
    (headless) training loop. Requires `pip install moviepy`.
    """
    env = gym.make(ENV_ID, frameskip=1, render_mode="rgb_array")
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env = RecordVideo(env, video_folder=video_dir, name_prefix=name_prefix, episode_trigger=lambda ep: True)

    state, _ = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(state, greedy=True)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    env.close()
    print(f"Saved eval video ({name_prefix}, reward={total_reward}) to {video_dir}")


def train(
    num_frames: int = 2_000_000,
    warmup_frames: int = 50_000,
    checkpoint_path: str = "breakout_dqn.pt",
    checkpoint_every: int = 50_000,
    log_every: int = 1_000,
    render: bool = False,
    video_dir: str = None,
    video_every_frames: int = 100_000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(render_mode="human" if render else None)
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions=n_actions, device=device)

    episode_reward = 0.0
    episode_count = 0
    recent_rewards = collections.deque(maxlen=100)

    state, _ = env.reset()
    start_time = time.time()

    for frame_idx in range(1, num_frames + 1):
        agent.steps_done = frame_idx

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.replay_buffer.push(np.array(state), action, reward, np.array(next_state), done)
        state = next_state
        episode_reward += reward

        if frame_idx > warmup_frames:
            agent.optimize()

        if done:
            recent_rewards.append(episode_reward)
            episode_count += 1
            episode_reward = 0.0
            state, _ = env.reset()

        if frame_idx % log_every == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            elapsed = time.time() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            print(
                f"frame={frame_idx} episodes={episode_count} "
                f"avg_reward(100)={avg_reward:.2f} eps={agent.epsilon():.3f} "
                f"fps={fps:.1f}"
            )

        if frame_idx % checkpoint_every == 0:
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        if video_dir and frame_idx % video_every_frames == 0:
            record_eval_video(agent, video_dir, name_prefix=f"frame_{frame_idx}")

    agent.save(checkpoint_path)
    print(f"Training complete. Final checkpoint saved to {checkpoint_path}")
    env.close()


def play(checkpoint_path: str, episodes: int = 5, render: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(render_mode="human" if render else None)
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions=n_actions, device=device)

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    agent.load(checkpoint_path)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {ep + 1}: reward={total_reward}")

    env.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DQN agent for Atari Breakout (PyTorch + CUDA)")
    parser.add_argument("--train", action="store_true", help="Train a new agent")
    parser.add_argument("--play", type=str, metavar="CHECKPOINT", help="Watch a trained agent play")
    parser.add_argument("--frames", type=int, default=2_000_000, help="Number of training frames")
    parser.add_argument("--checkpoint", type=str, default="breakout_dqn.pt", help="Checkpoint file path")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to play")
    parser.add_argument(
        "--render",
        action="store_true",
        help="During --train, open a live window showing the agent play in real time (slows training a lot)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="During --train, periodically save an mp4 of a greedy eval episode into this directory (requires `pip install moviepy`)",
    )
    parser.add_argument(
        "--video-every",
        type=int,
        default=100_000,
        help="Frames between periodic eval video recordings (used with --video-dir)",
    )
    args = parser.parse_args()

    if args.train:
        train(
            num_frames=args.frames,
            checkpoint_path=args.checkpoint,
            render=args.render,
            video_dir=args.video_dir,
            video_every_frames=args.video_every,
        )
    elif args.play:
        play(checkpoint_path=args.play, episodes=args.episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
