"""
Deep Q-Network (DQN) agent for Atari Breakout, using PyTorch + CUDA.

================================================================================
WHAT THIS FILE IMPLEMENTS, IN PLAIN ENGLISH
================================================================================
This is an implementation of the algorithm from the DeepMind paper
"Human-level control through deep reinforcement learning" (Mnih et al., 2015),
usually just called "DQN" (Deep Q-Network). It's the algorithm that first
learned to play dozens of Atari games directly from raw pixels, with no
hand-crafted features, and Breakout is one of its signature demos.

The core idea, at a very high level:

  1. An "agent" (our neural network) looks at the screen and picks an action
     (move the paddle left, right, or stay, or fire the ball).
  2. The game responds with a new screen and a reward (+1 per brick broken).
  3. We store every (screen, action, reward, next_screen) experience in a
     big memory buffer.
  4. We repeatedly sample random batches from that memory and train the
     network to better predict "how much future reward will I get if I take
     this action in this situation" — this predicted quantity is called a
     Q-value (Q for "Quality").
  5. Over millions of these experiences, the network's Q-value estimates
     become accurate enough that "always pick the action with the highest
     predicted Q-value" turns into genuinely good play.

Everything else in this file (the replay buffer, the target network, epsilon
decay, frame preprocessing) exists to make that training process numerically
stable — naively implementing step 4 on raw, correlated, non-stationary data
diverges almost immediately in practice.

================================================================================
REQUIREMENTS (install into your venv first)
================================================================================
    pip install torch numpy gymnasium "gymnasium[atari]" "gymnasium[accept-rom-license]" ale-py
    pip install moviepy   # optional, only needed for --video-dir

================================================================================
USAGE
================================================================================
    python breakout_dqn.py --train                        # train from scratch (headless, fastest)
    python breakout_dqn.py --train --render                # train with a live window (slow, real-time viewing)
    python breakout_dqn.py --train --video-dir videos/     # periodically save mp4 clips of the agent's progress
    python breakout_dqn.py --play checkpoint.pt             # watch a trained agent play

The script automatically uses CUDA if available (checks torch.cuda.is_available()),
otherwise falls back to CPU. Training on CPU is possible but very slow (Atari DQN
is typically trained on the order of millions to tens of millions of frames).
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
#
# Why we can't just feed the raw game to the network:
#
#   The raw Atari screen is a 210x160 RGB image (three color channels), and
#   the game engine runs at 60 frames per second. Feeding that directly to a
#   neural network, one raw frame at a time, has two problems:
#
#     (a) It's needlessly expensive: 210*160*3 = 100,800 numbers per frame,
#         most of which (exact color, fine resolution) carries no information
#         relevant to "should the paddle move left or right".
#
#     (b) A single frame can't show motion. Looking at one frame you can't
#         tell whether the ball is moving up or down, or how fast. But motion
#         is exactly what matters for playing Breakout.
#
#   The wrappers below solve both problems and reproduce the standard
#   preprocessing pipeline from the original DeepMind DQN paper.

ENV_ID = "ALE/Breakout-v5"


def make_env(render_mode=None):
    """Create a preprocessed Breakout environment.

    The pipeline, in order:

    1. gym.make(..., frameskip=1): get the *unmodified* Atari env. We set
       frameskip=1 here (i.e. "don't skip") because we do our own frame
       skipping explicitly below via AtariPreprocessing's frame_skip=4 -
       ALE's built-in frameskip and AtariPreprocessing's frame_skip would
       otherwise stack, skipping frames twice.

    2. AtariPreprocessing wraps the raw env and applies, in this order:
         - noop_max=30: at the start of every episode, take between 0 and 30
           random "do nothing" actions before the agent gets control. This
           randomizes the starting state so the agent can't just memorize a
           fixed opening sequence of moves - it forces genuine generalization.
         - frame_skip=4: repeat the agent's chosen action for 4 consecutive
           game frames, only asking the agent to pick a new action every 4th
           frame. Atari games run at 60fps, but decisions don't need to be
           made that often, and skipping frames both speeds up training
           (4x fewer network forward passes) and simplifies the effective
           action-space-over-time. Internally, the *last two* of those 4
           raw frames are combined with a pixel-wise max (to eliminate
           flicker some Atari games have when sprites are drawn on
           alternating frames only - not really an issue for Breakout, but
           it's the standard, safe default).
         - screen_size=84: resize/crop the frame down to 84x84 pixels. This
           is small enough to keep the convolutional network fast, while
           still preserving enough detail to see the ball, paddle, and
           bricks.
         - grayscale_obs=True: collapse RGB to a single grayscale channel.
           Color doesn't carry meaningful information in Breakout, so this
           reduces data volume by 3x for free.
         - terminal_on_life_loss=True: Breakout gives you 5 lives per game.
           With this flag, we tell the RL algorithm an "episode" ends every
           time you lose a life (not just when the game truly ends after all
           5 lives are gone). This gives the agent a much more immediate
           learning signal ("that specific sequence of actions cost me a
           life") instead of one delayed signal every 5 lives.
         - scale_obs=False: keep pixel values as uint8 (0-255) instead of
           float32 (0.0-1.0) at the environment level. This is purely a
           memory optimization - storing millions of frames in the replay
           buffer as uint8 instead of float32 cuts RAM usage 4x. We convert
           to float and normalize by 255 later, inside the network itself
           (see DQN.forward below), right before it's needed.

    3. FrameStackObservation(stack_size=4): stack the last 4 preprocessed
       frames together into a single (4, 84, 84) observation. This is what
       actually solves the "can't see motion from one frame" problem - by
       giving the network 4 consecutive frames at once, it can infer
       velocity and direction (e.g. "the ball's position shifted down-left
       over these 4 frames, so it's heading down-left") the same way flipping
       through a few consecutive comic-strip panels shows you motion that a
       single panel can't.

    The end result: each observation the agent sees is a (4, 84, 84) uint8
    array - 4 stacked grayscale 84x84 frames - instead of a raw
    (210, 160, 3) RGB frame.
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
    """Convolutional network mapping stacked 84x84 grayscale frames to Q-values.

    This is the "brain" of the agent. It's a function approximator for the
    Q-function, Q(state, action): given a state (our 4-frame stack), it
    outputs one number per possible action, estimating "the total future
    reward I expect to get, from now until the game ends, if I take this
    action right now and then play optimally afterward".

    We don't need to run the network once per action - it outputs all
    Q-values for all actions in a single forward pass, and we just read off
    whichever action's output is highest.

    Architecture (identical in spirit to the original DeepMind Atari DQN
    paper): three convolutional layers extract increasingly abstract visual
    features (edges -> shapes -> object configurations), followed by two
    fully-connected layers that turn those features into per-action
    Q-value estimates.

        input:  (batch, 4, 84, 84)      4 stacked grayscale frames
        conv1:  32 filters, 8x8, stride 4   -> (batch, 32, 20, 20)
        conv2:  64 filters, 4x4, stride 2   -> (batch, 64, 9, 9)
        conv3:  64 filters, 3x3, stride 1   -> (batch, 64, 7, 7)
        flatten:                             -> (batch, 3136)
        fc1:    3136 -> 512
        fc2:    512 -> n_actions            (one Q-value per action)

    The large strides (4, then 2) in the early conv layers aggressively
    downsample the spatial resolution, which is both a speed optimization
    and a form of translation-invariant feature pooling (similar in effect
    to max-pooling, but learned).
    """

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
        # We don't hardcode the flattened size (3136) - instead we compute it
        # by actually running a dummy zero tensor through the conv stack once.
        # This makes the code robust to changes in input size or conv config
        # without needing to hand-recompute the arithmetic.
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
        """Run a batch of stacked frames through the network to get Q-values.

        x arrives as (batch, 4, 84, 84) with dtype uint8, values in [0, 255]
        (this is exactly the format the replay buffer stores states in, to
        save memory - see ReplayBuffer below). We convert to float and scale
        into [0, 1] here, right before the conv layers need it, so that only
        one batch at a time is ever expanded into (memory-hungry) floats
        rather than the whole replay buffer.

        Returns a (batch, n_actions) tensor of raw Q-value estimates.
        """
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------
#
# Why we need this at all:
#
#   The most natural way to train would be: play a step, immediately train
#   the network on that one step's outcome, play the next step, train again,
#   and so on. This fails badly for two related reasons:
#
#     1. Correlation: consecutive frames in a game are extremely similar to
#        each other (the ball moves a tiny bit, the paddle moves a tiny bit).
#        Training a neural network on a long run of highly similar,
#        correlated examples in a row makes it overfit to whatever narrow
#        situation it's currently in, and can cause the network's weights to
#        oscillate or diverge instead of converging.
#
#     2. Non-stationarity: as the network's weights change, the policy (which
#        actions it prefers) changes, which changes what data it collects
#        next. Training directly on live, freshly-collected data means
#        you're always learning from a policy-in-progress, which resembles
#        "chasing a moving target".
#
#   The fix, from the original DQN paper, is "experience replay": rather
#   than training on the most recent step, store every step's experience
#   in a large FIFO memory buffer, and train on random batches sampled from
#   the *whole* buffer. Random sampling breaks the temporal correlation
#   (a training batch might contain frames from a hundred different points
#   in a hundred different past episodes), and because the buffer holds a
#   long history, it smooths out the non-stationarity problem too.

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    """A fixed-capacity FIFO memory of past (state, action, reward,
    next_state, done) experiences, with uniform random sampling.

    Implemented with collections.deque(maxlen=capacity): once the buffer is
    full, appending a new transition automatically evicts the oldest one, so
    memory usage stays bounded no matter how long training runs. This does
    mean that once training has run past `capacity` steps, the buffer only
    ever contains the most recent `capacity` experiences - old, "how it
    played when it was much worse" experiences eventually age out. This
    default capacity is 100,000 transitions, which at (4, 84, 84) uint8 per
    state is roughly 100,000 * 2 * 4*84*84 bytes = ~5.6 GB of RAM (states are
    stored twice, once as `state` and once as `next_state`, so it isn't
    quite that much thanks to Python object reuse in practice, but budget
    for it).
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, *args):
        """Record one transition (state, action, reward, next_state, done)."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        """Draw `batch_size` transitions uniformly at random (without
        replacement within the batch) and return them as stacked numpy
        arrays, ready to be converted to tensors.
        """
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
    """Everything needed to act in the environment and learn from experience:
    the two networks (policy + target), the optimizer, the replay buffer, and
    the exploration schedule.

    Hyperparameters (the defaults below match the original DQN paper fairly
    closely, adapted to a more modest training budget):

        lr (1e-4): the Adam optimizer's learning rate - how big a step to
            take on the network's weights after each gradient computation.

        gamma (0.99): the discount factor. When estimating "total future
            reward", a reward received now counts fully, a reward 1 step in
            the future counts *0.99, 2 steps in the future counts *0.99^2,
            and so on. This both models the intuition that immediate rewards
            are worth slightly more than distant ones, and - importantly -
            keeps the sum of infinitely many future rewards mathematically
            finite (a discount factor < 1 guarantees the geometric-like sum
            converges).

        buffer_capacity (100,000): see ReplayBuffer above.

        batch_size (32): how many transitions to sample per training step.
            32 is the classic DQN-paper value; it's small enough to keep
            memory/compute per step cheap, while still averaging out enough
            random samples to give a reasonably stable gradient estimate.

        eps_start / eps_end / eps_decay_steps: control the exploration
            schedule - see epsilon() and select_action() below.

        target_update_freq (10,000): how often (in frames) to sync the
            target network's weights to the policy network's - see the
            "target network" explanation in optimize() below.
    """

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

        # The policy network: this is the one we actually train, and the one
        # whose Q-value predictions decide which action to take.
        self.policy_net = DQN(n_actions).to(device)

        # The target network: a second, separate copy of the same
        # architecture, used ONLY to compute training targets (see
        # optimize() below) - never used to pick actions, never directly
        # trained by the optimizer. It starts as an exact copy of the policy
        # network's weights and is periodically re-synced (every
        # target_update_freq frames). Its whole purpose is explained in
        # optimize() below; in short, it keeps the "target" that the policy
        # network is chasing fixed for a while, instead of letting the
        # target shift on every single training step (which would be like
        # trying to hit a target that's always moving in response to your
        # last shot).
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # eval() disables training-only behaviors (e.g. dropout, batchnorm
        # running-stat updates). This network has none of those layers, so
        # it's mostly documentation of intent here, but it's good practice:
        # this network is never meant to be trained directly.
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Counts environment steps (frames) seen so far. Used both to
        # compute the current exploration rate (epsilon()) and to decide
        # when to sync the target network.
        self.steps_done = 0

    def epsilon(self) -> float:
        """Compute the current exploration rate (epsilon) via linear decay.

        This implements "epsilon-greedy" exploration's schedule: epsilon
        starts at eps_start (1.0 = always act randomly) and linearly
        decreases to eps_end (0.1 = mostly greedy, but still 10% random)
        over eps_decay_steps frames, then stays at eps_end forever after.

        Why decay epsilon at all, rather than picking one fixed value?
        Early in training, the network's Q-value estimates are essentially
        random noise (the network hasn't learned anything yet), so acting
        "greedily" with respect to them is no better than acting randomly -
        we might as well explore the environment as broadly as possible to
        gather data. As training progresses and the Q-value estimates
        become more accurate, greedy actions start being genuinely better
        than random ones, so we want to exploit that knowledge more and
        more, while still keeping a small amount of permanent randomness
        (eps_end=0.1) so the agent never completely stops exploring and
        can't get permanently stuck always repeating one suboptimal
        strategy.
        """
        frac = min(1.0, self.steps_done / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Pick an action for the given state, using epsilon-greedy
        exploration (or purely greedy, if greedy=True, which is used at
        evaluation/play time).

        With probability epsilon: pick a uniformly random action
        ("explore"). Otherwise: run the state through the policy network and
        pick the action with the highest predicted Q-value ("exploit" - use
        our current best knowledge).

        This is the "exploration vs. exploitation" trade-off that's central
        to all of reinforcement learning: pure exploitation risks getting
        stuck on a mediocre strategy the agent found early and never
        discovering a better one; pure exploration never uses what's been
        learned to actually play well. Epsilon-greedy is the simplest
        possible way to balance the two, and it's what the original DQN
        paper uses.
        """
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = torch.from_numpy(np.array(state, dtype=np.uint8)).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def optimize(self):
        """Run one training step: sample a batch from replay, compute the
        DQN loss, and update the policy network's weights by one step of
        gradient descent.

        This is the heart of the algorithm. Step by step:

        1. Bail out early if we don't have enough experience yet to fill a
           full batch (this only matters right at the very start of
           training).

        2. Sample a random batch of `batch_size` past transitions from the
           replay buffer (see ReplayBuffer above for why random sampling
           matters), and move them onto the training device (GPU, if
           available).

        3. Compute the network's *current* Q-value estimate for the action
           that was actually taken in each sampled transition:

               q_values = policy_net(states)[actions taken]

           `.gather(1, actions_t.unsqueeze(1))` is just an indexing
           operation: policy_net(states) gives a (batch, n_actions) tensor
           of Q-values for *every* action, but we only care about the
           Q-value for the specific action that was recorded in that
           transition - gather selects exactly that column, per row.

        4. Compute the training *target* for each transition, using the
           Bellman equation - the mathematical relationship that a correct
           Q-function must satisfy:

               Q(s, a) = reward + gamma * max_a'[ Q(next_s, a') ]

           In words: "the value of taking action a in state s should equal
           the immediate reward you get, plus the discounted value of the
           *best* action available in whatever state you land in next".
           This is a recursive definition - it defines Q in terms of Q - and
           training pushes the network's current predictions to satisfy it
           more and more closely over time. This particular family of
           methods is called "temporal difference learning".

           Two important details in the code:

             - `self.target_net` (not `self.policy_net`) is used to compute
               max_a'[Q(next_s, a')]. This is the "target network" trick
               mentioned above: if we used the same network that we're
               actively updating to both estimate current values AND define
               the targets those values are being pushed toward, the target
               would shift underneath us on every single gradient step
               (because updating the network to better match today's target
               immediately changes tomorrow's target too). That's a moving
               target that provably tends to cause oscillation/divergence.
               Using a separate, slowly-updated copy (synced only every
               target_update_freq frames, in the `if` block near the bottom
               of this method) holds the target still for long enough that
               training converges smoothly.

             - `(1.0 - dones_t)`: if the transition ended the episode
               (`done=True`), there *is* no "next state" whose value should
               be added - the game (or life) just ended. Multiplying by
               (1 - done) zeroes out the future-value term exactly in that
               case, so the target correctly reduces to just the immediate
               reward.

           `torch.no_grad()` wraps this whole computation because the target
           is meant to be a fixed number to regress toward, not something we
           backpropagate through - gradients should only flow through the
           policy network's *current* prediction (step 3), not through how
           the target itself was computed.

        5. Compute the loss between the current prediction (step 3) and the
           target (step 4), using the Huber loss (`smooth_l1_loss`) rather
           than plain mean-squared-error. Huber loss behaves like MSE for
           small errors but like mean-absolute-error for large ones, which
           makes training much less sensitive to occasional huge outlier
           errors (e.g. right after the target network resyncs, or from a
           rare surprising reward) - those outliers would otherwise produce
           huge squared-error gradients that destabilize training.

        6. Backpropagate the loss, clip the gradient norm to 10.0 (another
           stability safeguard - it caps how large a single weight update
           can be, preventing rare huge gradients from throwing the
           network's weights way off), and step the optimizer.

        7. Every target_update_freq frames, copy the policy network's
           current weights into the target network (the "target network
           sync" mentioned above).
        """
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

        # Current Q-value estimate for the action actually taken in each
        # sampled transition (see step 3 in the docstring above).
        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Bellman-equation training target, computed with the frozen target
        # network so it doesn't shift as the policy network is updated
        # (see step 4 in the docstring above).
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically sync target_net <- policy_net (see the "target
        # network" explanation above).
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """Persist the policy network's weights and training progress to
        disk. Note we don't need to separately save the target network -
        it's always derivable from the policy network's weights (they're
        periodically synced), so load() below just re-derives it.
        """
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str):
        """Restore a checkpoint saved by save(). Both policy_net and
        target_net are loaded with the same weights - see the note in
        save() above.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["policy_net"])
        self.steps_done = checkpoint.get("steps_done", 0)


# --------------------------------------------------------------------------
# Training / evaluation loops
# --------------------------------------------------------------------------


def record_eval_video(agent: DQNAgent, video_dir: str, name_prefix: str):
    """Run one greedy (epsilon=0, i.e. no random exploration moves) episode
    in a fresh environment instance and save it as an mp4, so training
    progress can be watched visually without slowing down the main training
    loop.

    This creates its OWN separate environment (rather than reusing the
    training env) specifically so it can run with render_mode="rgb_array"
    (needed to capture frames for video) independent of whatever render
    mode the main training loop is using, and so recording an episode never
    interrupts or slows the frame-by-frame training loop itself - it happens
    as a brief, self-contained side trip using the agent's current weights,
    then throws its own environment away.

    Requires `pip install moviepy` (used internally by gymnasium's
    RecordVideo wrapper to encode the mp4).
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
    """The main training loop.

    Parameters:
        num_frames: total number of environment steps (post frame-skip - see
            make_env's frame_skip=4) to train for. 2,000,000 is a modest
            budget; the original DQN paper trains for ~50 million frames per
            game to reach human-level play. Expect a visibly-improving-but-
            still-weak agent at 2M frames, and meaningfully better play in
            the tens of millions.
        warmup_frames: number of initial frames to spend purely on
            experience collection (acting, mostly randomly since epsilon
            starts near 1.0, and filling the replay buffer) BEFORE any
            training steps run. This ensures optimize() has a reasonably
            large and diverse pool of experience to sample from from the
            very first gradient step, instead of training on a tiny, highly
            repetitive handful of early transitions.
        checkpoint_path / checkpoint_every: where to periodically save
            model weights (via DQNAgent.save), and how often (in frames).
        log_every: how often (in frames) to print a progress line to the
            console (frame count, episode count, recent average reward,
            current epsilon, and training speed in frames/sec).
        render: if True, opens a live "human" render window so you can watch
            the agent play in real time as it trains. This renders every
            single frame synchronously, which can meaningfully reduce
            training throughput - prefer video_dir for a lower-overhead way
            to visualize progress.
        video_dir / video_every_frames: if video_dir is set, every
            video_every_frames frames, run one greedy evaluation episode
            (via record_eval_video, using a separate environment so it
            doesn't disturb the main training loop) and save it as an mp4
            in that directory - a good way to literally watch the agent get
            better over the course of training.

    The loop body, once per frame:
        1. Pick an action (agent.select_action - epsilon-greedy).
        2. Step the environment with that action, observing the resulting
           next state, reward, and whether the episode ended.
        3. Store this (state, action, reward, next_state, done) transition
           in the replay buffer.
        4. Advance `state` to `next_state` for the next iteration, and
           accumulate the reward for logging.
        5. Once past warmup_frames, run one training step (agent.optimize()
           - see its docstring for the full Bellman-equation/target-network
           explanation).
        6. If the episode ended, reset the environment and record the
           episode's total reward for the rolling average shown in logs.
        7. Periodically: print a log line, save a checkpoint, and/or record
           an evaluation video.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = make_env(render_mode="human" if render else None)
    n_actions = env.action_space.n
    agent = DQNAgent(n_actions=n_actions, device=device)

    episode_reward = 0.0
    episode_count = 0
    # A rolling window of the last 100 episodes' total rewards, used to
    # report a smoothed "average reward" in the log line below - a single
    # episode's reward is noisy, but the trend of the last 100 is a much
    # more reliable signal of whether the agent is actually improving.
    recent_rewards = collections.deque(maxlen=100)

    state, _ = env.reset()
    start_time = time.time()

    for frame_idx in range(1, num_frames + 1):
        agent.steps_done = frame_idx

        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # `terminated` = the episode ended for an in-game reason (in our
        # case, terminal_on_life_loss means this fires on every life lost).
        # `truncated` = the episode was cut off for an external reason (e.g.
        # a max-steps time limit), unrelated to in-game outcome. We treat
        # both the same way here (end the episode, reset), which is the
        # standard simplification for this kind of training loop.
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
    """Load a trained checkpoint and watch it play, acting purely greedily
    (no random exploration - see select_action's greedy=True path), for a
    fixed number of episodes.
    """
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
