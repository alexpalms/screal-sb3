#!/usr/bin/env python3
"""
Minimal PPO training script for testing modifications to the PPO code.
Supports CartPole and Atari Pong environments.

Usage:
    python scripts/ppo_test.py --env cartpole
    python scripts/ppo_test.py --env pong
"""

import argparse

import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack


# Fixed seed for reproducibility
SEED = 42


def set_global_seeds(seed: int) -> None:
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def train_cartpole() -> PPO:
    """Train PPO on CartPole-v1."""
    env = make_vec_env("CartPole-v1", n_envs=1, seed=SEED)

    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=1,
        n_steps=512,
        batch_size=64,
        n_epochs=4,
        policy_kwargs=dict(net_arch=[64, 64]),
        device="cpu",  # CPU for determinism
    )
    model.learn(total_timesteps=10_000)
    model.save("ppo_cartpole")
    print("Model saved to ppo_cartpole.zip")
    return model


def train_pong() -> PPO:
    """Train PPO on Atari Pong."""
    env = make_atari_env("PongNoFrameskip-v4", n_envs=1, seed=SEED)
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        seed=SEED,
        verbose=1,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        learning_rate=2.5e-4,
        clip_range=0.1,
        device="cpu",  # CPU for determinism
    )
    model.learn(total_timesteps=10_000)
    model.save("ppo_pong")
    print("Model saved to ppo_pong.zip")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO test script")
    parser.add_argument(
        "--env",
        type=str,
        choices=["cartpole", "pong"],
        required=True,
        help="Environment to train on: 'cartpole' or 'pong'",
    )
    args = parser.parse_args()

    set_global_seeds(SEED)

    if args.env == "cartpole":
        train_cartpole()
    else:
        train_pong()


if __name__ == "__main__":
    main()
