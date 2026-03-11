"""
Full training script for the multi-agent racing environment.

Trains a PPO agent with:
- Curriculum learning (start with fewer opponents, scale up)
- Periodic evaluation with detailed metrics
- Self-play support (optional)
- TensorBoard logging

Usage:
    python train.py
    python train.py --total-timesteps 2000000 --num-agents 6
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import RacingEnv, make_racing_env

UID = "706605995"  # Your UID for submission
NAME = "HO SHU HAN"  # Your agent's name


class RacingMetricsCallback(BaseCallback):
    """Logs additional racing-specific metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_lengths = []
        self._route_completions = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
            if "route_completion" in info:
                self._route_completions.append(info["route_completion"])

        if len(self._episode_rewards) >= 10:
            self.logger.record("racing/mean_reward", np.mean(self._episode_rewards))
            self.logger.record("racing/mean_length", np.mean(self._episode_lengths))
            if self._route_completions:
                self.logger.record("racing/mean_route_completion", np.mean(self._route_completions))
            self._episode_rewards.clear()
            self._episode_lengths.clear()
            self._route_completions.clear()

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train a racing agent (full example)")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-train-envs", type=int, default=8)
    parser.add_argument("--num-eval-envs", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--opponent-policy", type=str, default="aggressive",
                        choices=["random", "aggressive", "still"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Racing - Full Training Example")
    print("=" * 60)
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Train envs: {args.num_train_envs}")
    print(f"  Agents per race: {args.num_agents}")
    print(f"  Opponent: {args.opponent_policy}")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Create environments
    train_envs = SubprocVecEnv(
        [make_racing_env(
            rank=i,
            num_agents=args.num_agents,
            opponent_policy=args.opponent_policy,
        ) for i in range(args.num_train_envs)]
    )

    eval_envs = SubprocVecEnv(
        [make_racing_env(
            rank=100 + i,
            num_agents=args.num_agents,
            opponent_policy=args.opponent_policy,
        ) for i in range(args.num_eval_envs)]
    )

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(args.save_freq // args.num_train_envs, 1),
            save_path=args.save_dir,
            name_prefix="racing_ppo",
        ),
        EvalCallback(
            eval_envs,
            best_model_save_path=os.path.join(args.save_dir, "best"),
            log_path=args.log_dir,
            eval_freq=max(args.eval_freq // args.num_train_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
        RacingMetricsCallback(),
    ]

    # Create PPO agent with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        train_envs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=0.5,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )

    print(f"\nPolicy architecture: {model.policy}")
    print(f"Observation space: {train_envs.observation_space}")
    print(f"Action space: {train_envs.action_space}")
    print()

    t0 = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )
    elapsed = time.time() - t0

    # Save final model
    final_path = os.path.join(args.save_dir, "racing_ppo_final")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Final model saved to {final_path}")

    # Auto-convert to submission format
    print("\nConverting to submission format...")
    convert_to_submission(model, os.path.join("agents", f"agent_{UID}"))
    print(f"Done! Example agent saved to agents/agent_{UID}/")

    train_envs.close()
    eval_envs.close()


def convert_to_submission(model, output_dir):
    """Extract policy from SB3 model and save as standalone agent."""
    os.makedirs(output_dir, exist_ok=True)
    policy = model.policy

    obs_dim = policy.observation_space.shape[0]
    action_dim = policy.action_space.shape[0]

    # Extract MLP extractor layers
    pi_layers = policy.mlp_extractor.policy_net
    hidden_sizes = []
    state_dict = {}

    for i, layer in enumerate(pi_layers):
        if isinstance(layer, torch.nn.Linear):
            hidden_sizes.append(layer.out_features)
            state_dict[f"features.{i}.weight"] = layer.weight.data.clone()
            state_dict[f"features.{i}.bias"] = layer.bias.data.clone()

    state_dict["action_mean.weight"] = policy.action_net.weight.data.clone()
    state_dict["action_mean.bias"] = policy.action_net.bias.data.clone()

    checkpoint = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_sizes": hidden_sizes,
        "state_dict": state_dict,
    }
    torch.save(checkpoint, os.path.join(output_dir, "model.pt"))

    agent_code = '''"""Example trained racing agent."""

import os
import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.action_mean = nn.Linear(in_dim, action_dim)

    def forward(self, obs):
        x = self.features(obs)
        return self.action_mean(x)


class Policy:
    CREATOR_NAME = "__CREATOR_NAME__"
    CREATOR_UID = "__CREATOR_UID__"

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.obs_dim = checkpoint["obs_dim"]
        self.action_dim = checkpoint["action_dim"]
        hidden_sizes = checkpoint["hidden_sizes"]

        self.model = PolicyNetwork(self.obs_dim, self.action_dim, hidden_sizes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, obs):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.model(obs_tensor).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)
'''
    agent_code = agent_code.replace("__CREATOR_NAME__", NAME).replace("__CREATOR_UID__", UID)
    with open(os.path.join(output_dir, "agent.py"), "w") as f:
        f.write(agent_code)


if __name__ == "__main__":
    main()
