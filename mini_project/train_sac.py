"""
Full training script for the multi-agent racing environment using SAC.

Trains a SAC (Soft Actor-Critic) agent with:
- Off-policy learning (better sample efficiency)
- Replay buffer for experience reuse
- Periodic evaluation with detailed metrics
- TensorBoard logging

Usage:
    python train_sac.py
    python train_sac.py --total-timesteps 2000000 --num-agents 6
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, sync_envs_normalization

from env import make_racing_env

UID = "706605995"  # Your UID for submission
NAME = "HO SHU HAN"  # Your agent's name


class RacingEvalCallback(EvalCallback):
    """EvalCallback that uses route_completion for best model selection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_route_completion = -np.inf
        self._route_completion_buffer = []

    def _log_route_completion_callback(self, locals_: dict, globals_: dict) -> None:
        """Collect route_completion when episode ends."""
        info = locals_.get("info", {})
        done = locals_.get("done", False)
        if done:
            rc = info.get("route_completion", 0.0)
            if rc is None:
                rc = 0.0
            self._route_completion_buffer.append(float(rc))

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way."
                    ) from e

            self._route_completion_buffer = []
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_route_completion_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                )

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)
            std_ep_length = np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            mean_route_completion = (
                np.mean(self._route_completion_buffer)
                if self._route_completion_buffer
                else 0.0
            )
            arrive_count = sum(1 for rc in self._route_completion_buffer if rc >= 0.99)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                print(
                    f"Route completion: {mean_route_completion:.2%}, "
                    f"arrive={arrive_count}/{self.n_eval_episodes}"
                )

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_route_completion", mean_route_completion)
            self.logger.record("eval/arrive_count", arrive_count)
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Use route_completion for best model (primary) with reward as tiebreaker
            metric = mean_route_completion + 0.001 * mean_reward
            if metric > self.best_mean_route_completion:
                self.best_mean_route_completion = float(metric)
                self.best_mean_reward = float(mean_reward)
                if self.verbose >= 1:
                    print("New best model (route_completion)!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


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
                self.logger.record(
                    "racing/mean_route_completion", np.mean(self._route_completions)
                )
            self._episode_rewards.clear()
            self._episode_lengths.clear()
            self._route_completions.clear()

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a racing agent with SAC (Soft Actor-Critic)"
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-train-envs", type=int, default=32)
    parser.add_argument("--num-eval-envs", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument(
        "--opponent-policy",
        type=str,
        default="aggressive",
        choices=["random", "aggressive", "still", "mixed"],
    )
    parser.add_argument("--save-dir", type=str, default="checkpoints_sac")
    parser.add_argument("--log-dir", type=str, default="logs_sac")
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-starts", type=int, default=15_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[512, 512, 256],
        help="Hidden layer sizes, e.g. 512 512 or 256 256 256",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Racing - SAC Training")
    print("=" * 60)
    print(f"  Algorithm: SAC (Soft Actor-Critic)")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    print(f"  Train envs: {args.num_train_envs}")
    print(f"  Agents per race: {args.num_agents}")
    print(f"  Opponent: {args.opponent_policy}")
    print(f"  LR: {args.lr}, Batch: {args.batch_size}, Grad steps: {args.gradient_steps}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Net arch: {args.net_arch}")

    # Create environments
    train_envs = SubprocVecEnv(
        [
            make_racing_env(
                rank=i,
                num_agents=args.num_agents,
                opponent_policy=args.opponent_policy,
            )
            for i in range(args.num_train_envs)
        ]
    )

    eval_envs = SubprocVecEnv(
        [
            make_racing_env(
                rank=100 + i,
                num_agents=args.num_agents,
                opponent_policy=args.opponent_policy,
            )
            for i in range(args.num_eval_envs)
        ]
    )

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=max(args.save_freq // args.num_train_envs, 1),
            save_path=args.save_dir,
            name_prefix="racing_sac",
        ),
        RacingEvalCallback(
            eval_envs,
            best_model_save_path=os.path.join(args.save_dir, "best"),
            log_path=args.log_dir,
            eval_freq=max(args.eval_freq // args.num_train_envs, 1),
            n_eval_episodes=10,
            deterministic=True,
        ),
        RacingMetricsCallback(),
    ]

    # Create SAC agent
    model = SAC(
        "MlpPolicy",
        train_envs,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        policy_kwargs=dict(
            net_arch=args.net_arch,
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
    final_path = os.path.join(args.save_dir, "racing_sac_final")
    model.save(final_path)
    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Final model saved to {final_path}")

    # Auto-convert to submission format: prefer best checkpoint over final
    output_dir = os.path.join("agents", f"agent_{UID}_sac")
    best_path = os.path.join(args.save_dir, "best", "best_model.zip")
    print("\nConverting to submission format...")
    if os.path.exists(best_path):
        print(f"  Using best checkpoint from {best_path}")
        best_model = SAC.load(best_path)
        convert_to_submission(best_model, output_dir)
    else:
        print("  Best checkpoint not found, using final model")
        convert_to_submission(model, output_dir)
    print(f"Done! SAC agent saved to {output_dir}/")

    train_envs.close()
    eval_envs.close()


def convert_to_submission(model, output_dir):
    """Extract SAC actor policy and save as standalone agent.

    SAC uses a different policy structure: latent_pi (ReLU) -> mu -> tanh
    """
    os.makedirs(output_dir, exist_ok=True)
    actor = model.policy.actor

    obs_dim = actor.observation_space.shape[0]
    action_dim = actor.action_space.shape[0]

    # SAC actor: latent_pi is Sequential of (Linear, ReLU, Linear, ReLU) for net_arch [256,256]
    # mu is Linear to action. We extract only Linear layers; ReLU has no params.
    latent_pi = actor.latent_pi
    hidden_sizes = []
    state_dict = {}
    feat_idx = 0
    for layer in latent_pi:
        if isinstance(layer, torch.nn.Linear):
            hidden_sizes.append(layer.out_features)
            state_dict[f"features.{feat_idx}.weight"] = layer.weight.data.clone()
            state_dict[f"features.{feat_idx}.bias"] = layer.bias.data.clone()
            feat_idx += 2  # skip ReLU slot in our Sequential(Linear, ReLU, Linear, ReLU)
    state_dict["action_mean.weight"] = actor.mu.weight.data.clone()
    state_dict["action_mean.bias"] = actor.mu.bias.data.clone()

    checkpoint = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_sizes": hidden_sizes,
        "state_dict": state_dict,
        "hidden_activation": "relu",  # SAC uses ReLU
        "squash_output": True,  # SAC uses tanh squashing
    }
    torch.save(checkpoint, os.path.join(output_dir, "model.pt"))

    agent_code = '''"""SAC-trained racing agent."""

import os
import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, use_relu=True, squash_output=True):
        super().__init__()
        self.squash_output = squash_output
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU() if use_relu else nn.Tanh())
            in_dim = h
        self.features = nn.Sequential(*layers)
        self.action_mean = nn.Linear(in_dim, action_dim)

    def forward(self, obs):
        x = self.features(obs)
        action = self.action_mean(x)
        if self.squash_output:
            action = torch.tanh(action)
        return action


class Policy:
    CREATOR_NAME = "__CREATOR_NAME__"
    CREATOR_UID = "__CREATOR_UID__"

    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.obs_dim = checkpoint["obs_dim"]
        self.action_dim = checkpoint["action_dim"]
        hidden_sizes = checkpoint["hidden_sizes"]
        use_relu = checkpoint.get("hidden_activation", "relu") == "relu"
        squash_output = checkpoint.get("squash_output", True)

        self.model = PolicyNetwork(
            self.obs_dim, self.action_dim, hidden_sizes,
            use_relu=use_relu, squash_output=squash_output
        )
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
    agent_code = agent_code.replace("__CREATOR_NAME__", NAME).replace(
        "__CREATOR_UID__", UID
    )
    with open(os.path.join(output_dir, "agent.py"), "w") as f:
        f.write(agent_code)


if __name__ == "__main__":
    main()
