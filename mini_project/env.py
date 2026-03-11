"""
Racing environment wrapper for single-agent training.

Wraps MetaDrive's MultiAgentRacingEnv into a standard gymnasium.Env interface.
Supports training with configurable opponent policies including self-play.
"""

import gymnasium
import numpy as np
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv


def random_opponent(obs, agent_id):
    return np.random.uniform(-1, 1, size=(2,)).astype(np.float32)


def aggressive_opponent(obs, agent_id):
    return np.array([0.0, 1.0], dtype=np.float32)


def still_opponent(obs, agent_id):
    return np.array([0.0, 0.0], dtype=np.float32)


def _mixed_opponent(obs, agent_id):
    """Curriculum: randomly sample from random, aggressive, still each step."""
    policy = np.random.choice([random_opponent, aggressive_opponent, still_opponent])
    return policy(obs, agent_id)


OPPONENT_POLICIES = {
    "random": random_opponent,
    "aggressive": aggressive_opponent,
    "still": still_opponent,
    "mixed": _mixed_opponent,
}


class SelfPlayOpponent:
    """Opponent that uses a copy of the training policy (loaded from checkpoint)."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load(model_path)

    def load(self, model_path):
        import torch
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.eval()

    def __call__(self, obs, agent_id):
        if self.model is None:
            return aggressive_opponent(obs, agent_id)
        import torch
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = self.model(obs_t).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)


class RacingEnv(gymnasium.Env):
    """
    Single-agent wrapper around MultiAgentRacingEnv.

    Supports curriculum learning by adjusting opponent count and policies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_agents=2,
        opponent_policy="aggressive",
        extra_config=None,
        render_mode=None,
    ):
        super().__init__()

        config = {
            "num_agents": num_agents,
            "use_render": render_mode == "human",
            "crash_done": False,
            "crash_vehicle_done": False,
            "out_of_road_done": True,
            "allow_respawn": False,
            "horizon": 3000,
            "map_config": {
                "lane_num": max(2, num_agents),
                "exit_length": 20,
                "bottle_lane_num": max(4, num_agents),
                "neck_lane_num": 1,
                "neck_length": 20,
            },
        }
        if extra_config:
            config.update(extra_config)

        self.env = MultiAgentRacingEnv(config)
        self.num_agents = num_agents
        self.ego_id = "agent0"

        if isinstance(opponent_policy, str):
            self._opponent_fn = OPPONENT_POLICIES[opponent_policy]
        else:
            self._opponent_fn = opponent_policy

        # Get spaces
        temp_obs, _ = self.env.reset()
        sample_id = list(temp_obs.keys())[0]
        self.observation_space = self.env.observation_space[sample_id]
        self.action_space = self.env.action_space[sample_id]
        self._last_ego_obs = temp_obs.get(self.ego_id, np.zeros(self.observation_space.shape))
        self._opponent_obs = {k: v for k, v in temp_obs.items() if k != self.ego_id}
        self.env.close()
        self.env = MultiAgentRacingEnv(config)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.config["start_seed"] = seed
        obs_dict, info_dict = self.env.reset()
        self._last_ego_obs = obs_dict.get(
            self.ego_id, np.zeros(self.observation_space.shape)
        )
        self._opponent_obs = {k: v for k, v in obs_dict.items() if k != self.ego_id}
        self._last_route_completion = 0.0
        self._step_count = 0
        ego_info = info_dict.get(self.ego_id, {})
        return self._last_ego_obs.copy(), ego_info

    def step(self, action):
        actions = {}
        for agent_id in list(self.env.agents.keys()):
            if agent_id == self.ego_id:
                actions[agent_id] = action
            else:
                opp_obs = self._opponent_obs.get(agent_id, np.zeros(self.observation_space.shape))
                actions[agent_id] = self._opponent_fn(opp_obs, agent_id)

        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self._opponent_obs = {k: v for k, v in obs.items() if k != self.ego_id}
        self._step_count += 1

        if self.ego_id in obs:
            ego_obs = obs[self.ego_id]
            self._last_ego_obs = ego_obs
        else:
            ego_obs = self._last_ego_obs

        ego_reward = rewards.get(self.ego_id, 0.0)
        ego_terminated = terms.get(self.ego_id, terms.get("__all__", False))
        ego_truncated = truncs.get(self.ego_id, truncs.get("__all__", False))
        ego_info = infos.get(self.ego_id, {})

        # Reward shaping (1-4, 9)
        prev_completion = self._last_route_completion
        curr_completion = ego_info.get("route_completion")
        if curr_completion is None and self.ego_id in self.env.agents:
            curr_completion = self.env.agents[self.ego_id].navigation.route_completion
        curr_completion = curr_completion if curr_completion is not None else 0.0
        self._last_route_completion = curr_completion

        # 1. Dense route completion bonus (stronger signal for learning)
        ego_reward += 20.0 * (curr_completion - prev_completion)
        # 2. Survival bonus
        ego_reward += 0.01
        # 3. Speed bonus (normalize by max ~80 km/h)
        speed = ego_info.get("speed_km_h")
        if speed is None and self.ego_id in self.env.agents:
            speed = getattr(self.env.agents[self.ego_id], "speed_km_h", 0.0)
        speed = speed if speed is not None else 0.0
        ego_reward += 0.04 * (speed / 80.0)
        # 4. Arrival bonus: large fixed bonus + early-finish bonus
        if ego_info.get("arrive_dest", False):
            horizon = self.env.config.get("horizon", 3000)
            early_bonus = 5.0 * (1.0 - self._step_count / horizon)
            arrival_bonus = 50.0
            ego_reward += arrival_bonus + max(0.0, early_bonus)

        # 5. Crash/early-termination penalty (didn't finish, episode ended)
        if (ego_terminated or ego_truncated) and curr_completion < 0.99:
            ego_reward -= 8.0  # Stronger penalty for crash/out-of-road

        ego_info["route_completion"] = curr_completion
        return ego_obs.copy(), float(ego_reward), bool(ego_terminated), bool(ego_truncated), ego_info

    def set_opponent_policy(self, policy):
        """Update the opponent policy (e.g., for self-play curriculum).

        Args:
            policy: Either a string name ("random", "aggressive", "still")
                    or a callable with signature (obs, agent_id) -> action.
        """
        if isinstance(policy, str):
            self._opponent_fn = OPPONENT_POLICIES[policy]
        else:
            self._opponent_fn = policy

    def render(self):
        pass

    def close(self):
        self.env.close()


def make_racing_env(rank=0, num_agents=2, opponent_policy="aggressive", extra_config=None):
    def _init():
        env = RacingEnv(
            num_agents=num_agents,
            opponent_policy=opponent_policy,
            extra_config=extra_config,
        )
        return env
    return _init
