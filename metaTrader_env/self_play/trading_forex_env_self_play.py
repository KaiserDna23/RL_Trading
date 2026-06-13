from __future__ import annotations
import gymnasium as gym
import numpy as np
import pandas as pd

from metaTrader_env.helper.utils import soft_squash

class ForexSelfPlayEnv(gym.Env):
    """
    Self-Play Two-Agent reinforcement learning environment for forex.
    - Two agents ("challenger", "champion") compete simultaneously on the exact same historical timeline.
    - Both receive the exact same observation state.
    - `step()` processes an action dictionary: {"challenger": act1, "champion": act2}
    - The Challenger's reward is its PnL strictly subtracted by the Champion's PnL, naturally creating a Relative Alpha baseline.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 1,
                 roll_back: int = 60,
                 sl_atr_multipliers=(1.0, 1.5, 2.0),
                 tp_atr_multipliers=(1.5, 2.0, 3.0),
                 pip_value: float = 0.01,
                 spread_pips: float = 7.0,
                 commission_pips: float = 0.0,
                 max_slippage_pips: float = 0.2,
                 leverage: int = 500,
                 lot_size: int = 100,
                 reward_scale: float = 1.0,
                 random_start: bool = True,
                 episode_max_steps: int | None = 250,
                 open_penalty_pips: float = 0.5,
                 invalid_action_penalty: float = 0.1,
                 time_penalty_pips: float = 0.02,
                 initial_bidget:int = 50,
                 ):
        super().__init__()

        self.df = df.copy()
        self.n_steps = len(df)
        self.window_size = int(window_size)
        self.roll_back = int(roll_back)
        self.initial_budget = int(initial_bidget)

        if self.n_steps <= self.window_size + 2:
            raise ValueError("Dataset too short.")

        self.feature_columns = [
            "year_sin", "year_cos", "month_sin", "month_cos",
            "day_sin", "day_cos", "hour_sin", "hour_cos",
            "minute_sin", "minute_cos", "dist_to_support", "dist_to_resistance",
            "close_pct", f"std_close_{roll_back}", f"mean_close_{roll_back}", "atr_12",
            "atr_48", "atr_288"
        ]

        # OPTIMIZATION & LEAK FIX: Calculate causal stationary targets internally
        causal_mean = self.df["close"].rolling(self.roll_back).mean()
        causal_std = self.df["close"].rolling(self.roll_back).std() + 1e-8
        self.df[f"mean_close_{self.roll_back}"] = (self.df["close"] - causal_mean) / causal_std
        self.df[f"std_close_{self.roll_back}"] = causal_std / causal_mean

        self.features_array = self.df[self.feature_columns].values.astype(np.float32)
        self.features_array = np.nan_to_num(self.features_array, nan=0.0)

        self.pip_value = float(pip_value)
        self.usd_per_pip = self.pip_value * int(lot_size)
        self.leverage = int(leverage)

        self.sl_multipliers = list(sl_atr_multipliers) if isinstance(sl_atr_multipliers, (tuple, list)) else [sl_atr_multipliers]
        self.tp_multipliers = list(tp_atr_multipliers) if isinstance(tp_atr_multipliers, (tuple, list)) else [tp_atr_multipliers]

        if "atr_pips" not in self.df.columns:
            high, low, close_prev = self.df['high'], self.df['low'], self.df['close'].shift(1)
            tr = np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
            self.df["atr_pips"] = tr.rolling(48).mean() / self.pip_value 
            self.df["atr_pips"] = self.df["atr_pips"].fillna(15.0)

        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)
        
        self.reward_scaling = float(reward_scale)
        self.open_penalty_pips = float(open_penalty_pips)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.time_penalty_pips = float(time_penalty_pips)

        self.random_start = bool(random_start)
        self.episode_max_steps = episode_max_steps

        self.diversity_of_actions = {"champion": {"Long": 0, "Short": 0, "Hold": 0, "Close": 0},
                                     "challenger": {"Long": 0, "Short": 0, "Hold": 0, "Close": 0}}

        # --- Action space ---
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:
            for sl_m in self.sl_multipliers:
                for tp_m in self.tp_multipliers:
                    self.action_map.append(("OPEN", direction, float(sl_m), float(tp_m)))
        
        # Dual Actions Dictionary via Gym (conceptually)
        self.action_space = gym.spaces.Discrete(len(self.action_map))

        # --- Observation space ---
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3   # For each agent, though functionally they observe their own state appended to global metrics
        
        self.obs_shape = (self.window_size * self.base_num_features) + self.state_num_features
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

        self.agents = ["challenger", "champion"]

        self.equity = {agent: self.initial_budget for agent in self.agents}

        self._reset_state()

    def get_equity(self):
        return self.equity

    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Independent Agent tracking dicts
        self.pos = {agent: 0 for agent in self.agents}
        self.entry = {agent: None for agent in self.agents}
        self.sl = {agent: None for agent in self.agents}
        self.tp = {agent: None for agent in self.agents}
        self.time_in = {agent: 0 for agent in self.agents}
        self.equity = {agent: 50.0 for agent in self.agents}

        self.diversity_of_actions = {"champion": {"Long": 0, "Short": 0, "Hold": 0, "Close": 0},
                                     "challenger": {"Long": 0, "Short": 0, "Hold": 0, "Close": 0}}
        
        # We don't trace continuous unrealized shaping heavily for relative alpha, 
        # because the true signal is strictly realized PnL differentials.

    def _get_observation(self, agent: str) -> np.ndarray:
        start = max(0, self.current_step - self.window_size)
        base = self.features_array[start:self.current_step]

        if base.shape[0] < self.window_size:
            pad_rows = self.window_size - base.shape[0]
            pad = np.tile(base[0], (pad_rows, 1)) if base.shape[0] > 0 else np.zeros((pad_rows, self.base_num_features))
            base = np.vstack([pad, base])

        base_flat = base.flatten()
        
        # Each agent observes its own internal position state cleanly paired with the global chart
        pos = float(self.pos[agent])
        time_norm = float(self.time_in[agent]) / 1000.0
        
        # compute unrealized selectively
        unreal_pips = 0.0
        if self.pos[agent] != 0 and self.entry[agent] is not None:
            close_price = float(self.df.loc[self.current_step, "close"])
            diff = (close_price - self.entry[agent]) if self.pos[agent] == 1 else (self.entry[agent] - close_price)
            unreal_pips = diff / self.pip_value

        state_feat = np.array([pos, time_norm, soft_squash(unreal_pips * self.pip_value)], dtype=np.float32)
        return np.hstack([base_flat, state_feat]).astype(np.float32)

    def _sample_slippage(self) -> float:
        if self.max_slippage_pips <= 0: return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _execute_open(self, agent: str, direction: int, sl_mult: float, tp_mult: float):
        close_price = float(self.df.loc[self.current_step, "close"])
        slip_price = self._sample_slippage() * self.pip_value
        
        current_atr_pips = float(self.df.loc[self.current_step, "atr_pips"])
        sl_pips = current_atr_pips * sl_mult
        tp_pips = current_atr_pips * tp_mult

        if direction == 1:   # long
            entry = close_price + slip_price
            self.sl[agent] = entry - (sl_pips * self.pip_value)
            self.tp[agent] = entry + (tp_pips * self.pip_value)
            self.diversity_of_actions[agent]["Long"] += 1
        else:                 # short
            entry = close_price - slip_price
            self.sl[agent] = entry + (sl_pips * self.pip_value)
            self.tp[agent] = entry - (tp_pips * self.pip_value)
            self.diversity_of_actions[agent]["Short"] += 1

        self.pos[agent] = 1 if direction == 1 else -1
        self.entry[agent] = entry
        self.time_in[agent] = 0

    def _execute_close(self, agent: str, exit_price: float) -> float:
        if self.pos[agent] == 0: return 0.0

        pnl_price = (exit_price - self.entry[agent]) if self.pos[agent] == 1 else (self.entry[agent] - exit_price)
        realized_pips = pnl_price / self.pip_value
        net_pips = realized_pips - (self.spread_pips + self.commission_pips)

        self.equity[agent] += net_pips * self.usd_per_pip

        self.pos[agent] = 0
        self.entry[agent] = None
        self.sl[agent] = None
        self.tp[agent] = None
        self.time_in[agent] = 0

        return net_pips

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        self.diversity_of_actions = {"champion":{"Long":0, "Short":0, "Hold":0, "Close":0},
                            "challenger":{"Long":0, "Short":0, "Hold":0, "Close":0}}

        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps if hasattr(self, 'min_episode_steps') else 12, self.window_size)
            self.current_step = int(np.random.randint(self.window_size, max_start)) if max_start > self.window_size else self.window_size
        else:
            self.current_step = self.window_size

        return {"challenger": self._get_observation("challenger"), "champion": self._get_observation("champion")}, {}

    def step(self, actions: dict):
        """ Expects dict format: {"challenger": int_action, "champion": int_action} """
        self.steps_in_episode += 1
        
        rewards_raw = {agent: 0.0 for agent in self.agents}

        # 1. Process Actions for each agent independently
        for i, agent in enumerate(self.agents):
            if self.equity[agent] < 0:
                continue # Agent is bankrupt, ignores actions
                
            action = actions[agent]
            act_type, direction, sl_m, tp_m = self.action_map[int(action)]

            if act_type == "HOLD":
                self.diversity_of_actions[agent]["Hold"] += 1
                continue

            if act_type == "CLOSE" and self.pos[agent] != 0:
                cp = float(self.df.loc[self.current_step, "close"])
                slip = self._sample_slippage() * self.pip_value
                ep = cp - slip if self.pos[agent] == 1 else cp + slip
                rewards_raw[agent] += self._execute_close(agent, ep)
                self.diversity_of_actions[agent]["Close"] += 1

            elif act_type == "OPEN":
                if self.pos[agent] == 0:
                    self._execute_open(agent, direction, sl_m, tp_m)
                    rewards_raw[agent] -= self.open_penalty_pips

                else:
                    rewards_raw[agent] -= self.invalid_action_penalty


        # 2. Process intra-bar SL/TP dynamically for both 
        if self.current_step < self.n_steps - 1:
            next_h = float(self.df.loc[self.current_step + 1, "high"])
            next_l = float(self.df.loc[self.current_step + 1, "low"])
            
            for agent in self.agents:
                if self.pos[agent] == 0: continue
                
                if self.pos[agent] == 1:
                    sl_hit, tp_hit = next_l <= self.sl[agent], next_h >= self.tp[agent]
                else:
                    sl_hit, tp_hit = next_h >= self.sl[agent], next_l <= self.tp[agent]
                    
                if sl_hit and tp_hit:
                    ep = self.sl[agent] - (self._sample_slippage() * self.pip_value) if self.pos[agent] == 1 else self.sl[agent] + (self._sample_slippage() * self.pip_value)
                    rewards_raw[agent] += self._execute_close(agent, ep)
                elif sl_hit:
                    ep = self.sl[agent] - (self._sample_slippage() * self.pip_value) if self.pos[agent] == 1 else self.sl[agent] + (self._sample_slippage() * self.pip_value)
                    rewards_raw[agent] += self._execute_close(agent, ep)
                elif tp_hit:
                    rewards_raw[agent] += self._execute_close(agent, self.tp[agent]) # Limit un-slipped
                    
                # Time decay
                if self.pos[agent] != 0:
                    self.time_in[agent] += 1
                    rewards_raw[agent] -= self.time_penalty_pips

        # 3. Advance and Check Termination
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            self.terminated = True
        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        # 4. Construct Relative Alpha Reward specifically for Challenger 
        # The Challenger mathematically ONLY gets positive gradient momentum if it out-traded the Champion!
        diff_pips = rewards_raw["challenger"] - rewards_raw["champion"]
        relative_baseline = soft_squash((diff_pips * self.reward_scaling) * self.pip_value)
        absolute_baseline = soft_squash((rewards_raw["challenger"] - self.reward_scaling) * self.pip_value)

        # 50% score for beating champion and 50% score for pure absolute PnL
        relative_reward = (relative_baseline * 0.5) + (absolute_baseline * 0.5)

        obs = {agent: self._get_observation(agent) for agent in self.agents}
        info = {"equity": self.equity, "relative_alpha_diff": diff_pips, "pips": rewards_raw,
                "actions":self.diversity_of_actions, "raw_reward":(rewards_raw["challenger"]*self.reward_scaling)*self.pip_value,}

        return obs, relative_reward, self.terminated, self.truncated, info

