from __future__ import annotations
import gymnasium as gym
import numpy as np
import pandas as pd

from metaTrader_env.helper.utils import soft_squash


class ForexTradingEnv4(gym.Env):
    """
    Optimized RL trading environment for forex (single position only).

    - Observation: Numpy flattened vector of rolling window of features + 3 state features (position, time in trade, unrealized pips).
    - Actions: HOLD, CLOSE, OPEN(direction, sl, tp).
    - Position persistence: once open, position remains until CLOSE action or SL/TP hit on next bar.
    - Friction: spread + commission + optional slippage applied on BOTH manual closes and SL hits.
    - Reward: realized PnL (pips) minus costs on closes, plus delta-unrealized and time penalties. (No holding bonus).
    - Random episode start to reduce overfitting.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 1,
                 roll_back: int = 60,
                 sl_atr_multipliers=(1.0, 1.5, 2.0),
                 tp_atr_multipliers=(1.5, 2.0, 3.0),
                 pip_value: float = 0.01,
                 spread_pips: float = 6.0,
                 commission_pips: float = 0.0,
                 max_slippage_pips: float = 0.0,
                 leverage: int = 500,
                 lot_size: int = 100,
                 reward_scale: float = 1.0,
                 anterior_roll_back: bool = False,
                 unrealized_delta_weight: float = 0.2,
                 random_start: bool = True,
                 min_episode_steps: int = 12,
                 episode_max_steps: int | None = None,
                 allow_flip: bool = False,
                 open_penalty_pips: float = 0.5,
                 invalid_action_penalty: float = 0.1,  # New: penalty for invalid OPEN
                 time_penalty_pips: float = 0.02,
                initial_equity = 50.0
                 ):
        super().__init__()

        # Data initialization
        self.df = df.copy()
        self.n_steps = len(df)
        self.anterior_roll_back = anterior_roll_back
        self.window_size = int(window_size)
        self.initial_equity_copy = initial_equity
        self.initial_equity = initial_equity

        if self.n_steps <= self.window_size + 2:
            raise ValueError("Dataset too short for given window_size.")

        # Features (must match columns in df after preprocessing)
        self.feature_columns = [
            "year_sin", "year_cos", "month_sin", "month_cos",
            "day_sin", "day_cos", "hour_sin", "hour_cos",
            "minute_sin", "minute_cos", "dist_to_support", "dist_to_resistance",
            "close_pct", f"std_close_{roll_back}", f"mean_close_{roll_back}",
            'atr_12','atr_48','atr_288'
        ] if not self.anterior_roll_back else [
            "year_sin", "year_cos", "month_sin", "month_cos",
            "day_sin", "day_cos", "hour_sin", "hour_cos",
            "minute_sin", "minute_cos", "dist_to_support", "dist_to_resistance",
            "close_pct", f"std_close_{roll_back}", f"mean_close_{roll_back}",
            f"std_close_3", f"mean_close_3",'atr_12','atr_48','atr_288'
        ]

        # Verify all columns exist
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        # OPTIMIZATION: Convert Pandas df to fast numpy array up-front
        self.features_array = self.df[self.feature_columns].values.astype(np.float32)

        self.pip_value = float(pip_value)
        self.usd_per_pip = self.pip_value * int(lot_size)
        self.leverage = int(leverage)

        # Dynamic SL/TP ATR multipliers
        if sl_atr_multipliers is None or tp_atr_multipliers is None:
            raise ValueError("sl_atr_multipliers and tp_atr_multipliers must be provided.")
        self.sl_multipliers = [sl_atr_multipliers] if isinstance(sl_atr_multipliers, (int, float)) else list(
            sl_atr_multipliers)
        self.tp_multipliers = [tp_atr_multipliers] if isinstance(tp_atr_multipliers, (int, float)) else list(
            tp_atr_multipliers)

        # Pre-calculate ATR (Average True Range) strictly for dynamic position sizing (not as an observation feature)
        if "atr_pips" not in self.df.columns:
            high = self.df['high']
            low = self.df['low']
            close_prev = self.df['close'].shift(1)
            tr = np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
            self.df["atr_pips"] = tr.rolling(12).mean() / self.pip_value  # 12 periods = 1h | previously 48 periods = 4 Hours on M5
            self.df["atr_pips"] = self.df["atr_pips"].fillna(20.0)  # 20 pips minimal for xau | previously Fallback to 15 pips if NaNs at start

        # Friction & Rewards
        self.spread_pips = float(spread_pips)
        self.commission_pips = float(commission_pips)
        self.max_slippage_pips = float(max_slippage_pips)

        self.reward_scaling = float(reward_scale)
        self.unrealized_delta_weight = float(unrealized_delta_weight)
        self.open_penalty_pips = float(open_penalty_pips)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.time_penalty_pips = float(time_penalty_pips)

        # Episode config
        self.random_start = bool(random_start)
        self.min_episode_steps = int(min_episode_steps)
        self.episode_max_steps = int(episode_max_steps) if episode_max_steps else None
        self.allow_flip = bool(allow_flip)

        # --- Action space ---
        self.action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
        for direction in [0, 1]:
            for sl_m in self.sl_multipliers:
                for tp_m in self.tp_multipliers:
                    self.action_map.append(("OPEN", direction, float(sl_m), float(tp_m)))
        self.action_space = gym.spaces.Discrete(len(self.action_map))

        # --- Observation space ---
        self.base_num_features = len(self.feature_columns)
        self.state_num_features = 3  # [position, time_in_trade_norm, unrealized_pips_scaled]

        # Flattened observation: (window_size * base_features) + state_features
        self.obs_shape = (self.window_size * self.base_num_features) + self.state_num_features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_shape,),
            dtype=np.float32
        )

        self._reset_state()

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _reset_state(self):
        self.current_step = 0
        self.steps_in_episode = 0
        self.terminated = False
        self.truncated = False

        # Position state
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        # Equity accounting
        self.initial_equity = self.initial_equity_copy
        self.equity = self.initial_equity

        # Logging
        self.equity_curve = []
        self.last_trade_info = None

    def get_equity(self):
        return self.equity
    def _get_state_features(self) -> np.ndarray:
        """Return 3-element state vector for the current step."""
        pos = int(self.position)
        time_norm = float(self.time_in_trade) / 1000.0
        unreal = self._compute_unrealized_pips()
        unreal_scaled = soft_squash(unreal * self.pip_value)
        return np.array([pos, time_norm, unreal_scaled], dtype=np.float32)

    def _compute_unrealized_pips(self) -> float:
        if self.position == 0 or self.entry_price is None:
            return 0.0
        close_price = float(self.df.loc[self.current_step, "close"])
        diff = (close_price - self.entry_price) if self.position == 1 else (self.entry_price - close_price)
        return diff / self.pip_value

    def _sample_slippage_pips(self) -> float:
        """Random slippage between 0 and max_slippage_pips."""
        if self.max_slippage_pips <= 0:
            return 0.0
        return float(np.random.uniform(0.0, self.max_slippage_pips))

    def _round_trip_cost_pips(self) -> float:
        """Total cost (spread + commission) for a round trip, in pips."""
        return self.spread_pips + self.commission_pips

    def _open_position(self, direction: int, sl_mult: float, tp_mult: float):
        close_price = float(self.df.loc[self.current_step, "close"])
        slip_price = self._sample_slippage_pips() * self.pip_value

        # Calculate dynamic SL and TP distances based on current ATR
        current_atr_pips = float(self.df.loc[self.current_step, "atr_pips"])
        sl_pips = current_atr_pips * sl_mult
        tp_pips = current_atr_pips * tp_mult

        if sl_pips <= 20 and tp_pips <= 20:
            # avoid trades that are below gold trading stop loss
            pass
        else:
            if direction == 1:  # long
                entry = close_price + slip_price
                sl_price = entry - (sl_pips * self.pip_value)
                tp_price = entry + (tp_pips * self.pip_value)
                self.position = 1
            else:  # short
                entry = close_price - slip_price
                sl_price = entry + (sl_pips * self.pip_value)
                tp_price = entry - (tp_pips * self.pip_value)
                self.position = -1

            self.entry_price = entry
            self.sl_price = sl_price
            self.tp_price = tp_price
            self.time_in_trade = 0
            self.prev_unrealized_pips = 0.0

            self.last_trade_info = {
                "event": "OPEN",
                "step": self.current_step,
                "direction": direction,
                "entry_price": entry,
                "sl_price": sl_price,
                "tp_price": tp_price,
            }

    def _close_position(self, reason: str, exit_price: float) -> float:
        if self.position == 0:
            return 0.0

        pnl_price = (exit_price - self.entry_price) if self.position == 1 else (self.entry_price - exit_price)
        realized_pips = pnl_price / self.pip_value
        cost_pips = self._round_trip_cost_pips()
        net_pips = realized_pips - cost_pips

        self.equity += net_pips * self.usd_per_pip

        self.last_trade_info = {
            "event": "CLOSE",
            "reason": reason,
            "step": self.current_step,
            "position": self.position,
            "entry_price": self.entry_price,
            "exit_price": exit_price,
            "realized_pips": realized_pips,
            "cost_pips": cost_pips,
            "net_pips": net_pips,
            "equity": self.equity,
            "time_in_trade": self.time_in_trade,
        }

        # Reset position state
        self.position = 0
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.time_in_trade = 0
        self.prev_unrealized_pips = 0.0

        return net_pips

    def _check_sl_tp_intra(self) -> float | None:
        if self.position == 0:
            return None

        # Last bar handling
        if self.current_step >= self.n_steps - 1:
            exit_price = float(self.df.loc[self.current_step, "close"])
            return self._close_position("END_OF_DATA", exit_price)

        next_high = float(self.df.loc[self.current_step + 1, "high"])
        next_low = float(self.df.loc[self.current_step + 1, "low"])

        if self.position == 1:  # long
            sl_hit = next_low <= self.sl_price
            tp_hit = next_high >= self.tp_price
        else:  # short
            sl_hit = next_high >= self.sl_price
            tp_hit = next_low <= self.tp_price

        # OPTIMIZATION: Market order Slippage applied to Stop Loss hit
        if sl_hit and tp_hit:
            exit_px = self.sl_price - (
                        self._sample_slippage_pips() * self.pip_value) if self.position == 1 else self.sl_price + (
                        self._sample_slippage_pips() * self.pip_value)
            return self._close_position("SL_AND_TP_SAME_BAR", exit_px)
        elif sl_hit:
            exit_px = self.sl_price - (
                        self._sample_slippage_pips() * self.pip_value) if self.position == 1 else self.sl_price + (
                        self._sample_slippage_pips() * self.pip_value)
            return self._close_position("SL_HIT", exit_px)
        elif tp_hit:
            return self._close_position("TP_HIT", self.tp_price)  # TP are Limit orders = No Slippage

        return None

    def _get_observation(self) -> np.ndarray:
        """Build the observation flat array."""
        start = max(0, self.current_step - self.window_size)

        # Fast numpy slice instead of pandas loc
        base = self.features_array[start:self.current_step]

        # Ensure correct window padding
        if base.shape[0] < self.window_size:
            pad_rows = self.window_size - base.shape[0]
            pad = np.tile(base[0], (pad_rows, 1)) if base.shape[0] > 0 else np.zeros((pad_rows, self.base_num_features))
            base = np.vstack([pad, base])

        # Flatten the base array
        base_flat = base.flatten()

        # Append state vector ONE time instead of tiling
        state_feat = self._get_state_features()
        obs = np.hstack([base_flat, state_feat]).astype(np.float32)
        return obs

    # ----------------------------------------------------------------------
    # Gym API
    # ----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        if self.random_start:
            max_start = self.n_steps - max(self.min_episode_steps, self.window_size)
            if max_start > self.window_size:
                self.current_step = int(np.random.randint(self.window_size, max_start))
            else:
                self.current_step = self.window_size
        else:
            self.current_step = self.window_size

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        if self.terminated or self.truncated or self.equity < 0:
            obs = self._get_observation()
            return obs, 0.0, True, False, {}

        self.steps_in_episode += 1
        reward_pips = 0.0

        act_type, direction, sl_mult, tp_mult = self.action_map[int(action)]

        # --- 1) Execute action ---
        if act_type == "HOLD":
            pass

        elif act_type == "CLOSE":
            if self.position != 0:
                close_price = float(self.df.loc[self.current_step, "close"])
                slip_price = self._sample_slippage_pips() * self.pip_value
                exit_price = close_price - slip_price if self.position == 1 else close_price + slip_price
                reward_pips += self._close_position("MANUAL_CLOSE", exit_price)

        elif act_type == "OPEN":
            if self.position == 0:
                self._open_position(direction, sl_mult, tp_mult)
                reward_pips -= self.open_penalty_pips
            elif self.allow_flip:
                close_price = float(self.df.loc[self.current_step, "close"])
                exit_price = close_price - (
                            self._sample_slippage_pips() * self.pip_value) if self.position == 1 else close_price + (
                            self._sample_slippage_pips() * self.pip_value)
                reward_pips += self._close_position("FLIP_CLOSE", exit_price)
                self._open_position(direction, sl_mult, tp_mult)
                reward_pips -= self.open_penalty_pips
            else:
                # OPTIMIZATION: Invalid OPEN operation while already holding a trade and not allowed to flip
                reward_pips -= self.invalid_action_penalty

        # --- 2) Check SL/TP on next bar ---
        realized = self._check_sl_tp_intra()
        if realized is not None:
            reward_pips += realized

        # --- 3) Continuous shaping rewards ---
        if self.position != 0:
            self.time_in_trade += 1
            unreal_now = self._compute_unrealized_pips()
            delta_unreal = unreal_now - self.prev_unrealized_pips

            # Delta unrealized drives movement
            if self.unrealized_delta_weight != 0.0:
                reward_pips += self.unrealized_delta_weight * delta_unreal

            # Time penalty (discourage staying too long)
            reward_pips -= self.time_penalty_pips

            self.prev_unrealized_pips = unreal_now

        # --- 4) Advance ---
        self.current_step += 1

        # --- 5) Check termination ---
        if self.current_step >= self.n_steps - 1:
            self.terminated = True
        if self.episode_max_steps and self.steps_in_episode >= self.episode_max_steps:
            self.truncated = True

        self.equity_curve.append(self.equity)

        # --- 6) Observation & Scaled Reward ---
        obs = self._get_observation()
        reward = soft_squash((reward_pips * self.reward_scaling) * self.pip_value)

        # --- 7) Info ---
        info = {
            "equity": self.equity,
            "position": self.position,
            "time_in_trade": self.time_in_trade,
            "reward_pips": reward_pips,
            "Squash_reward": reward,
            "last_trade_info": self.last_trade_info,
            "current_price": float(self.df.loc[self.current_step, "close"]),
        }

        return obs, reward, self.terminated, self.truncated, info

    def render(self):
        print(f"Step={self.current_step} | Equity=${self.equity:.2f} | "
              f"Pos={self.position} | Entry={self.entry_price} | SL={self.sl_price} | TP={self.tp_price}")
