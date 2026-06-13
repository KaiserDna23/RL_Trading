# This class manages a series of bots
# The idea is to effectively is that each bot place a bet and since the model has no means of managing
# multiple positions, the manager will
import math

# from metaTrader_env.envs.actor_ltc_recurrent import AgentRLTC
from metaTrader_env.envs.actor_ltc_recurrent_V4 import AgentRLTC_Seq

import os
import pandas as pd
import shutil
import numpy as np
import time

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')

import talib as ta
from matplotlib import pyplot as plt

from metaTrader_env.envs.mt5_api import MetaTrader
from metaTrader_env.envs.telegram_api import send_message
from metaTrader_env.helper.utils_2 import soft_squash, add_snr_features, encode_cycle_data
from concurrent.futures import ThreadPoolExecutor


class Manager:
    def __init__(self, n_bots: int, meta_api: MetaTrader,
                 model_env_dic: dict,
                 max_draw_down: int = 50, risk_percentage: float = 0.1,
                 ):

        self.ready = False
        self.n_bots = n_bots
        self.max_draw_down = max_draw_down
        self.model_env_dic = model_env_dic
        self.risk_percentage = risk_percentage
        self.config = {}

        self.bots = []
        self.bot_internal_clock = [0 for _ in range(n_bots)]
        self.bot_positions = [0 for _ in range(n_bots)]
        self.meta_trader = meta_api

        self.equity = None
        self.previous_equity = None
        self.warmup_done = None
        self.balance = None
        self.equity_curve = []
        self.open_positions = []

        self.sl_option = self.model_env_dic['sl_opts']
        self.tp_option = self.model_env_dic['tp_opts']

        self.action_map = [("HOLD", None, None, None), ('CLOSE', None, None, None)]
        for direction in [0, 1]:
            for sl in self.sl_option:
                for tp in self.tp_option:
                    self.action_map.append(('OPEN', direction, int(sl), int(tp)))

    # ------------------------------------------------------------------
    # Initialisation and config
    def load_config(self, file_path):
        filepath = os.path.normpath(file_path)
        load_dotenv(filepath)
        self.config["user"] = int(os.getenv("user"))
        self.config["pwd"] = str(os.getenv("pwd"))
        self.config['server'] = str(os.getenv("server"))
        self.config["symbols"] = os.getenv("symbols").split(",")
        self.config["magic_num"] = int(os.getenv("magic_num"))

    def initialize_mql5(self):
        if self.meta_trader.initialize(self.config["user"], self.config["pwd"],
                                       self.config['server'], self.config["magic_num"]):
            self.balance = self.meta_trader.get_balance()
            self.equity = self.meta_trader.get_equity()
            self.previous_equity = self.equity
            self.ready = True
            # self._sync_open_positions()
        else:
            print("Error initializing MetaTrader.")

    def initialize_bots(self):
        # initiliaze n bots and get them ready for action
        for k in range(self.n_bots):
            b = str(self.model_env_dic[f"child_path_"] + f"{k}")
            b = os.path.normpath(b)
            if not os.path.exists(b):
                os.mkdir(b)  #
            # copy the master model to it's children
            shutil.copy(self.model_env_dic["master_path_actor"],
                        b)  # self.model_env_dic[f"child_path_{k}"]  # send to directory
            shutil.copy(self.model_env_dic["master_path_critic"], b)

            self.bots.append(
                AgentRLTC_Seq(n_actions=self.model_env_dic["n_actions"], input_dims=self.model_env_dic["input_dims"],
                          lr=self.model_env_dic["lr"], ckpt_dir=b, load_back_file=False))
            self.bots[-1].load_model()
            # set to evaluation mode
            self.bots[-1].actor.eval()
            # load model from saved one

        print(f">: Manager initialized and {self.n_bots} ready ...")

    # State Features
    def _get_state_features(self, idx: int) -> np.ndarray:
        """Return 3-element state vector for the current step."""
        # Position
        pos = self.bot_positions[idx]  # int(self.position)

        # Time in trade (normalized to avoid large values)
        time_norm = float(self.bot_internal_clock[idx] / 1000.)  # float(self.time_in_trade) / 1000.0

        # Unrealized pips (scaled)
        unreal = (self.meta_trader.get_equity() - self.meta_trader.get_balance())  # self._compute_unrealized_pips()
        # Use soft squash
        unreal_scaled = soft_squash(unreal * self.meta_trader._point) #unreal / 100.  # soft_squash(unreal)#unreal / 100.0

        return np.array([pos, time_norm, unreal_scaled], dtype=np.float32)

    def _close_position_(self, all_positions: bool = True, symbol: str = "XAUUSD+", idx: int = 0) -> bool:
        if all_positions:
            try:
                self.meta_trader.close_all_positions(symbol)
                self.bot_positions = [0 for _ in range(self.n_bots)]
            except Exception as e:
                print(f"Failed to close positions: {e}")
                return False
        else:
            if not self.bot_positions[idx]:
                return False

            try:
                # we close specific position by father of the position
                self.meta_trader.close_position_at(symbol, idx)
                self.bot_positions[idx] = 0
                self.open_positions.pop(idx)
            except Exception as e:
                print(f">: Failed to close positions: {e}")
                return False

        self.balance = self.meta_trader.get_balance()
        self.equity = self.meta_trader.get_equity()
        self.equity_curve.append(self.equity)
        return True

    # ----------------------------------------------------------
    # Order execution with error handling
    def _open_position(self, direction: int, sl_mult: int, tp_mult: int, symbol: str, idx: int = 0, curr_atr:float = None) -> bool:

        result = None
        if self.bot_positions[idx] != 0:
            print(">: Position limit reached. No action taken placed")
            return False

        price = self.meta_trader.get_bid(symbol) if direction == 0 else self.meta_trader.get_ask(
            symbol)  # 0 sell, 1 buy
        order_type = "buy" if direction == 1 else "sell"
        self.position = 1 if order_type == "buy" else -1
        # Can be wrong.
        sl_pips = math.ceil(curr_atr  * sl_mult)*100
        tp_pips = math.ceil(curr_atr * tp_mult)*100

        lot = self.meta_trader.calculate_lots(sl_pips, symbol, self.risk_percentage)
        if lot <= 0:
            print(">: Invalid lot.")
            return False

        try:
            result = self.meta_trader.place_order(order_type=order_type, lot_size=lot, price=price,
                                                  symbol=symbol, sl_point=sl_pips, tp_point=tp_pips)
            pos = self.meta_trader.get_position_details().iloc[-1]
            self.open_positions.append({
                "ticket": pos["ticket"],
                'symbol': symbol,
                "direction": order_type,
                "lot": lot,
                "entry_price": price,
                "sl": sl_pips,
                "tp": tp_pips,
                "bot": idx,
            })
            return True
        except Exception as e:
            print(f">: Failed to place order (1): {e}. Meta trader last error {result}")
            return False

    def _get_data(self, symbol: str = "XAUUSD+", timeframe: str = "M5", limit_bars: int = 240):
        return self.meta_trader.get_bars(symbol, timeframe, limit_bars)  # 12bars of 5M is 1H, so 4H is 48 bars

    def create_features(self, symbol: str, timeframe: str = "M5", limit_bars: int = 4000,
                        roll_back: int = 60, anterior_roll_back=False, support_resistance=True):
        # remember the most recent elements are placed at the end of frame
        data = self._get_data(symbol=symbol, timeframe=timeframe, limit_bars=limit_bars)
        # print(">: inci",data)
        data = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'volume'])
        data["date"] = pd.to_datetime(data["time"], unit="s")
        features = data.copy()

        features = encode_cycle_data(features)

        #if support_resistance:
        #    features = add_snr_features(features, interval=5, lookback=100, atr_period=12)

            # Set volatility features
        features['atr_12'] = ta.NATR(high=features['high'], low=features['low'], close=features['close'], timeperiod=12)
        features['atr_48'] = ta.NATR(high=features['high'], low=features['low'], close=features['close'], timeperiod=48)
        features['atr_288'] = ta.NATR(high=features['high'], low=features['low'], close=features['close'],
                                      timeperiod=288)
        atr_ = ta.ATR(high=features['high'], low=features['low'], close=features['close'], timeperiod=12)
        #
        features["close_pct"] = features["close"].pct_change()
        mean = f"mean_close_{roll_back}"
        std = f"std_close_{roll_back}"

        features[mean] = features["close"].rolling(roll_back).mean()
        features[std] = features["close"].rolling(roll_back).std()
        # standardize
        features["close_pct"] = (features["close_pct"] - features["close_pct"].mean()) / features["close_pct"].std()
        features[mean] = (features[mean] - features[mean].mean()) / features[mean].std()
        features[std] = (features[std] - features[std].mean()) / features[std].std()

        if anterior_roll_back:
            features["mean_close_3"] = features["close"].rolling(3).mean()
            features["std_close_3"] = features["close"].rolling(3).std()
            features["mean_close_3"] = (features["mean_close_3"] - features["mean_close_3"].mean()) / features[
                "mean_close_3"].std()
            features["std_close_3"] = (features["std_close_3"] - features["std_close_3"].mean()) / features[
                "std_close_3"].std()

        # Add the Support and Resistance distance features
        features = features.iloc[-400:]
        features = add_snr_features(features)

        # drop empty row
        features.drop(["time", "open", "high", "low", "close", "tick_volume", "volume", "spread", ], axis=1,
                      inplace=True)
        del data
        # print("END ", features)
        features = features.dropna()

        # to be changed after retraining
        feature_columns = [
            "year_sin", "year_cos", "month_sin", "month_cos",
            "day_sin", "day_cos", "hour_sin", "hour_cos",
            "minute_sin", "minute_cos", "dist_to_support", "dist_to_resistance",
            "close_pct", f"std_close_{roll_back}", f"mean_close_{roll_back}"
        ]
        features = features.loc[:, feature_columns]
        return atr_, features

        # ---------------------------------------------------------------------------
        # Warm-up: feed historical data to agent without trading
        # warm up bars for M1, 4H = 480 (60*60*4), for M5 (12*4) since 1H = 12 * M5

    def warm_up_agent(self, symbol: str, bars: int = 480):
        print(">: Starting war up process across all bots...")
        # _, features = dataset_creation(trader=self.meta_trader, symbol=symbol, timeframe="M1", number_of_candles=bars, roll_back=15)#self.create_features(symbol=symbol, limit_bars=bars)
        _, features = self.create_features(symbol=symbol, timeframe="M5", limit_bars=bars, anterior_roll_back=False)
        # print(features.head())
        for k, bot in enumerate(self.bots):
            for i in range(len(features)):
                obs_features = features.iloc[i]
                # during warm up assume no opened position
                # hard coded features numbers
                state_features = np.zeros(3, dtype=np.float32)  # watch for the error in difference of input_shape
                full_obs = np.hstack([obs_features, state_features]).astype(np.float32)
                full_obs = full_obs.flatten()
                _, _, _ = bot.choose_action(full_obs)
                # self.total_bars += 1

        self.warmup_done = True
        print(">: Finished warm up process")

    def _rest(self):
        self.open_positions = []
        self.total_unrealized = 0
        # self.total_bars = 0
        self.time_in_trade = 0

    def place_a_bet(self, symbol: str, obs_features: pd.Series, warmup: bool = False, done: bool = False, idx: int = 0, atr_=None):
        # Let a bot place their bets using the precalculated features
        state_feat = self._get_state_features(idx=idx)
        full_obs = np.hstack([obs_features, state_feat]).astype(np.float32)
        #with self.bots[idx].eval()
        action, prob, val = self.bots[idx].choose_action(full_obs)
        act_type, direction, sl, tp = self.action_map[action]

        # Can't add another position
        if self.bot_positions[idx] >= 1 and act_type == "OPEN":
            print(">: Position limit reached. Double checked")
            # increase time spent on position
            self.bot_internal_clock[idx] += 1
            return False

        else:
            if not warmup:
                if act_type == "HOLD":
                    pass
                elif act_type == "CLOSE":
                    self._close_position_(all_positions=False, symbol=symbol, idx=idx)
                    self.bot_internal_clock[idx] = 0
                    self.bot_positions[idx] = 0

                elif act_type == "OPEN":
                    self._open_position(direction, sl, tp, symbol, idx, curr_atr=atr_)
                    # increase time spent on position
                    self.bot_internal_clock[idx] += 1
                    self.bot_positions[idx] += 1

                self.equity = self.meta_trader.get_equity()
                self.equity_curve.append(self.equity)

                reward_ = (self.equity - self.previous_equity) / 100.
                self.previous_equity = self.equity
                # bot.remember(full_obs, action, prob, val, reward_, done)
            return True

    def manage_positions(self, symbol: str, trigger_fraction: float = 25, step_points: int = 100, ):
        # essential track each position made by the different boss
        # uses the
        self.meta_trader.trailing_stop_fraction_2(symbol, trigger_fraction, step_points)

    def _evaluate_bots_async(self, symbol: str, bars_processed: int, done: bool):
        """Background routine executed on a ThreadPool to process features without blocking the main event loop."""
        print(">: Thread started: Calculating features for new bar...")
        try:
            # Generate features once per new bar
            atr_, features = self.create_features(symbol=symbol, timeframe="M5", limit_bars=4000, anterior_roll_back=False)
            obs_features = features.iloc[-1]
            curr_atr = atr_.iloc[-1]
            #print(obs_features)

            for i, bot in enumerate(self.bots):
                # Check for empty positions and reset if necessary
                if self.meta_trader.get_position_details().empty:
                    self.bot_positions = [0 for _ in range(self.n_bots)]
                    self.open_positions = []

                self.balance = self.meta_trader.get_balance()
                self.equity = self.meta_trader.get_equity()
                self.equity_curve.append(self.equity)

                # Check Kill Switches
                #if self.equity <= self.balance - 50:
                #    print(">: Agent killed for exceeding money threshold")
                #    send_message("Agent killed for exceeding money threshold. RESTARTING")
                #    self._close_position_(all_positions=True, symbol=symbol)
                #    self._rest()
                #    time.sleep(5)
                #    continue

                if self.balance <= 10:
                    print(">: Agent dead. No more funds")
                    send_message("Agent dead. No more funds.")
                    self._rest()

                # Evaluate individual bot
                try:
                    self.place_a_bet(symbol, obs_features, warmup=False, done=done, idx=i, atr_=curr_atr)
                except Exception as e:
                    print(f">: Failed to place order (2) for bot {i}: {e}")

                print(f">: Bar {bars_processed}: Capital {self.balance}, Equity {self.equity}")
                if self.open_positions:
                    try:
                        print(f">: Past positions: {self.open_positions[-i + 1 % len(self.bots) - 1]}")
                    except IndexError as e:
                        print(f"Unable to access open positions: {e}. We skip.")

                if bars_processed % 10 == 0 and i == 0:
                    print(">: Plotting equity")
                    self._plot_equity(symbol, bars_processed)

        except Exception as e:
            print(f">: Thread error evaluating bots: {e}")

    # -------------------------------------------------------------------------------
    # MAin loop
    def start_play(self, symbol: str = "") -> bool:
        if not self.ready:
            print(">: MT5 not initialized.")
            return

        if not self.warmup_done:
            pass
            # self.warm_up_agent(symbol)

        new_bar = self.meta_trader.NewBarDetector(symbol, self.meta_trader.set_query_timeframe("M5"))
        bars_processed = 0
        done = False
        past_balance = -np.inf
        executor = ThreadPoolExecutor(max_workers=2)

        # first plot
        self._plot_equity(symbol, bars_processed)
        called = False

        while True:
            if True:  # PlayGroundSimulationV2.is_trading_time_allowed():
                # trail each position aggressively, unblocked by features

                #self.manage_positions(symbol, trigger_fraction=20, step_points=200)
                #called = self.meta_trader.kill_switch(symbol, 10, called)  # kill the position at 25% of sl

                # if all position gone, renew the kill switch
                if self.meta_trader.get_position_details().empty:
                    called = True
                    # no positions, reset all param
                    self._rest()

                if new_bar.is_new_bar():
                    if bars_processed % 12 == 0: done = True  # end episode after each hour

                    # Offload the heavily blocking feature execution and place bet logics
                    executor.submit(self._evaluate_bots_async, symbol, bars_processed, done)

                    if self.balance and self.balance > past_balance + (past_balance * 0.10):
                        past_balance = self.balance

                    bars_processed += 1
                else:
                    # print(">: Waiting on a new bar...")
                    pass
            else:
                print(">: Outside trading hours.")

            time.sleep(5)  # Spin very fast so we manage stops aggressively

    def _plot_equity(self, symbol: str, step: int, dir: str = "images"):
        fig, ax = plt.subplots()
        ax.plot(self.equity_curve)
        ax.set_title(f"Equity Curve step {step} for {symbol}")
        p = os.path.normpath(
            fr"C:\Users\assee\Downloads\Trading lab\RL Trading\Gym\metaTrader_env\images\simulation\equity_curve_{step}_for_{symbol}.png")
        fig.savefig(p)
        plt.close(fig)





if __name__ == "__main__":
    bot_env = {"n_actions": 20,
               "input_dims": 18,
               "lr": 0.0001,
               "child_path_": r"C:\Users\assee\Downloads\Trading lab\RL Trading\Gym\metaTrader_env\ppo\manager\child_",
               "master_path_actor": r"C:\Users\assee\Downloads\Trading lab\RL Trading\Gym\metaTrader_env\ppo\manager\actor_torch_ppo_seq.pth",
               "master_path_critic": r"C:\Users\assee\Downloads\Trading lab\RL Trading\Gym\metaTrader_env\ppo\manager\critic_torch_ppo_seq.pth",
               "sl_opts": [1.0, 1.5, 2.0],
               "tp_opts": [1.5, 2.0, 3.0],
    }

    bot_env_btc = {"n_actions": 20,
                   "input_dims": 21,  # 18,
                   "lr": 0.0004,
                   "child_path_": r"C:\Users\assee\Downloads\Trading lab\RL Trading\Gym\metaTrader_env\ppo\manager\child_",
                   "master_path_actor": r"C:\Users\assee\Downloads\Trading lab\RL Trading\Gym\metaTrader_env\ppo\manager\actor_torch_ppo_seq.pth",
                   "master_path_critic": r"C:\Users\assee\Downloads\Trading lab\RL Trading\Gym\metaTrader_env\ppo\manager\critic_torch_ppo_seq.pth",
                   "sl_opts": [1.0, 1.5, 2.0],
                   "tp_opts": [1.5, 2.0, 3.0], }

    # Meta trader aPI
    meta = MetaTrader()

    # play ground
    config_file = r"../sec/config.env"

    # Model hyperparameter
    symbol_ = ["XAUUSD+", "BTCUSD"]

    manager = Manager(n_bots=1, meta_api=meta,
                      model_env_dic=bot_env, risk_percentage=10)

    manager.load_config(config_file)
    manager.initialize_mql5()

    #print(manager.meta_trader.get_balance())
    #manager.meta_trader.set_symbol_info(symbol_[0])
    #print(manager.meta_trader._stops_level)

    manager.initialize_bots()
    manager.start_play(symbol=symbol_[0])