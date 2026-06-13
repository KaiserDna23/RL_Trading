import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mt5_api import MetaTrader
from metaTrader_env.envs.trading_forex_env_V4 import ForexTradingEnv4
from metaTrader_env.helper.utils_2 import dataset_creation, loadData, soft_squash, dataset_creation_v2
from metaTrader_env.envs.actor_ltc_recurrent_V4 import AgentRLTC_Seq

meta_api = MetaTrader()


def evaluate_model(model: AgentRLTC_Seq, env: ForexTradingEnv4, deterministic: bool = True):
    obs = env.reset()
    equity_curve = []
    while True:
        action = model.choose_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = (terminated or truncated)

        eq = info.get("equity_used", env.equity_used)
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def main(gold=False, is_btc=False, load_data=False, load_model_=False):
    features_map = None

    if is_btc and load_data:
        features_map = loadData("../data/features_BTCUSD_M5_19_15_05.csv")
        print(">: Loaded Gold from csv file ...")
    elif is_btc:
        _, features_map = dataset_creation_v2(trader=meta_api, timeframe="M5", symbol="BTCUSD",
                                           anterior_roll_back=False, support_resistance=True)
        print(">: Loaded BTC directly ...")

    if gold and load_data:
        features_map = loadData("../data/features_v4_XAUUSD+_M5_16_47_36.csv")
        print(">: Loaded Gold from csv file ...")
    elif gold:
        _, features_map = dataset_creation_v2(trader=meta_api, timeframe="M5", symbol="XAUUSD+",
                                           anterior_roll_back=False, support_resistance=True, number_of_candles=86400) #86400 = 1yr data
        print(">: Loaded Gold directly ...")

    # split
    split_index = int(len(features_map) * 0.8)
    train_df = features_map[:split_index].copy()
    # reset index
    train_df.reset_index(inplace=True)

    #test_df = features_map[split_index:-1].copy()  # avoid the last row

    # ---- Env factories
    # ATR Multipliers!
    if gold:
        SL_OPTS = [1.0, 1.5, 2.0]
        TP_OPTS = [1.5, 2.0, 3.0]
        window = 1

    if is_btc:
        SL_OPTS = [1.0, 1.5, 2.0]
        TP_OPTS = [1.5, 2.0, 3.0]
        window = 1

    # Train en: random start to reduce memorization
    def make_train_env():
        return ForexTradingEnv4(
            df=train_df,
            window_size=window,
            anterior_roll_back=False,
            reward_scale=1.0 if gold else 0.008,
            pip_value=0.01,
            sl_atr_multipliers=SL_OPTS,
            tp_atr_multipliers=TP_OPTS,
            spread_pips=7.0,  # as usual with gold
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=True,
            min_episode_steps=12,
            episode_max_steps=288, # day in 5min
            open_penalty_pips=0.5,  # 0.5 half a pip per open
            time_penalty_pips=0.06,  # 0.02 pips per bar in trade
            invalid_action_penalty=0.15, #previous 0.1
            roll_back = 60,
            unrealized_delta_weight=0.4
        )

    # --- Model param---
    env = make_train_env()
    _obs, __ = env.reset()
    input_shape = torch.tensor(_obs).flatten().shape[0]
    print(f">: Input shape {input_shape}...")

    n_actions = len(env.action_map)
    lr = 0.00045  # 3e-4 is standard for PPO

    # Properly batch the agent for PPO stability and lower entropy penalty
    model = AgentRLTC_Seq(n_actions=n_actions, input_dims=input_shape,
                      gamma=0.99, lr=lr, entropy_coef=0.02, policy_clip=0.2, batch_size=64) #128

    if load_model_:
        if is_btc:
            model.load_model()
            print(f">: Loaded BTCUSD previous model...")
        if gold:
            model.load_model()
            print(f">: Loaded XAUUSD+ previous model...")

    print("Model active ...")

    reward_over_episodes = []
    best_score = -np.inf
    N = 64
    epochs = 25000

    for ep in range(epochs):
        done = False
        obs, info = env.reset()
        # flatten observation
        obs = obs.flatten()
        n_steps = 0
        episode_reward = 0

        while not done:
            action, probs, val = model.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            model.remember(obs, action, probs, val, reward, done)
            # LEarn if possible every N, e.g., 10 steps
            if n_steps % N == 0:
                print(f"Last info {info}")
                model.learn()

            # update states
            obs = torch.tensor(obs_).flatten()
            episode_reward += reward
            reward_over_episodes.append(reward)
            avg_score = np.mean(reward_over_episodes[-64:])

            if avg_score > best_score:
                print(f">: ... Saving model...")
                best_score = avg_score
                model.save_model()

            print(f">: Episode {ep} | Actual reward {reward} | Avg reward: {best_score:.3f}")
            print(info)
            # env.render()

        reward_over_episodes.append(episode_reward)

        # display improvement at N episode
        if ep % N == 0:
            plt.plot(reward_over_episodes)
            plt.suptitle(f"Reward / Episode {ep}")
            plt.savefig(fname=f"../images/Figs-{ep}.png")
            plt.close()
            model.plot_error(ep)
    model.save_model("last")

def run_one_episode(model, env):
    obs, info = env.reset()
    obs = obs.flatten()  # reduce dim of input
    equity_curve = []
    closed_trades = []
    count = 0
    event = ""

    while True:
        print(f"Episode Count {count} \n")

        action, probs, val = model.choose_action(obs)
        obs_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity_curve.append(env.get_equity())

        try:
            event = info["last_trade_info"]["event"]
        except Exception as e:
            print(e)
        if event == 'CLOSE' and event != "":
            closed_trades.append(env.last_trade_info)
        if done:
            break
        print(f">: {info}")

        obs = torch.tensor(obs_).flatten()
        count += 1

    return equity_curve, closed_trades


def Rec_env_(gold=False, is_btc=False):
    # ---- Env factories
    # -- Data and features --
    features_map = None
    if gold:
        SL_OPTS = [1.0, 1.5, 2.0]
        TP_OPTS = [1.5, 2.0, 3.0]
        window = 1
        features_map = loadData("../data/features_v4_XAUUSD+_M5_16_47_36.csv")

    if is_btc:
        SL_OPTS = [1.0, 1.5, 2.0]
        TP_OPTS = [1.5, 2.0, 3.0]
        window = 1
        print("Load BTC")
        pass
        #features_map = loadData("../data/features_v4_BTCUSD+_M5_00_05_57.csv")

    # split
    split_index = int(len(features_map) * 0.8)
    test_df = features_map[split_index:-1].copy()
    test_df = test_df[:60]
    test_df = test_df.reset_index().drop("index", axis='columns')  # avoid the last row

    def make_test_eval_env():
        return ForexTradingEnv4(
            df=test_df,
            window_size=window,
            sl_atr_multipliers=SL_OPTS,
            tp_atr_multipliers=TP_OPTS,
            reward_scale=1.0 if gold else 0.008,
            spread_pips=7.0,
            commission_pips=0.0,
            max_slippage_pips=0.2,
            random_start=False,
            episode_max_steps=None,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            invalid_action_penalty=0.0,
            unrealized_delta_weight=0.0,
            roll_back=60,
            initial_equity= 50
        )

    env = make_test_eval_env()
    _obs, __ = env.reset()

    # --- Model Hyperparameters ---
    input_shape = torch.tensor(_obs).flatten().shape[0]  # actual:= 216; previous := 1152 * 21 -> 24192
    n_actions = len(env.action_map)
    lr = 0.001

    # -- Load model --
    model = AgentRLTC_Seq(n_actions=n_actions, input_dims=input_shape, gamma=0.99, lr=lr)
    model.load_model()

    equity_curve, closed_trades = run_one_episode(model, env)

    # save trades
    if closed_trades:
        trade_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history.csv"
        trade_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to {out_csv}")
    else:
        print("No closed trade history")

        # Plot equity
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity (Test)")
    plt.title("Equity Curve - Evaluation")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Todo
#   10/03/26 - (1) Use of the modified Agent code - normalization of advantages
#              (2) Use of modified reward - application of squash function to reward
#              (3) Reduction of learning rate and entropy, from 0.02 to 0.004
#   11/03/26   (1) STUPIDDDD, haven't updated the features list, the model haven't used the SnR features
#              (2) Squash the reward inside the get_features in env_V3


if __name__ == "__main__":
    # main(is_btc=True, load_data=True, load_model_=True)
    #main(gold=True, load_data=True)
    Rec_env_(gold=True)
