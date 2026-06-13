import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metaTrader_env.envs.mt5_api import MetaTrader
from metaTrader_env.envs.actor_ltc_recurrent_V4 import AgentRLTC_Seq
from metaTrader_env.self_play.trading_forex_env_self_play import ForexSelfPlayEnv
from metaTrader_env.helper.utils_2 import dataset_creation_v2, loadData

meta_api = MetaTrader()

def main(gold=True, is_btc=False, load_data=False, load_model_=False):
    # Load standardized test features (Ensure these were created CAUSALLY without global leaks!)
    features_map = None
    if gold and load_data:
        features_map = loadData("../data/features_v4_XAUUSD+_M5_00_02_58.csv")
    elif gold:
        _, features_map = dataset_creation_v2(trader=meta_api, timeframe="M5", symbol="XAUUSD+",
                                           anterior_roll_back=False, support_resistance=True, number_of_candles=86400)

    if features_map is None:
        raise ValueError("Could not load features.")


    split_index = int(len(features_map) * 0.8)
    train_df = features_map[:split_index].copy()
    train_df.reset_index(inplace=True)

    # ---- Env setup ----
    SL_OPTS = [1.0, 1.5, 2.0]
    TP_OPTS = [1.5, 2.0, 3.0]

    env = ForexSelfPlayEnv(
        df=train_df,
        window_size=1,
        roll_back=60,
        reward_scale=1.0 if gold else 0.008,
        pip_value=0.01,
        sl_atr_multipliers=SL_OPTS,
        tp_atr_multipliers=TP_OPTS,
        spread_pips=7.0,
        random_start=True,
        episode_max_steps=250,
        open_penalty_pips=0.5,
    )

    _obs_dict, _ = env.reset()
    input_shape = torch.tensor(_obs_dict["challenger"]).flatten().shape[0]
    n_actions = len(env.action_map)
    lr = 0.0003

    # Initialize Two PPO Agents for Classic Self-Play
    print(">: Initializing Challenger Model...")
    challenger = AgentRLTC_Seq(n_actions=n_actions, input_dims=input_shape,
                               gamma=0.99, lr=lr, entropy_coef=0.01, batch_size=32, seq_len=16)

    print(">: Initializing Frozen Champion Model...")
    champion = AgentRLTC_Seq(n_actions=n_actions, input_dims=input_shape,
                             gamma=0.99, lr=0.0, entropy_coef=0.0, batch_size=32, seq_len=16)

    if load_model_:
        if gold:
            champion.load_model()
            # Sync brains immediately so they start on equal footing
            challenger.actor.load_state_dict(champion.actor.state_dict())
            print(f">: Loaded XAUUSD+ previous model...")

    # Sync brains immediately so they start on equal footing
    champion.actor.load_state_dict(challenger.actor.state_dict())
    
    epochs = 5000
    global_steps = 0
    N_horizon = 512 

    # Elo Tracking
    recent_relative_scores = []
    
    for ep in range(epochs):
        done = False
        obs_dict, info = env.reset()
        
        obs_chal = obs_dict["challenger"].flatten()
        obs_champ = obs_dict["champion"].flatten()
        
        ep_diff_pips = 0
        ep_champ_pips = 0.0

        while not done:
            # 1. Champion trades using frozen weights, detached from gradient graph
            with torch.no_grad():
                act_champ, _, _ = champion.choose_action(obs_champ)
            
            # 2. Challenger trades and collects memories
            act_chal, probs, val = challenger.choose_action(obs_chal)
            
            # 3. Multi-Agent Step
            actions = {"challenger": act_chal, "champion": act_champ}
            next_obs_dict, relative_reward, term, trunc, info = env.step(actions)

            # 50% relative score, 50% absolute PnL score!
            #relative_reward = raw_relative_reward + (info['pips']['challenger'] * 0.5)

            done = term or trunc
            
            ep_diff_pips += info["relative_alpha_diff"]
            ep_champ_pips += info["pips"]["challenger"]
            global_steps += 1
            
            # 4. Storage ONLY applies to the Challenger using the Relative Reward
            challenger.remember(obs_chal, act_chal, probs, val, relative_reward, done)
            
            if global_steps % N_horizon == 0:
                print(f"[{global_steps}] Learning on 512 Horizon Block...")
                challenger.learn()

            obs_chal = next_obs_dict["challenger"].flatten()
            obs_champ = next_obs_dict["champion"].flatten()

        recent_relative_scores.append(ep_diff_pips)
        print(f">: Episode {ep} | Net Diff (Chal - Champ): {ep_diff_pips:.1f} pips | Champ PnL: {ep_champ_pips:.1f}")

        # --- The Elo Snapshot Clone Logic ---
        # If the Challenger sustains a statistically consistent victory over 10 episodes,
        # it becomes the new reigning Champion.
        if ep % 10 == 0 and ep > 0:
            avg_diff = np.mean(recent_relative_scores[-10:])
            # Benchmark to beat: Must consistently beat the champion by averaging > 10 pips per episode
            if avg_diff > 10.0:
                print(f"\n >>> NEW CHAMPION ENTHRONED! (Avg Diff: {avg_diff:.1f} pips) <<<")
                champion.actor.load_state_dict(challenger.actor.state_dict())
                challenger.save_model()
                print(f">: Variety of actions of the challenger model {info['actions']['challenger']}")
            else:
                print(f"\n --- Challenger failed to overthrow Champion. (Avg Diff: {avg_diff:.1f} pips) ---")
                print(f">: Variety of actions of the champion model {info['actions']['champion']}")
            # Plot the relative learning curve
            plt.plot(recent_relative_scores)
            plt.suptitle(f"Challenger vs Champion Relative PnL | Ep {ep}")
            plt.savefig(fname=f"../images/SelfPlay_Alpha_{ep}.png")
            plt.close()

            challenger.plot_error(ep)



def run_one_episode(model, env):
    obs, info = env.reset()
    obs = torch.tensor(obs["champion"]).flatten()#obs.flatten()  # reduce dim of input
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

        obs = torch.tensor(obs_["champion"]).flatten()
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
        features_map = loadData("../data/features_v4_XAUUSD+_M5_00_02_58.csv")

    # split
    split_index = int(len(features_map) * 0.8)
    test_df = features_map[split_index:-1].copy()
    test_df = test_df[:60]
    test_df = test_df.reset_index().drop("index", axis='columns')  # avoid the last row

    def make_test_eval_env():
        env = ForexSelfPlayEnv(
            df=test_df,
            window_size=1,
            roll_back=60,
            reward_scale=1.0 if gold else 0.008,
            pip_value=0.01,
            sl_atr_multipliers=SL_OPTS,
            tp_atr_multipliers=TP_OPTS,
            spread_pips=7.0,
            random_start=False,
            episode_max_steps=250,
            open_penalty_pips=0.0,
        )
        return env

    env = make_test_eval_env()
    _obs, __ = env.reset()

    # --- Model Hyperparameters ---
    input_shape = torch.tensor(_obs["champion"]).flatten().shape[0]#torch.tensor(_obs["champion"]).flatten().shape[0]  # actual:= 216; previous := 1152 * 21 -> 24192
    n_actions = len(env.action_map)
    lr = 0.001

    # -- Load model --
    model_ = AgentRLTC_Seq(n_actions=n_actions, input_dims=input_shape, gamma=0.99, lr=lr)
    #model = model_.actor.load_checkpoint()
    model_.load_model()

    equity_curve, closed_trades = run_one_episode(model_, env)

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






if __name__ == "__main__":
    main(gold=True, load_data=True)
    #Rec_env_(gold=True)
