# A.I trading agent though reinforcement learning

An A.I model capable of trading on the MetaTrader5 (mt5) platform 

### Important notes

- Inside the metaTrader folder there's a requirement file, please install the modules listed. using the ``` pip install requirements.txt ```
- You need to have the MetaTrader software install, this will allow the agent to place and manage it's positions
- You will also need a demo account on any broker or with MetaTrader, place the credential inside `sec/config.env` folder as such
  ```
  user=XXXXXXX
  pwd=XXXXXXX
  server=XXXXXX
  symbols=EURUSD,XAUUSD
  magic_num=007
  ```

## Key files

  | File | Description |
  |------|-------------|
  | `actor_ltc_recurrent_V4` | Model architecture file |
  | `manager_latest` | Testing the model in real condition inside the MetaTrader 5 platform |
  | `agent_training_V4` | Training the agent on the GYM environment |
  | `trading_forex_env_V4` | GYM environment simulating the forex market |
  | `mt5_api` | Allowing communication between the agent and the MetaTrader 5  plateform |


## Usage
A check point is provided and can be used to test directly on mt5
``` python manager_latest.py ```

Or you can train the model using the ``` python agent_training_V4.py```
