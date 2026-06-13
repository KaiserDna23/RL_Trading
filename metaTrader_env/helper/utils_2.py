# All helper functions
import os
from datetime import datetime

# test out the different features group
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import talib as ta
from dotenv import load_dotenv
from mplfinance.original_flavor import candlestick_ohlc
from sklearn.cluster import MeanShift, estimate_bandwidth

from metaTrader_env.envs.mt5_api import MetaTrader

# Improve S n R detection


def find_swing_points(highs: pd.Series, lows: pd.Series, interval: int = 5):
    highs_ = [False for _ in range(len(highs))]
    lows_ = [False for _ in range(len(highs))]
    for i in range(interval, (len(highs) - interval)):
        if highs.iloc[i] == max(highs[i - interval:i + interval + 1]):
            highs_[i] = True

        if lows.iloc[i] == min(lows[i - interval:i + interval + 1]):
            lows_[i] = True
    return highs_, lows_


# get clusters levels
def cluster_levels(points, bandwidth=None):
    """
    Cluster 1D points using MeanShift.

    Parameters:
    - points: array of price values
    - bandwidth: if None, estimate automatically

    Returns:
    - cluster_centers: array of representative price levels
    """
    if len(points) == 0:
        return np.array([])

    # Reshape for sklearn
    X = points.reshape(-1, 1)

    if bandwidth is None:
        # Estimate bandwidth from the data (scaled by a factor)
        bandwidth = estimate_bandwidth(X, quantile=0.2)
        if bandwidth == 0:  # fallback if estimate fails
            bandwidth = np.std(X) * 0.5 + 0.02

    ms = MeanShift(bandwidth=bandwidth)  # , bin_seeding=True
    ms.fit(X)
    return np.sort(ms.cluster_centers_.flatten())


def nearest_levels(price, supports, resistances):
    """
    Get nearest support and resistance levels relative to current price.

    Returns:
    - nearest_support: closest support level below price (or NaN if none)
    - nearest_resistance: closest resistance level above price (or NaN if none)
    """
    # Supports below price
    supports_below = supports[supports < price]
    nearest_support = supports_below.max() if len(supports_below) > 0 else np.nan

    # Resistances above price
    resistances_above = resistances[resistances > price]
    nearest_resistance = resistances_above.min() if len(resistances_above) > 0 else np.nan

    return nearest_support, nearest_resistance


# Apply soft squashing instead of tanh
def soft_squash(x):
    if np.isnan(x):
        return 0.0
    return x / (1 + abs(x))


def nearest_levels_np(price, supports, resistances):
    supports_below = supports[supports < price]
    nearest_support = np.max(supports_below) if len(supports_below) > 0 else np.nan

    resistances_above = resistances[resistances > price]
    nearest_resistance = np.min(resistances_above) if len(resistances_above) > 0 else np.nan
    return nearest_support, nearest_resistance


def add_snr_features(data: pd.DataFrame, interval: int = 5, lookback=100, bandwidth=None, atr_period=12):
    """
     Add the Support and Resistance levels on dataset, heavily optimized with Numpy and cluster caching.
    """
    if data is None or data.empty:
        return None

    n = len(data)
    dist_rest_ = np.full(n, np.nan)
    dist_supp_ = np.full(n, np.nan)

    # Pre-extract fast numpy arrays to avoid Pandas overhead
    high_all = data['high'].values
    low_all = data['low'].values
    close_all = data['close'].values

    # Convert Pandas Series cleanly for TA Lib
    close_s = pd.Series(close_all)
    high_s = pd.Series(high_all)
    low_s = pd.Series(low_all)
    atr_all = ta.ATR(close_s, high_s, low_s, timeperiod=atr_period).values

    # Vectorized caching memory to prevent repeating Sklearn clustering
    last_high_pts = None
    last_low_pts = None
    cached_resist = np.array([])
    cached_support = np.array([])

    for i in range(max(lookback, atr_period), n):
        w_start = i - lookback
        w_high = high_all[w_start:i]
        w_low = low_all[w_start:i]

        highs_idx = []
        lows_idx = []

        # Fast swing point parsing
        for j in range(interval, lookback - interval):
            if w_high[j] == np.max(w_high[j - interval:j + interval + 1]):
                highs_idx.append(j)
            if w_low[j] == np.min(w_low[j - interval:j + interval + 1]):
                lows_idx.append(j)

        high_points = w_high[highs_idx]
        low_points = w_low[lows_idx]

        # MeanShift is O(N^2). We only re-cluster if the extracted points radically changed!
        if not np.array_equal(high_points, last_high_pts):
            cached_resist = cluster_levels(high_points, bandwidth=bandwidth)
            last_high_pts = high_points

        if not np.array_equal(low_points, last_low_pts):
            cached_support = cluster_levels(low_points, bandwidth=bandwidth)
            last_low_pts = low_points

        curr_price = close_all[i]
        nearest_supp, nearest_res = nearest_levels_np(curr_price, cached_support, cached_resist)

        atr_val = atr_all[i - 1]

        if not np.isnan(nearest_supp) and not np.isnan(atr_val) and atr_val != 0:
            dist_supp_[i] = soft_squash((curr_price - nearest_supp) / atr_val)

        if not np.isnan(nearest_res) and not np.isnan(atr_val) and atr_val != 0:
            dist_rest_[i] = soft_squash((nearest_res - curr_price) / atr_val)

    data['dist_to_support'] = dist_supp_
    data['dist_to_resistance'] = dist_rest_
    return data


def encode_cycle_data(df):
    d_ = df.copy()
    # d_['date'] = pd.to_datetime(d_['date'])
    d_["hour"] = d_["date"].dt.hour
    d_["minute"] = d_["date"].dt.minute
    d_["second"] = d_["date"].dt.second
    d_["dayofweek"] = d_["date"].dt.dayofweek
    d_["month"] = d_["date"].dt.month
    d_["year"] = d_["date"].dt.year

    # cyclic encoding
    d_["year_sin"] = np.sin(2 * np.pi * d_["year"] / 365.)
    d_["year_cos"] = np.cos(2 * np.pi * d_["year"] / 365.)

    d_["month_sin"] = np.sin(2 * np.pi * d_["month"] / 12.)
    d_["month_cos"] = np.cos(2 * np.pi * d_["month"] / 12.)

    d_["day_sin"] = np.sin(2 * np.pi * d_["dayofweek"] / 7.)
    d_["day_cos"] = np.cos(2 * np.pi * d_["dayofweek"] / 7.)

    d_["hour_sin"] = np.sin(2 * np.pi * d_["hour"] / 24.)
    d_["hour_cos"] = np.cos(2 * np.pi * d_["hour"] / 24.)

    d_["minute_sin"] = np.sin(2 * np.pi * d_["minute"] / 60.)
    d_["minute_cos"] = np.cos(2 * np.pi * d_["minute"] / 60.)

    d_.drop(columns=["second", "minute", "hour", "dayofweek", "month", "year", "date"], inplace=True)

    return d_


def dataset_creation(trader: MetaTrader, symbol: str = "XAUUSD+", timeframe: str = "M5",
                     number_of_candles: int = 5000, roll_back=15, anterior_roll_back=False,
                     support_resistance=False):  # 4000 bars, i.e., 5000/ 12(1H)/ 24(1D)-> 17 days
    load_dotenv(
        "C:\\Users\\assee\\Downloads\\Trading lab\\RL Trading\\Gym\\metaTrader_env\\sec\\config.env")  # ("metaTrader_env/sec/config.env")
    user = int(os.getenv("user"))
    pwd = str(os.getenv("pwd"))
    server = os.getenv("server")
    symbols = os.getenv("symbols").split(',')
    trader.initialize(user, pwd, server)
    print(">: Checking for symbols desired")
    trader.initialize_symbols(symbols)

    his = trader.queryHistory(symbol, timeframe, number_of_candles)
    his = pd.DataFrame(his)
    his["date"] = pd.to_datetime(his['time'], unit='s')
    his.drop(["real_volume"], axis=1, inplace=True)

    now_time = datetime.now().strftime("%H_%M_%S")

    features = his.copy(True)
    # Encode cycle data to feature map
    features = encode_cycle_data(features)

    # include Support and Resistance zone
    # lockback period number of 100 candles
    # ATR period of 12, i.e., 1H and interval of candles of 5 for swing prices detections
    if support_resistance:
        features = add_snr_features(features, interval=5, lookback=100, atr_period=12)

    features["close_pct"] = features["close"].pct_change()

    # taking the mean every 15mins, by default the timeframe is 5mins, that's 3 bars
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

    # drop empty row
    features.drop(["time", "tick_volume", "spread", ], axis=1, inplace=True)  # "open","high","low","close",
    features.dropna(inplace=True)

    features.to_csv(f"../data/features_{symbol}_{timeframe}_{now_time}.csv", columns=features.columns, index=False)
    return his, features

def dataset_creation_v2(trader: MetaTrader, symbol: str = "XAUUSD+", timeframe: str = "M5",
                     number_of_candles: int = 5000, roll_back=60, anterior_roll_back=False,
                     support_resistance=False): #4000 bars, i.e., 5000/ 12(1H)/ 24(1D)-> 17 days
    load_dotenv("C:\\Users\\assee\\Downloads\\Trading lab\\RL Trading\\Gym\\metaTrader_env\\sec\\config.env")#("metaTrader_env/sec/config.env")
    user = int(os.getenv("user"))
    pwd = str(os.getenv("pwd"))
    server = os.getenv("server")
    symbols = os.getenv("symbols").split(',')
    trader.initialize(user, pwd, server)
    print(">: Checking for symbols desired")
    trader.initialize_symbols(symbols)

    his = trader.queryHistory(symbol, timeframe, number_of_candles)
    his = pd.DataFrame(his)
    his["date"] = pd.to_datetime(his['time'], unit='s')
    his.drop(["real_volume"], axis=1, inplace=True)

    now_time = datetime.now().strftime("%H_%M_%S")

    features = his.copy(True)
    # Encode cycle data to feature map
    features = encode_cycle_data(features)

    # include Support and Resistance zone
    # lockback period number of 100 candles
    # ATR period of 12, i.e., 1H and interval of candles of 5 for swing prices detections
    if support_resistance:
        features = add_snr_features(features, interval=5, lookback=100, atr_period=12)

    # Set volatility features
    features['atr_12'] = ta.NATR(high=features['high'], low=features['low'], close=features['close'], timeperiod=12)
    features['atr_48'] = ta.NATR(high=features['high'], low=features['low'], close=features['close'], timeperiod=48)
    features['atr_288'] = ta.NATR(high=features['high'], low=features['low'], close=features['close'], timeperiod=288)

    features["close_pct"] = np.nan_to_num(
        (features["close"].pct_change() - features["close"].pct_change().rolling(roll_back).mean()) / features[
            "close"].pct_change().rolling(roll_back).std())

    # CRITICAL FIX: The user's external preprocessing standardized `mean_close` over the entire dataset,
    # injecting future data leaks. We must calculate perfectly causal stationary distances natively.
    causal_mean = features["close"].rolling(roll_back).mean()
    causal_std = features["close"].rolling(roll_back).std() + 1e-8

    # 'Z-score' relative strictly to the past N bars (perfectly causal stationary feature)
    features[f"mean_close_{roll_back}"] = (features["close"] - causal_mean) / causal_std
    # f = d[f"mean_close_{roll_back}"].values
    #features[f"mean_close_{roll_back}"] = [soft_squash(x) for x in features[f"mean_close_{roll_back}"].values]
    features[f"mean_close_{roll_back}"] = np.vectorize(soft_squash)(np.nan_to_num(features[f"mean_close_{roll_back}"].to_numpy(), nan=0.0))
    # Pure percentage volatility metric
    features[f"std_close_{roll_back}"] = (causal_std / causal_mean) * 100 # remove the tiny factor

    # drop empty row
    features.drop(["time","tick_volume", "spread",], axis=1, inplace=True) #"open","high","low","close",
    features.dropna(inplace=True)

    features.to_csv(f"../data/features_v4_{symbol}_{timeframe}_{now_time}.csv", columns=features.columns, index=False)
    return his, features

def loadData(path_file):
    data = pd.read_csv(path_file, )
    return data

def graph_data(data:pd.DataFrame):
    """
    Draw candle data base on dataframe provided
    :param data:
    :return:
    """
    candlestick_ohlc(data["open"], data["high"], data["low"], data["close"])