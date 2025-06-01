import time
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, KeltnerChannel, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.others import DonchianChannel
from telegram import Bot

# ====== CONFIGURATION =======
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
TELEGRAM_BOT_TOKEN = "8030718150:AAFp5QuwaC-103ruvB5TsBMGY5MwMvkq-5g"
TELEGRAM_CHAT_ID = "@iopuygy"

FOREX_PAIRS = ["USDJPY", "AUDCAD", "EURUSD", "USDCAD", "GBPUSD"]
INTERVAL_SECONDS = 60

# Threshold for sending message (combined confidence)
CONFIDENCE_THRESHOLD = 0.7

# ======= INIT BOT =======
bot = Bot(token=TELEGRAM_BOT_TOKEN)


def send_message(text):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
        print(f"[Telegram] Sent message")
    except Exception as e:
        print(f"[Telegram Error] {e}")


def fetch_historical_data(pair, limit=100):
    """
    Fetch historical 1-minute candle data for the last 'limit' candles from Polygon.io
    """
    ticker = f"C:{pair}"
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/now-100minute/now?adjusted=true&sort=asc&limit={limit}&apiKey={POLYGON_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "results" not in data:
            print(f"[Data Error] No results for {pair}")
            return None

        df = pd.DataFrame(data["results"])
        # Rename columns for convenience
        df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        return df

    except Exception as e:
        print(f"[Fetch Error] {pair}: {e}")
        return None


def calculate_indicators(df):
    """
    Calculate 15+ technical indicators and add as columns to df
    """

    # Momentum Indicators
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['stoch_k'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14).stoch()
    df['stoch_d'] = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14).stoch_signal()

    # Trend Indicators
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()

    # Volatility Indicators
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    kc = KeltnerChannel(high=df['high'], low=df['low'], close=df['close'], window=20)
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_low'] = kc.keltner_channel_lband()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

    # Volume Indicators
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()

    # Other Indicators
    donchian = DonchianChannel(high=df['high'], low=df['low'], window=20)
    df['donchian_high'] = donchian.donchian_channel_hband()
    df['donchian_low'] = donchian.donchian_channel_lband()

    return df


def compute_confidence(df):
    """
    Combine indicators for a confidence score and direction:
    Returns (direction, confidence) tuple where direction = "UP" or "DOWN"
    """

    latest = df.iloc[-1]

    score = 0
    weight_sum = 0

    # Rule-based weighted scoring of indicators:

    # 1. RSI: below 30 oversold (buy), above 70 overbought (sell)
    if latest.rsi < 30:
        score += 1 * 1.5
    elif latest.rsi > 70:
        score -= 1 * 1.5
    weight_sum += 1.5

    # 2. MACD crossover
    if latest.macd > latest.macd_signal:
        score += 1 * 2
    else:
        score -= 1 * 2
    weight_sum += 2

    # 3. ADX: trend strength > 25 is strong trend, direction by MACD above
    if latest.adx > 25:
        score += 0.5 * 1
    weight_sum += 1

    # 4. CCI: below -100 oversold, above 100 overbought
    if latest.cci < -100:
        score += 1 * 1
    elif latest.cci > 100:
        score -= 1 * 1
    weight_sum += 1

    # 5. Stochastic %K and %D: buy signal if %K crosses above %D below 20; sell if opposite above 80
    if latest.stoch_k > latest.stoch_d and latest.stoch_k < 20:
        score += 1 * 1.5
    elif latest.stoch_k < latest.stoch_d and latest.stoch_k > 80:
        score -= 1 * 1.5
    weight_sum += 1.5

    # 6. Bollinger Bands: price crossing lower band is buy; crossing upper band is sell
    if latest.close < latest.bb_low:
        score += 1 * 1.5
    elif latest.close > latest.bb_high:
        score -= 1 * 1.5
    weight_sum += 1.5

    # 7. Keltner Channel: price below lower band buy; above upper band sell
    if latest.close < latest.kc_low:
        score += 1 * 1.2
    elif latest.close > latest.kc_high:
        score -= 1 * 1.2
    weight_sum += 1.2

    # 8. ATR: higher ATR means volatility, consider neutral (0)
    weight_sum += 0

    # 9. OBV: rising OBV bullish, falling bearish
    if df['obv'].iloc[-1] > df['obv'].iloc[-2]:
        score += 1 * 1
    else:
        score -= 1 * 1
    weight_sum += 1

    # 10. MFI: below 20 oversold buy, above 80 overbought sell
    if latest.mfi < 20:
        score += 1 * 1
    elif latest.mfi > 80:
        score -= 1 * 1
    weight_sum += 1

    # 11. Donchian Channel: price below lower band buy; above upper band sell
    if latest.close < latest.donchian_low:
        score += 1 * 1
    elif latest.close > latest.donchian_high:
        score -= 1 * 1
    weight_sum += 1

    # Normalize confidence between 0 and 1
    confidence = min(max((score / weight_sum + 1) / 2, 0), 1)

    direction = "UP" if confidence >= 0.5 else "DOWN"

    return direction, confidence


def main():
    print("🚀 Advanced Forex Telegram Bot Started")
    while True
