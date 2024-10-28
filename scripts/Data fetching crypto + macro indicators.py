import ccxt
import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Set up save directory for all data, creating separate folders for crypto and macro data
crypto_save_directory = r"C:\Users\BCPLACE\Documents\MEGA\WI Studium\3. Semester\Unternehmenssoftware\AKI-Trading-Bot\data\raw\crypto"
macro_save_directory = r"C:\Users\BCPLACE\Documents\MEGA\WI Studium\3. Semester\Unternehmenssoftware\AKI-Trading-Bot\data\raw\macro"
os.makedirs(crypto_save_directory, exist_ok=True)
os.makedirs(macro_save_directory, exist_ok=True)

# Binance API keys
api_key = 'G3FE5OP45RXiQ21ma6ZOC2xcchF3iPeVzz7pPxA7opWd1xtSGrPFZSI7l4R70rQa'
api_secret = 'ymP7iZHFN9Q8Zb7mM6OANh3PPEgQzeWJkltMGGzzk61shfYFsW7pwgEUzSzWu5oX'

# Initialize Binance instance
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

# List of top 10 non-stablecoin cryptocurrencies
crypto_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "LTC/USDT"]

# Macro indicators with Yahoo Finance symbols
macro_symbols = {
    "FED_FUNDS_RATE": "^IRX",         # Approximation using 3-month Treasury rate
    "CRUDE_OIL": "CL=F",
    "NATURAL_GAS": "NG=F",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "COPPER": "HG=F",
    "STEEL": "PL=F",                  # Palladium as a proxy for steel
    "LUMBER": "LBS=F"
}

# Function to fetch and save crypto data from Binance
def fetch_and_save_crypto_data(symbol, since_days=365, interval="1h"):  # Set to 365 days for more data
    since_timestamp = binance.parse8601((datetime.now() - timedelta(days=since_days)).isoformat())
    all_data = []
    while since_timestamp < binance.milliseconds():
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe=interval, since=since_timestamp, limit=1000)
            if not ohlcv:
                break
            since_timestamp = ohlcv[-1][0] + 1
            all_data.extend(ohlcv)
        except ccxt.BaseError as e:
            print(f"Error fetching data for {symbol}: {e}")
            break
    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    symbol_filename = symbol.replace("/", "_")
    save_path = os.path.join(crypto_save_directory, f"{symbol_filename}_data.csv")
    df.to_csv(save_path, index=False)
    print(f"Crypto data for {symbol} saved to {save_path}")

# Function to fetch and save macro data from Yahoo Finance
def fetch_and_save_macro_data(ticker, name, period="5y", interval="1d"):  # Retain 5 years for macro indicators
    print(f"Fetching data for {name} ({ticker})...")
    data = yf.download(ticker, period=period, interval=interval)
    if not data.empty:
        save_path = os.path.join(macro_save_directory, f"{name}_data.csv")
        data.to_csv(save_path)
        print(f"{name} data saved to {save_path}")
    else:
        print(f"No data retrieved for {name} ({ticker}).")

# Fetch and save crypto data for each symbol in crypto_symbols
for symbol in crypto_symbols:
    fetch_and_save_crypto_data(symbol)

# Fetch and save macro data for each symbol in macro_symbols
for name, ticker in macro_symbols.items():
    fetch_and_save_macro_data(ticker, name)
