import pandas as pd
import pandas_ta as ta
import os

# Paths for raw and processed data
raw_data_dir = r"C:\Users\BCPLACE\Documents\MEGA\WI Studium\3. Semester\Unternehmenssoftware\AKI-Trading-Bot\data\raw\crypto"
processed_data_dir = r"C:\Users\BCPLACE\Documents\MEGA\WI Studium\3. Semester\Unternehmenssoftware\AKI-Trading-Bot\data\processed"
os.makedirs(processed_data_dir, exist_ok=True)

# List of cryptocurrency symbols to process
crypto_symbols = ["BTC_USDT", "ETH_USDT", "BNB_USDT", "XRP_USDT", "ADA_USDT",
                  "SOL_USDT", "DOGE_USDT", "MATIC_USDT", "DOT_USDT", "LTC_USDT"]


# Function to load data, calculate indicators, and save processed file
def add_indicators(file_path, save_path):
    df = pd.read_csv(file_path)

    # Calculate RSI with a 14-period window
    df['RSI'] = ta.rsi(df['close'], length=14)

    # Calculate MACD (12, 26, 9 are standard periods)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']

    # Calculate Simple Moving Averages (SMA) with 50 and 200 periods
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)

    # Calculate Hull Moving Average (HMA) with a 50-period window
    df['HMA_50'] = ta.hma(df['close'], length=50)

    # Calculate Bollinger Bands with a 20-day window and 2 standard deviations
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['BB_upper'] = bbands['BBU_20_2.0']
    df['BB_middle'] = bbands['BBM_20_2.0']
    df['BB_lower'] = bbands['BBL_20_2.0']

    # Calculate Average True Range (ATR) with a 14-day period
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # Calculate On-Balance Volume (OBV)
    df['OBV'] = ta.obv(df['close'], df['volume'])

    # Calculate Aroon (14-day period) using high and low columns
    aroon = ta.aroon(df['high'], df['low'], length=14)
    df['Aroon_Up'] = aroon['AROONU_14']
    df['Aroon_Down'] = aroon['AROOND_14']
    df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']

    # Save the processed data with indicators
    df.to_csv(save_path, index=False)
    print(f"Processed data saved to {save_path}")


# Process each cryptocurrency data file
for file_name in os.listdir(raw_data_dir):
    # Check if the file name matches one of the expected crypto symbols
    symbol = file_name.split("_data.csv")[0]  # Extract symbol part
    if symbol in crypto_symbols:
        raw_file_path = os.path.join(raw_data_dir, file_name)
        processed_file_path = os.path.join(processed_data_dir, file_name)
        add_indicators(raw_file_path, processed_file_path)
    else:
        print(f"Skipping non-crypto file: {file_name}")
