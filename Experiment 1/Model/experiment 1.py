from random import random
import torch
from copy import deepcopy as dc
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Seed als Konstante
SEED = 42

def set_seed(seed=42):
    """Setzt den globalen Seed für Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed gesetzt: {seed}")

def download_data(tickers, start_date, end_date):
    """Lädt Daten von yFinance herunter und entfernt Duplikate."""
    data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[~df.index.duplicated(keep='first')]
        data[name] = df
    return data

def prepare_dataframe_for_lstm(df, n_steps):
    """Bereitet einen DataFrame vor, indem Lookback-Features erstellt werden."""
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

def combine_features(btc_data, market_data, lookback):
    """Kombiniert historische BTC-Daten mit Marktdaten in einem DataFrame."""
    btc_data['Date'] = btc_data.index
    btc_shifted = prepare_dataframe_for_lstm(btc_data[['Date', 'Close']], lookback)

    combined_data = btc_data[['Close']].copy()
    for i in range(1, lookback + 1):
        combined_data[f'BTC_Close(t-{i})'] = btc_shifted[f'Close(t-{i})']

    for name, df in market_data.items():
        combined_data[name] = df['Close']

    combined_data = combined_data.interpolate(method='linear')
    combined_data = combined_data.ffill()
    combined_data.dropna(inplace=True)

    assert not combined_data.isnull().values.any(), "Es gibt noch fehlende Werte im kombinierten DataFrame!"
    print(f"Shape der kombinierten Daten: {combined_data.shape}")
    return combined_data

def scale_data(data, scaler_path):
    """Lädt den gespeicherten Scaler und skaliert die Daten."""
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    scaled_data = scaler.transform(data)
    return scaled_data

def create_sequences(data, sequence_length, num_features):
    """Erstellt Sequenzen und Zielwerte mit korrekter Dimension."""
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data[i: i + sequence_length]  # Form: (sequence_length, num_features)
        label = data[i + sequence_length][0]  # Ziel: Close-Preis
        sequences.append(seq)
        targets.append(label)
    return np.array(sequences), np.array(targets)

def main():
    set_seed(SEED)

    # Zeiträume definieren
    TRAINING_START = "2024-10-02"
    TRAINING_END = "2024-11-01"

    # Scaler-Dateipfad
    SCALER_PATH = "./scaler.pkl"

    tickers = {
        "BTC": "BTC-USD",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Oil": "CL=F",
        "Gas": "NG=F",
        "FedFunds": "^IRX"
    }

    # 1. Daten herunterladen
    data = download_data(tickers, TRAINING_START, TRAINING_END)

    # 2. Features kombinieren
    combined_data = combine_features(data["BTC"], {k: data[k] for k in tickers if k != "BTC"}, lookback=7)

    # 3. Zeitliche Trennung
    training_data = combined_data.loc[TRAINING_START:TRAINING_END]

    # 4. Daten skalieren
    scaled_training_data = scale_data(training_data.values, SCALER_PATH)

    # 5. Sequenzen erstellen
    X_train_full, y_train_full = create_sequences(scaled_training_data, sequence_length=30, num_features=scaled_training_data.shape[1])

    print(f"Shape von X_train_full: {X_train_full.shape}")
    print(f"Shape von y_train_full: {y_train_full.shape}")

if __name__ == "__main__":
    main()
