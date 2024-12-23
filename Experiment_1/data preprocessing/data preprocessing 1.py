import torch
from copy import deepcopy as dc
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import pickle
import os
import pandas as pd

# Seed als Konstante
SEED = 42

def set_seed(seed=42):
    """Setzt den globalen Seed für Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Für Torch hinzufügen
    print(f"Seed gesetzt: {seed}")

def download_data(tickers, start_date, end_date):
    """Lädt Daten von yFinance herunter und entfernt Duplikate."""
    data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df[~df.index.duplicated(keep='first')]  # Duplikate entfernen
        data[name] = df  # Verwende benutzerdefinierte Namen als Schlüssel
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

    # BTC Lookback-Features
    combined_data = btc_data[['Close']].copy()
    for i in range(1, lookback + 1):
        combined_data[f'BTC_Close(t-{i})'] = btc_shifted[f'Close(t-{i})']

    # Externe Marktindikatoren
    for name, df in market_data.items():
        combined_data[name] = df['Close']

    # Fehlende Werte behandeln
    combined_data = combined_data.interpolate(method='linear')  # Interpolation
    combined_data = combined_data.ffill()  # Vorwärtsauffüllen
    combined_data.dropna(inplace=True)  # Verbleibende NaNs entfernen

    # Debugging: Überprüfen, ob keine fehlenden Werte mehr existieren
    assert not combined_data.isnull().values.any(), "Es gibt noch fehlende Werte im kombinierten DataFrame!"
    print(f"Shape der kombinierten Daten: {combined_data.shape}")
    return combined_data

def scale_data(data, scaler=None):
    """Skaliert die Daten mit MinMaxScaler und gibt den Scaler zurück."""
    if not scaler:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
    else:
        scaled_data = scaler.transform(data)
    return scaled_data, scaler

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

def save_data(data_dict, save_dir):
    """Speichert die vorbereiteten Daten als Pickle-Dateien."""
    for name, data in data_dict.items():
        save_path = os.path.join(save_dir, f"{name}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Gespeichert: {save_path}")

def main():
    set_seed(SEED)  # Seed setzen
    SAVE_DIR = "./"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Zeiträume definieren
    TRAINING_START = "2010-10-01"
    TRAINING_END = "2024-10-01"

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
    scaled_training_data, scaler = scale_data(training_data.values)

    # Angenommen, `combined_data` ist der DataFrame mit den kombinierten Daten
    # Skaliere die Daten
    scaled_combined_data, scaler = scale_data(combined_data.values)

    # Konvertiere die skalierten Daten zurück in einen DataFrame mit den gleichen Spalten
    scaled_combined_df = pd.DataFrame(scaled_combined_data, columns=combined_data.columns, index=combined_data.index)

    # Speichere die kombinierten Daten vor dem Skalieren
    combined_data.to_csv("combined_data_before_scaling.csv")
    print("Kombinierte Daten vor dem Skalieren gespeichert: combined_data_before_scaling.csv")

    # Speichere die kombinierten Daten nach dem Skalieren
    scaled_combined_df.to_csv("combined_data_after_scaling.csv")
    print("Kombinierte Daten nach dem Skalieren gespeichert: combined_data_after_scaling.csv")

    # 5. Sequenzen erstellen
    X_train_full, y_train_full = create_sequences(scaled_training_data, sequence_length=30, num_features=scaled_training_data.shape[1])

    # 6. Training/Validierung Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, shuffle=False, random_state=SEED
    )

    # 7. Reshape der Daten für LSTM
    X_train = X_train.reshape((X_train.shape[0], 30, -1))  # 30 = Sequenzlänge
    X_test = X_test.reshape((X_test.shape[0], 30, -1))

    # y optional reshapen
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Print shape
    print(f"Shape von X_train: {X_train.shape}")
    print(f"Shape von y_train: {y_train.shape}")
    print(f"Shape von X_test: {X_test.shape}")
    print(f"Shape von y_test: {y_test.shape}")

    # 8. Konvertierung in Torch-Tensoren
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # 9. Daten speichern
    save_data({
        "X_train": X_train_tensor,
        "y_train": y_train_tensor,
        "X_test": X_test_tensor,
        "y_test": y_test_tensor,
        "scaler": scaler
    }, SAVE_DIR)

    print(f"Daten erfolgreich vorbereitet und gespeichert im Verzeichnis: {SAVE_DIR}")

if __name__ == "__main__":
    main()
