import torch
from copy import deepcopy as dc
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
import pickle
import os

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

def handle_missing_values(df):
    """Behandelt fehlende Werte durch Interpolation und Auffüllen."""
    df = df.interpolate(method='linear', axis=0, limit_direction='both')  # Interpolieren
    df = df.bfill().ffill()  # Backward- und Forward-Fill
    return df


def add_correlations(combined_data, market_data, window_size=30):
    combined_data = dc(combined_data)
    for name, df in market_data.items():
        combined_data[f'Corr_{name}'] = (
            combined_data['Close']
            .rolling(window=window_size, min_periods=1)  # Erlaube kleinere Fenster
            .corr(df['Close'])
        )

    # Entferne komplett leere Spalten und Zeilen
    combined_data = combined_data.dropna(axis=1, how='all')  # Entferne leere Spalten
    combined_data = combined_data.dropna(axis=0, how='all')  # Entferne leere Zeilen

    # Fehlende Werte behandeln (Interpolation und Auffüllen)
    combined_data = combined_data.interpolate(method='linear', axis=0, limit_direction='both')  # Interpolieren
    combined_data = combined_data.bfill().ffill()  # Auffüllen

    return combined_data

def filter_low_correlations(combined_data, threshold=0.2):
    """
    Entfernt Features mit niedriger Korrelation basierend auf einem Schwellenwert.

    Args:
        combined_data (pd.DataFrame): Der kombinierte Datensatz mit Korrelationen als Features.
        threshold (float): Der Schwellenwert für den Absolutwert der Korrelationen.

    Returns:
        pd.DataFrame: Der gefilterte Datensatz ohne irrelevante Korrelationen.
    """
    # Identifiziere Korrelationen-Features
    correlation_columns = [col for col in combined_data.columns if col.startswith('Corr_')]

    # Filtere basierend auf dem Schwellenwert
    for col in correlation_columns:
        if combined_data[col].abs().mean() < threshold:
            combined_data.drop(columns=[col], inplace=True)
            print(f"Feature {col} entfernt: Korrelation unterhalb des Schwellenwerts ({threshold})")

    return combined_data



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

    # 2.1 Korrelationen hinzufügen und fehlende Werte behandeln
    combined_data = add_correlations(combined_data, {k: data[k] for k in tickers if k != "BTC"}, window_size=30)

    # 2.2 Entferne irrelevante Korrelationen
    combined_data = filter_low_correlations(combined_data, threshold=0.45)

    # 3. Zeitliche Trennung
    training_data = combined_data.loc[TRAINING_START:TRAINING_END]

    # 4. Daten skalieren
    scaled_training_data, scaler = scale_data(training_data.values)

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
