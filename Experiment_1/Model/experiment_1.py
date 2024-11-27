import random
import torch
from copy import deepcopy as dc
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from Experiment_1.Model.train_Model_1 import DualAttentionLSTM
import pickle
import os

MODEL_PATH = "dual_attention_lstm.pth"

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
    """
    Skaliert die Daten mit einem gespeicherten Scaler.

    Args:
        data: Die zu skalierenden Daten.
        scaler_path: Pfad zum gespeicherten Scaler.

    Returns:
        Skaliertes Datenarray.
    """
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

def load_model(model_path, input_size, hidden_size_1, hidden_size_2, sequence_length):
    model = DualAttentionLSTM(
        input_size=input_size,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        sequence_length=sequence_length
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def predict_for_future(model, data, start_date, prediction_date, scaler):
    """
    Create predictions for a specific period and rescale the output.

    Args:
        model: The trained model.
        data: The input data, scaled and prepared.
        start_date: Start date of the prediction.
        prediction_date: Date for which the prediction is made.
        scaler: The scaler used to transform the data.

    Returns:
        The rescaled predicted value.
    """
    start_idx = data.index.get_loc(start_date)
    pred_seq = data.iloc[start_idx - 30 : start_idx].values  # Last 30 days before start date
    pred_seq = torch.tensor(pred_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        prediction = model(pred_seq).item()

    # Entskalieren
    prediction_rescaled = scaler.inverse_transform([[prediction] + [0] * (data.shape[1] - 1)])[0][0]
    return prediction_rescaled


def main():
    set_seed(SEED)

    TRAINING_START = "2010-10-01"
    TRAINING_END = "2024-09-30"  # Training ends before the prediction period
    PREDICTION_START = "2024-10-01"
    PREDICTION_DATE = "2024-11-01"

    SCALER_PATH = "./scaler.pkl"
    tickers = {
        "BTC": "BTC-USD",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Oil": "CL=F",
        "Gas": "NG=F",
        "FedFunds": "^IRX"
    }

    # 1. Download data
    data = download_data(tickers, TRAINING_START, PREDICTION_DATE)

    # 2. Combine features
    combined_data = combine_features(data["BTC"], {k: data[k] for k in tickers if k != "BTC"}, lookback=7)

    # 3. Split into training and prediction sets
    training_data = combined_data.loc[TRAINING_START:TRAINING_END]
    prediction_data = combined_data.loc[PREDICTION_START:PREDICTION_DATE]

    SCALER_PATH = "../data preprocessing/scaler.pkl"

    # 4. Scale data
    scaled_training_data = scale_data(training_data.values, SCALER_PATH)

    # Scaler laden
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)


    # 6. Load model
    model = load_model(
        MODEL_PATH,
        input_size=scaled_training_data.shape[1],
        hidden_size_1=64,
        hidden_size_2=32,
        sequence_length=30
    )

    # 7. Make prediction for 1.11.2024
    predicted_value = predict_for_future(
        model,
        combined_data.loc[:PREDICTION_DATE],
        PREDICTION_START,
        PREDICTION_DATE,
        scaler  # Scaler übergeben
    )
    print(f"Predicted value for {PREDICTION_DATE}: {predicted_value}")


if __name__ == "__main__":
    main()
