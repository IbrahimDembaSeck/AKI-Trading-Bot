import random
import torch
from copy import deepcopy as dc
import yfinance as yf
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from train_model import DualAttentionLSTM
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

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





def load_actual_data(file_path, start_date, end_date):
    """
    Lädt tatsächliche Daten aus einer CSV-Datei und filtert sie für den angegebenen Zeitraum.

    Args:
        file_path: Pfad zur CSV-Datei mit tatsächlichen Daten.
        start_date: Startdatum des Vergleichs.
        end_date: Enddatum des Vergleichs.

    Returns:
        DataFrame mit tatsächlichen Preisen.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df.loc[start_date:end_date]


def calculate_gain_loss(predicted, actual):
    """
    Berechnet Gewinn/Verlust basierend auf vorhergesagten und tatsächlichen Preisen.

    Args:
        predicted: Der vorhergesagte Preis.
        actual: Der tatsächliche Preis.

    Returns:
        Gewinn/Verlust als Prozentwert.
    """
    return ((predicted - actual) / actual) * 100

def calculate_and_plot_baseline(predicted_date, predicted_value, actual_data):
    """
    Berechnet die Baseline (historischer Durchschnitt) und erstellt einen erweiterten Plot.

    Args:
        predicted_date (str): Datum der Vorhersage.
        predicted_value (float): Der vorhergesagte Wert.
        actual_data (pd.DataFrame): DataFrame mit tatsächlichen Preisen (enthält Spalte 'Close').

    Returns:
        float: Historischer Durchschnitt als Baseline.
    """
    # Baseline berechnen (historischer Durchschnitt)
    historical_mean = actual_data['Close'].mean()
    print(f"Baseline (Durchschnitt historischer Preise): {historical_mean:.2f}")

    # Abweichung der Baseline vom tatsächlichen Wert
    actual_value = actual_data.loc[predicted_date, 'Close']
    baseline_deviation = historical_mean - actual_value
    print(f"Baseline-Deviation (Abweichung vom tatsächlichen Wert): {baseline_deviation:.2f}")

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data.index, actual_data['Close'], label='Tatsächlicher Preis', marker='o')
    plt.scatter(pd.to_datetime(predicted_date), predicted_value, color='red', label='Vorhergesagter Preis', zorder=5)
    plt.axvline(pd.to_datetime(predicted_date), color='gray', linestyle='--', label='Vorhersagedatum')
    plt.axhline(historical_mean, color='green', linestyle='--', label='Baseline (historischer Durchschnitt)')
    plt.title("Vergleich: Tatsächlicher vs. vorhergesagter Preis mit Baseline")
    plt.xlabel("Datum")
    plt.ylabel("Preis (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return historical_mean



def plot_comparison(predicted_date, predicted_value, actual_data):
    """
    Erstellt ein Diagramm zum Vergleich von vorhergesagten und tatsächlichen Preisen.

    Args:
        predicted_date: Datum der Vorhersage.
        predicted_value: Der vorhergesagte Preis.
        actual_data: DataFrame mit tatsächlichen Preisen.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual_data.index, actual_data['Close'], label='Tatsächlicher Preis', marker='o')
    plt.scatter(pd.to_datetime(predicted_date), predicted_value, color='red', label='Vorhergesagter Preis', zorder=5)
    plt.axvline(pd.to_datetime(predicted_date), color='gray', linestyle='--', label='Vorhersagedatum')
    plt.title("Vergleich: Tatsächlicher vs. vorhergesagter Preis")
    plt.xlabel("Datum")
    plt.ylabel("Preis (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


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

    # Vor dem Entskalieren ausgeben
    print(f"Vorhergesagter Wert vor dem Entskalieren: {prediction}")

    # Entskalieren
    prediction_rescaled = scaler.inverse_transform([[prediction] + [0] * (data.shape[1] - 1)])[0][0]
    return prediction_rescaled

def calculate_rmse(predicted, actual):
    """
    Berechnet den Root Mean Squared Error (RMSE).

    Args:
        predicted: Vorhergesagte Werte (Einzelwert oder Array).
        actual: Tatsächliche Werte (Einzelwert oder Array).

    Returns:
        RMSE-Wert.
    """
    return np.sqrt(mean_squared_error([actual], [predicted]))

def calculate_rae(predicted, actual, actual_mean):
    """
    Berechnet den Relative Absolute Error (RAE).

    Args:
        predicted: Vorhergesagter Wert.
        actual: Tatsächlicher Wert.
        actual_mean: Mittelwert der tatsächlichen Werte.

    Returns:
        RAE-Wert.
    """
    return abs(predicted - actual) / abs(actual - actual_mean)

btc_actual_data = ".\\BTC-USD_data.csv"

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

    # 2.1 Korrelationen hinzufügen und fehlende Werte behandeln
    combined_data = add_correlations(combined_data, {k: data[k] for k in tickers if k != "BTC"}, window_size=30)

    # 2.2 Entferne irrelevante Korrelationen
    combined_data = filter_low_correlations(combined_data, threshold=0.45)

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

    # 7. Load actual data and calculate gain/loss
    actual_data = load_actual_data(btc_actual_data, PREDICTION_START, PREDICTION_DATE)
    actual_value = actual_data.loc[PREDICTION_DATE, 'Close']
    buy_value = actual_data.loc[PREDICTION_START, 'Close']
    price_deviation = predicted_value - actual_value
    print(f"Vorhergessagter Wert für 02.11.2024 {PREDICTION_DATE}: {predicted_value}")
    print(f"Kaufpreis {PREDICTION_START}: {buy_value}")
    print(f"Tatsächlicher Wert {PREDICTION_DATE}: {actual_value}")
    print(f"Preisunterschied: {price_deviation}")

    # 9. Calculate RMSE and RAE
    actual_mean = actual_data['Close'].mean()
    rmse = calculate_rmse(predicted_value, actual_value)
    rae = calculate_rae(predicted_value, actual_value, actual_mean)

    print(f"RMSE: {rmse:.2f}")
    print(f"RAE: {rae:.2f}")

    # Konvergenzdaten zusammenstellen
    convergence_data = {
        "Metric": ["Predicted Value", "Actual Value", "Price Deviation", "RMSE", "RAE"],
        "Value": [predicted_value, actual_value, price_deviation, rmse, rae]
    }

    # Konvergenztabelle erstellen
    convergence_table = pd.DataFrame(convergence_data)

    # Tabelle anzeigen
    print("\nKonvergenztabelle:")
    print(convergence_table)

    # 8. Plot comparison
    plot_comparison(PREDICTION_DATE, predicted_value, actual_data)

    # 9. Baseline berechnen und Plot erstellen
    historical_mean = calculate_and_plot_baseline(PREDICTION_DATE, predicted_value, actual_data)


if __name__ == "__main__":
    main()
