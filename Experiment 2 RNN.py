import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Datenvorbereitung

# Laden der Bitcoin-Daten (wöchentliche Daten von 2017 bis 2024)
btc_data = pd.read_csv('../bitcoin_weekly_2017_2024.csv')

# Konvertiere die Zeitstempel in ein Datetime-Format
btc_data['Timestamp'] = pd.to_datetime(btc_data['Timestamp'])

# Nur den "Close"-Preis für das Training verwenden
prices = btc_data['Close'].values.reshape(-1, 1)

# Normalisierung der Daten auf den Bereich [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Funktion zur Erstellung von Sequenzen und Zielwerten
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

# Sequenzlänge definieren (z.B. 10 Wochen)
seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

# Aufteilen in Trainingsdaten (2017-2022)
train_size = len(btc_data[btc_data['Timestamp'] < '2023-01-01'])
X_train, y_train = X[:train_size], y[:train_size]

# In PyTorch Tensoren konvertieren
X_train = torch.from_numpy(X_train).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)

# Erstellen eines TensorDataset und DataLoader mit Batch-Größe 64
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

# 2. Mehrschichtiges RNN-Modell in PyTorch definieren
class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=3):
        super(RNN, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        hidden_cell = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        rnn_out, hidden_cell = self.rnn(input_seq, hidden_cell)
        predictions = self.linear(rnn_out[:, -1])  # Verwende den letzten Zeitschritt
        return predictions

# Modell instanziieren
model = RNN()

# Verlustfunktion und Optimierer definieren
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Lernrate reduziert

# 3. Modelltraining auf den Trainingsdaten (2017-2022)
epochs = 300  # Anzahl der Epochen erhöht
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_function(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# 4. Vorhersage für den Zeitraum 2023 bis 30.09.2024

# Verwende die letzten 10 Wochen aus 2022, um Vorhersagen zu starten
future_sequence = scaled_data[train_size - seq_length:train_size]
future_sequence = torch.from_numpy(future_sequence).type(torch.Tensor).unsqueeze(0)

# Vorhersagen für den Zeitraum von 2023 bis 30.09.2024
predictions = []
num_weeks_to_predict = len(btc_data[btc_data['Timestamp'] >= '2023-01-01'])  # Anzahl der Wochen bis 30.09.2024

for _ in range(num_weeks_to_predict):
    predicted_price = model(future_sequence)
    predictions.append(predicted_price.item())

    # Aktualisiere die Sequenz für die nächste Vorhersage
    predicted_price = predicted_price.view(1, 1, 1)  # In 3D umwandeln
    future_sequence = torch.cat((future_sequence[:, 1:, :], predicted_price), dim=1)

# Rücktransformation der Vorhersagen in den ursprünglichen Bereich
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 5. Vergleich mit den realen Daten (2023 bis 30.09.2024)

# Extrahiere die echten Daten von 2023 bis 30.09.2024
actual_prices = prices[train_size:train_size + num_weeks_to_predict]

# Berechnung der Verifikationsmetriken
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)

# Berechnung der sMAPE (Symmetric Mean Absolute Percentage Error)
def smape(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))

smape_value = smape(actual_prices, predicted_prices)

print(f"Prognose - Mean Squared Error (MSE): {mse}")
print(f"Prognose - Root Mean Squared Error (RMSE): {rmse}")
print(f"Prognose - Mean Absolute Error (MAE): {mae}")
print(f"Prognose - R-squared (R²): {r2}")
print(f"Prognose - Symmetric Mean Absolute Percentage Error (sMAPE): {smape_value:.2f}%")

# 6. Visualisierung der Vorhersagen und echten Daten

# Zeitstempel für 2023 bis 30.09.2024
future_dates = btc_data['Timestamp'][train_size:train_size + num_weeks_to_predict]

# Visualisierung
plt.figure(figsize=(12, 6))
plt.plot(future_dates, actual_prices, color='blue', label='Tatsächlicher Bitcoin Preis')
plt.plot(future_dates, predicted_prices, color='red', label='Vorhergesagter Bitcoin Preis')

plt.title('Bitcoin Preisvorhersagen (2023 - 2024)')
plt.xlabel('Monat')
plt.ylabel('Preis in USD')

# X-Achse anpassen
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # Monatliche Ticks
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Zeige Monat und Jahr

plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
