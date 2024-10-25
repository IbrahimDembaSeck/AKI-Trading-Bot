import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import ccxt
import requests

# 1. Fetching Data (Bitcoin, Fed Funds Rate)
def get_ohlcv_interval(symbol='BTC/USDT', timeframe='1w', start_date=None, end_date=None):
    all_data = []
    since = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    while since < end_timestamp:
        try:
            ohlcv = binance.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not ohlcv:
                break
            all_data += ohlcv
            since = ohlcv[-1][0] + 60 * 60 * 24 * 7
        except Exception as e:
            print(f"Error fetching OHLCV data: {e}")
            break
    df = pd.DataFrame(all_data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df[df['Timestamp'] <= end_date]
    return df

# Binance API setup
binance = ccxt.binance({
    'apiKey': 'G3FE5OP45RXiQ21ma6ZOC2xcchF3iPeVzz7pPxA7opWd1xtSGrPFZSI7l4R70rQa',
    'secret': 'ymP7iZHFN9Q8Zb7mM6OANh3PPEgQzeWJkltMGGzzk61shfYFsW7pwgEUzSzWu5oX',
    'enableRateLimit': True
})

start_date = datetime(2017, 1, 1)
end_date = datetime(2024, 9, 30)
btc_weekly_data = get_ohlcv_interval(symbol='BTC/USDT', start_date=start_date, end_date=end_date)

# Fetching Fed Funds Rate
fedfunds_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS'
fedfunds_data = pd.read_csv(fedfunds_url)
fedfunds_data['DATE'] = pd.to_datetime(fedfunds_data['DATE'])
fedfunds_data = fedfunds_data[(fedfunds_data['DATE'] >= '2017-01-01') & (fedfunds_data['DATE'] <= '2024-09-30')]
fedfunds_data.rename(columns={'FEDFUNDS': 'Interest_Rate'}, inplace=True)

# Merging datasets
btc_data = pd.merge_asof(btc_weekly_data.sort_values('Timestamp'),
                         fedfunds_data.sort_values('DATE'),
                         left_on='Timestamp', right_on='DATE')
btc_data.ffill(inplace=True)
btc_data.drop(columns=['DATE'], inplace=True)

# 2. Data Preprocessing: Normalization and feature selection
features = btc_data[['Close', 'Open', 'High', 'Low', 'Volume', 'Interest_Rate']].values

sc = MinMaxScaler(feature_range=(0, 1))
scaled_data = sc.fit_transform(features)

def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length, 0])  # Predicting on Close price
    return np.array(sequences), np.array(targets)

seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# Train-test split
splitlimit = int(len(X) * 0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

# Convert to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# 3. LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_layer_size=150, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training the Model
epochs = 30
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# 5. Prediction
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test).squeeze()

# Inverse transform the predictions back to the original scale
y_test_rescaled = sc.inverse_transform(np.concatenate([y_test.numpy().reshape(-1, 1), np.zeros((y_test.shape[0], features.shape[1] - 1))], axis=1))[:, 0]
y_pred_rescaled = sc.inverse_transform(np.concatenate([y_pred_test.numpy().reshape(-1, 1), np.zeros((y_pred_test.shape[0], features.shape[1] - 1))], axis=1))[:, 0]

# 6. Visualization
plt.figure(figsize=(16, 8))
plt.plot(y_test_rescaled, color='black', label='Actual Price')
plt.plot(y_pred_rescaled, color='green', label='Predicted Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Weeks')
plt.ylabel('Price in USD')
plt.legend()
plt.show()

# 7. sMAPE calculation
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# 8. Model Evaluation (add sMAPE)
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
smape_value = smape(y_test_rescaled, y_pred_rescaled)

# Print metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(f"Symmetric Mean Absolute Percentage Error (sMAPE): {smape_value:.2f}%")
