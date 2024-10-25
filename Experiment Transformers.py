import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset (you can use your btc_weekly_data)
btc_data = pd.read_csv('../btc_modeling_data.csv')

# Convert timestamp to datetime and use the 'Close' price as the target feature
btc_data['Timestamp'] = pd.to_datetime(btc_data['Timestamp'])
prices = btc_data['Close'].values.reshape(-1, 1)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)


# Define a helper function to create sequences
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)


# Define sequence length (e.g., 30 days)
seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# Split into training and test datasets (80% training, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Step 3: Define Transformer Model
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_encoder_layers, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, model_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        # Fully connected layer for regression
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        # Add positional encoding
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1])  # Take the output of the last time step
        return x


# Step 4: Hyperparameters
input_dim = 1  # Single input feature (Close price)
model_dim = 64  # Size of the transformer model
num_heads = 4  # Number of heads in the multi-head attention mechanism
num_encoder_layers = 3  # Number of transformer encoder layers
dropout = 0.1  # Dropout for regularization

# Instantiate the model
model = TransformerTimeSeries(input_dim, model_dim, num_heads, num_encoder_layers, dropout)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 5: Model Training
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(X_train)
    loss = criterion(output, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# Step 6: Evaluate on the test set
model.eval()
with torch.no_grad():
    predicted_prices = model(X_test).squeeze().numpy()

# Step 7: Reverse scaling
predicted_prices_rescaled = scaler.inverse_transform(predicted_prices.reshape(-1, 1))
actual_prices_rescaled = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Step 8: Calculate Metrics
mse = mean_squared_error(actual_prices_rescaled, predicted_prices_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices_rescaled, predicted_prices_rescaled)
r2 = r2_score(actual_prices_rescaled, predicted_prices_rescaled)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Step 9: Plot Results
plt.figure(figsize=(12, 6))
plt.plot(actual_prices_rescaled, label="Actual Price", color='black')
plt.plot(predicted_prices_rescaled, label="Predicted Price", color='green')
plt.title("Bitcoin Price Prediction")
plt.xlabel("Weeks")
plt.ylabel("Price in USD")
plt.legend()
plt.show()
