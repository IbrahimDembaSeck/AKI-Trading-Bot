import torch
from torch.utils.data import DataLoader, TensorDataset
from model import DualAttentionLSTM
import pickle
import os
import math
import matplotlib.pyplot as plt

# Verzeichnis mit den vorbereiteten Daten
DATA_DIR = "..\data preprocessing"

# Hyperparameter
INPUT_SIZE = 20  # Anzahl der Features aus den Preprocessing-Daten
SEQUENCE_LENGTH = 30
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
MODEL_SAVE_PATH = "dual_attention_lstm.pth"

# Daten laden
def load_data(data_dir):
    """Lädt die vorbereiteten Daten aus Pickle-Dateien."""
    with open(os.path.join(data_dir, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(data_dir, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(data_dir, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(data_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)
    return X_train, y_train, X_test, y_test

# DataLoader erstellen
def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    """Erstellt DataLoader für Training und Test."""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Modell initialisieren
def initialize_model(input_size, hidden_size_1, hidden_size_2, sequence_length, learning_rate):
    """Initialisiert das Modell, den Optimierer und die Verlustfunktion."""
    model = DualAttentionLSTM(
        input_size=input_size,
        hidden_size_1=hidden_size_1,
        hidden_size_2=hidden_size_2,
        sequence_length=sequence_length
    )
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, criterion, optimizer

# MAE und RMSE berechnen
def calculate_mae(y_true, y_pred):
    """Berechnet den Mean Absolute Error (MAE)."""
    return torch.mean(torch.abs(y_true - y_pred)).item()

def calculate_rmse(y_true, y_pred):
    """Berechnet den Root Mean Squared Error (RMSE)."""
    return math.sqrt(torch.mean((y_true - y_pred) ** 2).item())

def plot_metrics(train_losses, val_losses, train_mae, val_mae):
    """Visualisiert Verluste und MAE-Werte während des Trainings."""
    plt.figure(figsize=(12, 8))

    # Subplot 1: Verluste
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.title("Trainings- und Validierungsverluste")
    plt.legend()
    plt.grid(True)

    # Subplot 2: MAE-Werte
    plt.subplot(2, 1, 2)
    plt.plot(train_mae, label="Train MAE", marker='o')
    plt.plot(val_mae, label="Validation MAE", marker='x')
    plt.xlabel("Epoche")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Trainings- und Validierungs-MAE")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Training
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    """Trainiert das Modell und speichert Train- und Validation-Losses."""
    print("Starte Training...")
    train_losses = []
    val_losses = []
    train_mae_values = []
    val_mae_values = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            mae = calculate_mae(batch_y, outputs)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += mae

        epoch_loss /= len(train_loader)
        epoch_mae /= len(train_loader)
        train_losses.append(epoch_loss)
        train_mae_values.append(epoch_mae)

        # Validierung nach jeder Epoche
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                mae = calculate_mae(batch_y, outputs)

                val_loss += loss.item()
                val_mae += mae

        val_loss /= len(test_loader)
        val_mae /= len(test_loader)
        val_losses.append(val_loss)
        val_mae_values.append(val_mae)

        print(
            f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Train MAE: {epoch_mae:.4f}, Val MAE: {val_mae:.4f}"
        )

    return train_losses, val_losses, train_mae_values, val_mae_values

# Verluste plotten
def plot_losses(train_losses, val_losses):
    """Visualisiert den Verlauf von Trainings- und Validierungsverlusten."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Bewertung des Modells
def evaluate_model(model, test_loader, criterion):
    """Bewertet das Modell auf den Testdaten."""
    print("\nBewertung auf den Testdaten...")
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss:.4f}")

# Modell speichern
def save_model(model, save_path):
    """Speichert das trainierte Modell."""
    torch.save(model.state_dict(), save_path)
    print(f"Modell erfolgreich gespeichert unter: {save_path}")

# Main-Funktion
def main():
    # Daten laden
    X_train, y_train, X_test, y_test = load_data(DATA_DIR)

    # Zielwerte prüfen und anpassen
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(1)  # Falls nötig, Zielwerte zu (batch_size, 1) umformen
    if y_test.dim() == 1:
        y_test = y_test.unsqueeze(1)

    # DataLoader erstellen
    train_loader, test_loader = create_dataloaders(
        X_train.clone().detach().float(),
        y_train.clone().detach().float(),
        X_test.clone().detach().float(),
        y_test.clone().detach().float(),
        BATCH_SIZE
    )

    # Modell initialisieren
    model, criterion, optimizer = initialize_model(
        input_size=INPUT_SIZE,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=LEARNING_RATE
    )

    # Modell trainieren
    train_losses, val_losses, train_mae_values, val_mae_values = train_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS
    )

    # Verluste visualisieren
    plot_losses(train_losses, val_losses)
    plot_metrics(train_losses, val_losses, train_mae_values, val_mae_values)

    # Modell bewerten
    evaluate_model(model, test_loader, criterion)

    # Modell speichern
    save_model(model, MODEL_SAVE_PATH)

# Startpunkt
if __name__ == "__main__":
    main()
