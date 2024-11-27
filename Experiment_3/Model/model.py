import torch
import torch.nn as nn

def ensure_correct_dimensions(x, expected_features):
    """
    Passt die Dimensionen eines Eingabetensors an, falls erforderlich.
    """
    if len(x.shape) == 2:  # Wenn die Batch-Dimension fehlt
        x = x.unsqueeze(0)  # Füge die Batch-Dimension hinzu
    elif len(x.shape) == 1:  # Wenn die Sequenz- und Batch-Dimension fehlen
        x = x.unsqueeze(0).unsqueeze(-1)
    assert x.shape[-1] == expected_features, f"Feature-Dimension stimmt nicht: {x.shape[-1]} vs {expected_features}"
    return x


class DualAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size_1=64, hidden_size_2=32, sequence_length=30):
        """
        Multi-LSTM-Modell mit Dual Attention Layer, Dropout, Residual Connections und Layer Normalization.
        """
        super(DualAttentionLSTM, self).__init__()
        self.sequence_length = sequence_length

        # LSTM-Schichten
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)

        # Dropout
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size_1)
        self.layer_norm2 = nn.LayerNorm(hidden_size_2)

        # Attention-Schichten
        self.attention1 = nn.Linear(hidden_size_2, sequence_length)
        self.attention2 = nn.Linear(hidden_size_2, sequence_length)

        # Kombinierte Attention-Schicht
        self.combine_attention = nn.Linear(hidden_size_2 * 2, hidden_size_2)

        # ReLU-Aktivierung
        self.relu = nn.ReLU()

        # Output-Schicht
        self.output_layer = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        """
        Vorwärtsdurchlauf durch das Modell.

        Args:
            x (torch.Tensor): Eingabedaten mit Form (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Vorhersagen mit Form (batch_size, 1).
        """
        # Dimensionen sicherstellen
        x = ensure_correct_dimensions(x, expected_features=self.lstm1.input_size)

        # Erste LSTM-Schicht
        x, _ = self.lstm1(x)
        x = self.dropout1(self.layer_norm1(x))  # Dropout + Layer Normalization

        # Zweite LSTM-Schicht
        x, _ = self.lstm2(x)
        x = self.dropout2(self.layer_norm2(x))  # Dropout + Layer Normalization

        # Attention-Schichten
        attn1 = self.attention1(x).softmax(dim=1)  # (batch_size, sequence_length, sequence_length)
        attn2 = self.attention2(x).softmax(dim=1)  # (batch_size, sequence_length, sequence_length)

        # Anwendung der Attention-Gewichte
        attn1 = attn1.transpose(1, 2)  # (batch_size, sequence_length, sequence_length)
        attn2 = attn2.transpose(1, 2)  # (batch_size, sequence_length, sequence_length)

        weighted1 = torch.bmm(attn1, x)  # (batch_size, sequence_length, hidden_size_2)
        weighted2 = torch.bmm(attn2, x)  # (batch_size, sequence_length, hidden_size_2)

        # Reduzierung entlang der Sequenzdimension
        weighted1 = weighted1.mean(dim=1)  # (batch_size, hidden_size_2)
        weighted2 = weighted2.mean(dim=1)  # (batch_size, hidden_size_2)

        # Kombination der Attention-Ausgaben
        combined = torch.cat((weighted1, weighted2), dim=1)  # (batch_size, hidden_size_2 * 2)
        combined = self.combine_attention(combined)  # (batch_size, hidden_size_2)

        # Residual Connection
        combined_with_residual = combined + x.mean(dim=1)  # Residual-Verbindung

        # Aktivierungsfunktion
        activated = self.relu(combined_with_residual)

        # Ausgabe
        output = self.output_layer(activated)  # (batch_size, 1)
        return output
