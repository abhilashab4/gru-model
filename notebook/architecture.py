
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size: int, units: int = 128, dropout: float = 0.3):
        super().__init__()

        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=units,
            num_layers=1,        
            batch_first=True
        )

        self.bn = nn.BatchNorm1d(units)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(units, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out, _ = self.gru(x)     # shape: (batch, seq_len, units)
        out = out[:, -1, :]      # last timestep (return_sequences=False)

        out = self.bn(out)       # BatchNorm expects (batch, features)
        out = self.dropout(out)
        out = self.fc(out)

        return out

