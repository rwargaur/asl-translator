import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

SIGNS = ["hello", "thanks", "yes", "no", "please"]
SEQUENCES = 30
FRAMES = 30
DATA_PATH = "data"

print("Loading data...")
X, y = [], []
for label, sign in enumerate(SIGNS):
    for seq in range(SEQUENCES):
        seq_frames = []
        missing = False
        for frame_num in range(FRAMES):
            path = f"{DATA_PATH}/{sign}/{seq}/{frame_num}.npy"
            if not os.path.exists(path):
                missing = True
                break
            seq_frames.append(np.load(path))
        if not missing:
            X.append(seq_frames)
            y.append(label)

print(f"Loaded {len(X)} sequences. Starting training...")

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

class ASLModel(nn.Module):
    def __init__(self, input_size=63, hidden=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = ASLModel(num_classes=len(SIGNS))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model/lstm_model.pth")
print("Model saved!")