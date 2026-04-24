import os
from pathlib import Path
import torch
import torch.nn as nn
from model import SixLayerPoolCNN
from dataset import get_loaders
from utils import test_model
from configs import batch_size, num_workers_local, model_name, learning_rate, step_size, gamma, num_epochs, patience, num_workers_cloud

device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_loader, val_loader, test_loader = get_loaders(
    batch_size,
    num_workers_cloud if device == 'cuda' else num_workers_local
)

model_exists = list(Path(f"models/{model_name}").glob("*.tar"))
model = SixLayerPoolCNN().to(device)

# -------------------------
# Load existing model if available
# -------------------------
if model_exists:
    print(f"Found saved model at {model_exists[0]}. Loading and skipping training...")
    model.load_state_dict(torch.load(model_exists[0], map_location=device))
    skip_training = True
else:
    print("No saved model found. Starting training...")
    skip_training = False

if not skip_training:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    train_accs, val_accs = [], []

    best_val = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        val_acc = test_model(model, val_loader, device)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: train {train_acc:.3f} val {val_acc:.3f}")

        scheduler.step()

        # save best model
        if val_acc > best_val:
            best_val = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    final_test = test_model(model, test_loader, device)
    print("Test acc:", final_test)

else:
    # If skipping training, still evaluate
    final_test = test_model(model, test_loader, device)
    print("Loaded model test acc:", final_test)