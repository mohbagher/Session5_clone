import torch
from torch.utils.data import DataLoader


def train_model_phase2(model, train_ds, val_ds, config):
    """Standard Phase 2 Training Loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.get('batch_size', 32))

    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(config.get('epochs', 10)):
        # Train
        model.train()
        loss_sum = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        history['train_loss'].append(loss_sum / len(train_loader))

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                preds = torch.argmax(model(X), dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        history['val_acc'].append(correct / total if total > 0 else 0)

    return model, history