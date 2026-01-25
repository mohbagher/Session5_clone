"""
Training Loop (Phase 2) - CORRECTED PHYSICS MAPPING
===================================================
Aligned with PhD Definitions:
- N: RIS Elements (Physics dimension)
- K: Probe Library Size (Model Output dimension / Label space)
- M: Sensing Budget (Subset size)

Model Input: 2*K (Sparse power vector + Mask vector)
Model Output: K (Scores for every candidate in the library)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.base_models import BaselineMLPPredictor, ResidualMLPPredictor, AttentionMLPPredictor

def get_model_class(preset_name):
    """Maps string preset to class."""
    if 'Residual' in preset_name: return ResidualMLPPredictor
    if 'Attention' in preset_name: return AttentionMLPPredictor
    return BaselineMLPPredictor

def train_model_phase2(train_ds, val_ds, config, callback=None):
    """
    Main training loop.
    """
    # 1. Config Setup
    device = config.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    batch_size = int(config.get('batch_size', 32))
    lr = float(config.get('learning_rate', 0.001))
    epochs = int(config.get('n_epochs', 10))

    # PHD DEFINITIONS
    # K = Total Probe Library Size (The Search Space)
    # M = Sensing Budget (The constraint)
    K = int(config['K'])
    M = int(config['M'])

    # 2. Data Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Model Init
    ModelClass = get_model_class(config.get('model_preset', 'Baseline_MLP'))

    # Feature Vector Logic from pipeline.py:
    # Input = [sensed_powers (size K), mask_vec (size K)] -> Total 2*K
    input_dim = 2 * K

    hidden_layers = [256, 128] # Default sizes

    model = ModelClass(
        input_size=input_dim,
        hidden_sizes=hidden_layers,
        # CRITICAL FIX: Output size is K because we classify into the full library
        output_size=K,
        dropout_prob=config.get('dropout_prob', 0.2),
        use_batch_norm=config.get('use_batch_norm', True)
    ).to(device)

    # 4. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                logits = model(X_val)
                loss = criterion(logits, y_val)
                val_loss += loss.item() * X_val.size(0)

                # Accuracy = % of time we picked the exact best probe from K
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)

        epoch_val_loss = val_loss / len(val_ds)
        epoch_acc = correct / total

        # Update History
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_acc)

        # Live Callback to Dashboard
        if callback:
            logs = {
                'train_loss': epoch_loss,
                'val_loss': epoch_val_loss,
                'val_acc': epoch_acc
            }
            callback(epoch, logs)

    return model, history