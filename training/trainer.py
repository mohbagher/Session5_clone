"""
Model Training
==============
Training loop with validation and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Optional, Any, Tuple
import logging
from pathlib import Path

from config import Config

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_data: Dict,
    val_data: Dict,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    n_epochs: int = 50,
    device: str = 'cpu',
    widget_dict: Optional[Dict] = None,
    config: Optional[Config] = None
) -> Tuple[nn.Module, Dict]:
    """
    Train model with validation.

    Args:
        model: PyTorch model to train
        train_data: Training dataset dict with 'Y' and 'Phi_opt'
        val_data: Validation dataset dict
        batch_size: Batch size
        learning_rate: Learning rate
        n_epochs: Number of epochs
        device: Device to use
        widget_dict: Dashboard widgets for updates
        config: Configuration object

    Returns:
        trained_model: Trained model
        history: Training history
    """

    # Move model to device
    model = model.to(device)

    # Create dataloaders
    train_loader = create_dataloader(train_data, batch_size, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Log progress
        log_message = f"Epoch [{epoch+1}/{n_epochs}] Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        logger.info(log_message)

        if widget_dict and 'status_output' in widget_dict:
            with widget_dict['status_output']:
                print(log_message)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model (optional)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    return model, history


def create_dataloader(
    data: Dict,
    batch_size: int,
    shuffle: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader from data dict."""

    Y = torch.FloatTensor(data['Y'])
    Phi_opt = torch.FloatTensor(data['Phi_opt'])

    dataset = TensorDataset(Y, Phi_opt)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to 0 for compatibility
    )