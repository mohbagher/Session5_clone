"""
Training loop for RIS probe-based control with limited probing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
import copy

from config import Config
from model import LimitedProbingMLP


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.best_model_state = None
        self.should_stop = False
    
    def __call__(self, value: float, model: nn.Module) -> bool:
        if self.best_value is None:
            self.best_value = value
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        
        if self.mode == 'min': 
            improved = value < self.best_value - self.min_delta
        else: 
            improved = value > self.best_value + self.min_delta
        
        if improved: 
            self.best_value = value
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
    
    def restore_best_model(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class TrainingHistory:
    """Track training metrics over epochs."""
    
    def __init__(self):
        self.train_loss:  List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.val_eta: List[float] = []
        self.learning_rates: List[float] = []
    
    def add_epoch(self, train_loss: float, val_loss: float,
                  train_acc: float, val_acc: float,
                  val_eta: float, lr: float):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.val_eta.append(val_eta)
        self.learning_rates.append(lr)
    
    def to_dict(self) -> Dict:
        return {
            'train_loss': self.train_loss,
            'val_loss':  self.val_loss,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc,
            'val_eta': self.val_eta,
            'learning_rates': self.learning_rates
        }


def train_one_epoch(model: nn.Module,
                    train_loader:  DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: str) -> Tuple[float, float]: 
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader: 
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += inputs.size(0)
    
    return total_loss / total, correct / total


def validate(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: str,
             powers_full: np.ndarray,
             labels: np.ndarray) -> Tuple[float, float, float]:
    """Validate model and compute eta metric."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    
    with torch.no_grad():
        for inputs, batch_labels in val_loader: 
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, batch_labels)
            
            total_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += inputs.size(0)
            all_predictions.append(predictions.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # Compute eta
    all_predictions = np.concatenate(all_predictions)
    eta_values = []
    for i, pred in enumerate(all_predictions):
        P_selected = powers_full[i, pred]
        P_best = powers_full[i, labels[i]]
        if P_best > 0:
            eta_values.append(P_selected / P_best)
    eta_top1 = np.mean(eta_values) if eta_values else 0.0
    
    return avg_loss, accuracy, eta_top1


def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          config: Config,
          metadata: Dict) -> Tuple[nn.Module, TrainingHistory]: 
    """Full training loop."""
    device = config.training.device
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    early_stopping = EarlyStopping(
        patience=config.training.early_stop_patience,
        mode='max'
    )
    
    history = TrainingHistory()
    
    print(f"\nStarting training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Sensing budget: M={config.system.M} out of K={config.system.K} probes")
    print("-" * 70)
    
    for epoch in range(config.training.n_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_eta = validate(
            model, val_loader, criterion, device,
            metadata['val_powers_full'],
            metadata['val_labels']
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        history.add_epoch(train_loss, val_loss, train_acc, val_acc, val_eta, current_lr)
        
        if (epoch + 1) % config.training.eval_interval == 0:
            print(f"Epoch {epoch+1:3d}/{config.training.n_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val η: {val_eta:.4f}")
        
        scheduler.step(val_eta)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr: 
            print(f"  LR reduced:  {current_lr:.2e} -> {new_lr:.2e}")
        
        if early_stopping(val_eta, model):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    early_stopping.restore_best_model(model)
    print(f"\nTraining complete. Best validation η: {early_stopping.best_value:.4f}")
    
    return model, history