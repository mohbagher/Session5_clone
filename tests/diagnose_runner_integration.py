# ============================================================================
# EXPERIMENT RUNNER DIAGNOSTIC - Find Integration Issues
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

print("=" * 70)
print("EXPERIMENT RUNNER DIAGNOSTIC")
print("=" * 70)

# ============================================================================
# STEP 1: Replicate Your Exact Config
# ============================================================================
print("\n[STEP 1] Creating Configuration")
print("-" * 70)

config = {
    # System
    'N': 32,
    'K': 64,
    'M': 8,
    'probe_type': 'continuous',

    # Data
    'n_train': 100,
    'n_val': 20,
    'n_test': 20,

    # Training
    'batch_size': 16,
    'learning_rate': 0.001,
    'n_epochs': 20,
    'device': 'cpu',

    # System params
    'P_tx': 1.0,
    'sigma_h_sq': 1.0,
    'sigma_g_sq': 1.0,
    'seed': 42,

    # Model
    'model_name': 'Baseline_MLP',
    'hidden_sizes': [256, 128],
    'dropout_prob': 0.1,
    'use_batch_norm': True
}

print(f"✓ Config created")
print(f"  Training samples: {config['n_train']}")
print(f"  Model: {config['model_name']}")
print(f"  Hidden layers: {config['hidden_sizes']}")

# ============================================================================
# STEP 2: Generate Data Using Your Pipeline
# ============================================================================
print("\n[STEP 2] Generating Data (Your Pipeline)")
print("-" * 70)

from data.probe_generators import get_probe_bank
from data.data_generation import generate_limited_probing_dataset

# Generate probe bank
probe_bank = get_probe_bank(
    probe_type=config['probe_type'],
    K=config['K'],
    N=config['N'],
    seed=config['seed']
)
print(f"✓ Probe bank generated")

# System config dict
system_config_dict = {
    'N': config['N'],
    'K': config['K'],
    'M': config['M'],
    'P_tx': config['P_tx'],
    'sigma_h_sq': config['sigma_h_sq'],
    'sigma_g_sq': config['sigma_g_sq'],
    'seed': config['seed']
}

# Generate datasets exactly as experiment_runner does
train_data = generate_limited_probing_dataset(
    probe_bank=probe_bank,
    n_samples=config['n_train'],
    M=config['M'],
    system_config=system_config_dict,
    normalize=True,
    seed=config['seed']
)

val_data = generate_limited_probing_dataset(
    probe_bank=probe_bank,
    n_samples=config['n_val'],
    M=config['M'],
    system_config=system_config_dict,
    normalize=True,
    seed=config['seed'] + 1
)

print(f"✓ Datasets generated")
print(f"  Train: {len(train_data['labels'])} samples")
print(f"  Val: {len(val_data['labels'])} samples")

# DIAGNOSTIC CHECK
print(f"\n  Train data stats:")
print(f"    masked_powers mean: {np.mean(train_data['masked_powers']):.6f}")
print(f"    masked_powers std: {np.std(train_data['masked_powers']):.6f}")
print(f"    Unique labels: {len(np.unique(train_data['labels']))}")

# ============================================================================
# STEP 3: Create DataLoaders (Your Way)
# ============================================================================
print("\n[STEP 3] Creating DataLoaders")
print("-" * 70)

# Prepare inputs exactly as your train_model function does
train_inputs = torch.cat([
    torch.FloatTensor(train_data['masked_powers']),
    torch.FloatTensor(train_data['masks'])
], dim=1)
train_labels = torch.LongTensor(train_data['labels'])

val_inputs = torch.cat([
    torch.FloatTensor(val_data['masked_powers']),
    torch.FloatTensor(val_data['masks'])
], dim=1)
val_labels = torch.LongTensor(val_data['labels'])

print(f"✓ Input tensors created")
print(f"  Train inputs shape: {train_inputs.shape}")
print(f"  Expected: ({config['n_train']}, {2 * config['K']})")
print(f"  Train labels shape: {train_labels.shape}")

# DIAGNOSTIC CHECK: Are inputs valid?
print(f"\n  Input stats:")
print(f"    Mean: {train_inputs.mean().item():.6f}")
print(f"    Std: {train_inputs.std().item():.6f}")
print(f"    Contains NaN: {torch.isnan(train_inputs).any().item()}")
print(f"    Contains Inf: {torch.isinf(train_inputs).any().item()}")

if torch.isnan(train_inputs).any() or torch.isinf(train_inputs).any():
    print("  ❌ PROBLEM: Inputs contain NaN or Inf!")

# Create datasets
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

print(f"✓ DataLoaders created")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")

# ============================================================================
# STEP 4: Create Model (Your Way)
# ============================================================================
print("\n[STEP 4] Creating Model")
print("-" * 70)

from models.base_models import BaselineMLPPredictor

model = BaselineMLPPredictor(
    input_size=2 * config['K'],
    hidden_sizes=config['hidden_sizes'],
    output_size=config['K'],
    dropout_prob=config['dropout_prob'],
    use_batch_norm=config['use_batch_norm']
)

model = model.to(config['device'])

print(f"✓ Model created and moved to {config['device']}")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# STEP 5: Training Setup (Your Way)
# ============================================================================
print("\n[STEP 5] Training Setup")
print("-" * 70)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['learning_rate']
)

print(f"✓ Criterion: CrossEntropyLoss")
print(f"✓ Optimizer: Adam (lr={config['learning_rate']})")

# ============================================================================
# STEP 6: Training Loop with VERBOSE Logging
# ============================================================================
print("\n[STEP 6] Training Loop (VERBOSE)")
print("-" * 70)

history = {
    'train_loss': [],
    'val_loss': [],
    'val_acc': []
}

for epoch in range(config['n_epochs']):
    # ========== TRAINING PHASE ==========
    model.train()
    train_loss = 0.0

    batch_count = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        # DIAGNOSTIC: First batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"\n  🔍 FIRST BATCH DIAGNOSTIC:")
            print(f"    Batch inputs shape: {inputs.shape}")
            print(f"    Batch labels shape: {labels.shape}")
            print(f"    Input mean: {inputs.mean().item():.6f}")
            print(f"    Input std: {inputs.std().item():.6f}")
            print(f"    Labels range: [{labels.min().item()}, {labels.max().item()}]")

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # DIAGNOSTIC: First batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"    Output shape: {outputs.shape}")
            print(f"    Output mean: {outputs.mean().item():.6f}")
            print(f"    Output std: {outputs.std().item():.6f}")

        loss = criterion(outputs, labels)

        # DIAGNOSTIC: First batch of first epoch
        if epoch == 0 and batch_idx == 0:
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Expected random: ~{np.log(config['K']):.4f}")

        # Backward pass
        loss.backward()

        # DIAGNOSTIC: Check gradients in first batch
        if epoch == 0 and batch_idx == 0:
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
            print(f"    Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")

        optimizer.step()

        train_loss += loss.item()
        batch_count += 1

    train_loss /= len(train_loader)

    # ========== VALIDATION PHASE ==========
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total if total > 0 else 0.0

    # Record history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Print every 5 epochs or last epoch
    if epoch % 5 == 0 or epoch == config['n_epochs'] - 1:
        print(f"  Epoch {epoch + 1:2d}/{config['n_epochs']}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Loss={val_loss:.4f}, "
              f"Val Acc={val_acc:.4f}")

# ============================================================================
# STEP 7: Analyze Training Results
# ============================================================================
print("\n[STEP 7] Training Results Analysis")
print("-" * 70)

initial_train_loss = history['train_loss'][0]
final_train_loss = history['train_loss'][-1]
initial_val_acc = history['val_acc'][0]
final_val_acc = history['val_acc'][-1]

loss_decrease = initial_train_loss - final_train_loss
acc_increase = final_val_acc - initial_val_acc

print(f"  Initial train loss: {initial_train_loss:.4f}")
print(f"  Final train loss: {final_train_loss:.4f}")
print(f"  Loss decrease: {loss_decrease:.4f}")
print(f"")
print(f"  Initial val accuracy: {initial_val_acc:.4f}")
print(f"  Final val accuracy: {final_val_acc:.4f}")
print(f"  Accuracy increase: {acc_increase:.4f}")

# Decision
if loss_decrease < 0.5:
    print("\n  ❌ PROBLEM FOUND: Loss barely decreased in experiment runner flow!")
    print("     → Model is NOT learning through the pipeline")
else:
    print("\n  ✓ Model IS learning through experiment runner!")

# ============================================================================
# STEP 8: Compare with Isolated Training
# ============================================================================
print("\n[STEP 8] Comparison with Isolated Training")
print("-" * 70)

# Train fresh model on same data but isolated
print("  Training fresh model in isolation...")
model_isolated = BaselineMLPPredictor(
    input_size=2 * config['K'],
    hidden_sizes=config['hidden_sizes'],
    output_size=config['K'],
    dropout_prob=config['dropout_prob'],
    use_batch_norm=config['use_batch_norm']
)
model_isolated = model_isolated.to(config['device'])

optimizer_isolated = torch.optim.Adam(
    model_isolated.parameters(),
    lr=config['learning_rate']
)

# Quick training (10 epochs)
isolated_losses = []
for epoch in range(10):
    model_isolated.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        optimizer_isolated.zero_grad()
        outputs = model_isolated(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_isolated.step()

        epoch_loss += loss.item()

    isolated_losses.append(epoch_loss / len(train_loader))

isolated_decrease = isolated_losses[0] - isolated_losses[-1]

print(f"  Isolated training loss decrease: {isolated_decrease:.4f}")
print(f"  Pipeline training loss decrease: {loss_decrease:.4f}")
print(f"  Difference: {abs(isolated_decrease - loss_decrease):.4f}")

if abs(isolated_decrease - loss_decrease) > 1.0:
    print("\n  ⚠️ Large difference detected!")
    print("     → Problem is in the pipeline integration")
else:
    print("\n  ✓ Both methods show similar learning")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

issues = []

if torch.isnan(train_inputs).any() or torch.isinf(train_inputs).any():
    issues.append("Input data contains NaN/Inf")

if loss_decrease < 0.5:
    issues.append("Model not learning in experiment runner flow")

if abs(isolated_decrease - loss_decrease) > 1.0:
    issues.append("Pipeline integration differs from isolated training")

if issues:
    print("\n❌ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    print("\n🔍 NEXT STEPS:")
    if "Input data contains NaN/Inf" in issues:
        print("  → Check data_generation.py normalize function")
    if "Model not learning" in issues:
        print("  → Check if model is set to .train() mode")
        print("  → Verify optimizer is updating parameters")
    if "Pipeline integration" in issues:
        print("  → Compare experiment_runner train_model with this script")
        print("  → Check for missing .train()/.eval() calls")
else:
    print("\n✓ Experiment runner flow works correctly!")
    print("  The issue must be in a specific configuration or experiment setup.")

print("=" * 70)