"""
Test normalization fix - pytest version
"""
import sys

sys.path.insert(0, '.')

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data.probe_generators import get_probe_bank
from data.data_generation import generate_limited_probing_dataset
from models.base_models import BaselineMLPPredictor


def train_quick_model(data, n_epochs=10):
    """Helper function to train a model quickly."""
    N, K, M = 32, 64, 8

    train_inputs = torch.cat([
        torch.FloatTensor(data['masked_powers']),
        torch.FloatTensor(data['masks'])
    ], dim=1)
    train_labels = torch.LongTensor(data['labels'])

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=128,
        shuffle=True
    )

    model = BaselineMLPPredictor(2 * K, [256, 128], K, 0.1, True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    initial_loss = None
    final_loss = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        if epoch == 0:
            initial_loss = epoch_loss
        if epoch == n_epochs - 1:
            final_loss = epoch_loss

    return initial_loss, final_loss


def test_broken_normalization():
    """Test that old normalization method performs poorly."""
    print("\n" + "=" * 70)
    print("TEST: Broken Normalization (mean_sample)")
    print("=" * 70)

    N, K, M = 32, 64, 8
    probe_bank = get_probe_bank('continuous', N=N, K=K, seed=42)

    system_config = {
        'N': N, 'K': K, 'M': M,
        'P_tx': 1.0, 'sigma_h_sq': 1.0, 'sigma_g_sq': 1.0
    }

    # Generate with BROKEN normalization
    train_data = generate_limited_probing_dataset(
        probe_bank=probe_bank,
        n_samples=10000,
        M=M,
        system_config=system_config,
        normalize=True,
        normalization_method='mean_sample',  # BROKEN
        seed=42
    )

    print(f"Data statistics:")
    print(f"  Masked powers mean: {np.mean(train_data['masked_powers']):.6f}")
    print(f"  Masked powers std: {np.std(train_data['masked_powers']):.6f}")

    initial_loss, final_loss = train_quick_model(train_data, n_epochs=10)
    loss_decrease = initial_loss - final_loss

    print(f"\nTraining results:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss decrease: {loss_decrease:.4f}")

    # Assert that broken method barely learns
    assert loss_decrease < 0.5, "Broken normalization should barely decrease loss"
    print("✓ Test passed: Broken normalization performs poorly as expected")


def test_fixed_normalization():
    """Test that fixed normalization method learns well."""
    print("\n" + "=" * 70)
    print("TEST: Fixed Normalization (max_global)")
    print("=" * 70)

    N, K, M = 32, 64, 8
    probe_bank = get_probe_bank('continuous', N=N, K=K, seed=42)

    system_config = {
        'N': N, 'K': K, 'M': M,
        'P_tx': 1.0, 'sigma_h_sq': 1.0, 'sigma_g_sq': 1.0
    }

    # Generate with FIXED normalization
    train_data = generate_limited_probing_dataset(
        probe_bank=probe_bank,
        n_samples=10000,
        M=M,
        system_config=system_config,
        normalize=True,
        normalization_method='max_global',  # FIXED
        seed=42
    )

    print(f"Data statistics:")
    print(f"  Masked powers mean: {np.mean(train_data['masked_powers']):.6f}")
    print(f"  Masked powers std: {np.std(train_data['masked_powers']):.6f}")

    initial_loss, final_loss = train_quick_model(train_data, n_epochs=10)
    loss_decrease = initial_loss - final_loss

    print(f"\nTraining results:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss decrease: {loss_decrease:.4f}")

    # Assert that fixed method learns significantly
    assert loss_decrease > 1.5, f"Fixed normalization should decrease loss significantly (got {loss_decrease:.4f})"
    print("✓ Test passed: Fixed normalization learns well")


def test_comparison():
    """Test that fixed is significantly better than broken."""
    print("\n" + "=" * 70)
    print("TEST: Comparison")
    print("=" * 70)

    N, K, M = 32, 64, 8
    probe_bank = get_probe_bank('continuous', N=N, K=K, seed=42)

    system_config = {
        'N': N, 'K': K, 'M': M,
        'P_tx': 1.0, 'sigma_h_sq': 1.0, 'sigma_g_sq': 1.0
    }

    # Test broken
    train_data_broken = generate_limited_probing_dataset(
        probe_bank=probe_bank,
        n_samples=10000,
        M=M,
        system_config=system_config,
        normalize=True,
        normalization_method='mean_sample',
        seed=42
    )
    _, final_loss_broken = train_quick_model(train_data_broken, n_epochs=10)

    # Test fixed
    train_data_fixed = generate_limited_probing_dataset(
        probe_bank=probe_bank,
        n_samples=10000,
        M=M,
        system_config=system_config,
        normalize=True,
        normalization_method='max_global',
        seed=42
    )
    _, final_loss_fixed = train_quick_model(train_data_fixed, n_epochs=10)

    improvement = final_loss_broken - final_loss_fixed

    print(f"\nComparison:")
    print(f"  Broken final loss: {final_loss_broken:.4f}")
    print(f"  Fixed final loss: {final_loss_fixed:.4f}")
    print(f"  Improvement: {improvement:.4f}")

    # Assert fixed is significantly better
    assert improvement > 1.0, f"Fixed should be much better than broken (improvement: {improvement:.4f})"
    print("✓ Test passed: Fixed normalization is significantly better")


if __name__ == '__main__':
    # Run tests
    print("=" * 70)
    print("NORMALIZATION FIX TESTS")
    print("=" * 70)

    test_broken_normalization()
    test_fixed_normalization()
    test_comparison()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)