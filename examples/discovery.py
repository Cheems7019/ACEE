#!/usr/bin/env python
"""
Discovery: Causal Model Testing with Energy Validation

This script:
1. Loads discovery data (X1, X2, X3)
2. Fits Model 1: X1 -> X2 <- X3, X1 -> X3
   - Validates with Energy tests on (X1, X2) and (X1, X3)
   - Estimates E[X2|X1,X3] using Monte Carlo
3. Fits Model 2: X1 -> X2 -> X3
   - Validates with Energy tests on (X1, X2) and (X2, X3)
   - Tests X1 ⊥ X3 | X2 using MCODEC

Usage:
    python discovery.py --n_epochs 3000 --D 500 --seed 42
"""

import numpy as np
import pandas as pd
import torch
import os
import sys
import random
import pickle
import argparse
from sklearn.preprocessing import QuantileTransformer

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.conditional_ddpm import (
    train_all_conditional_ddpms,
    generate_conditional_samples
)
from utils.mcodec import mcodec

try:
    from hyppo.ksample import Energy
except ImportError:
    print("ERROR: hyppo package not found. Please install it: pip install hyppo")
    sys.exit(1)


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_energy_test(X, Y, alpha=0.2):
    """
    Run Energy two-sample test.
    
    Args:
        X: numpy array (n, d)
        Y: numpy array (n, d)
        alpha: significance level
    
    Returns:
        passed: bool - True if p-value > alpha (fail to reject null = same distribution)
        p_value: float - the p-value from the test
    """
    result = Energy().test(X, Y)
    p_value = result.pvalue
    passed = p_value > alpha
    return passed, p_value


def fit_model_1(data, hidden_dims, dim_t, n_steps, n_epochs, lr, device, 
                mc_samples, output_dir, alpha=0.2, max_seeds=11):
    """
    Model 1: X1 -> X2 <- X3, X1 -> X3
    
    Train with Energy validation on (X1, X2) and (X1, X3).
    Estimate E[X2|X1,X3] using Monte Carlo sampling.
    
    Args:
        alpha: significance level for Energy tests
        max_seeds: maximum seeds to try (0-10)
    
    Returns:
        success: bool
        conditional_mean: np.array or None
    """
    print("\n" + "="*70)
    print("MODEL 1: X1 -> X2 <- X3, X1 -> X3")
    print("="*70)
    
    n = data.shape[0]
    
    # Group assignment: each variable is its own group
    # Group 0: X1, Group 1: X2, Group 2: X3
    groups = torch.tensor([0, 1, 2], dtype=torch.long)
    
    # Adjacency matrix: X1 -> X2 <- X3, X1 -> X3
    # Row i, Col j = 1 means group i -> group j
    A = np.array([
        [0, 1, 1],  # X1 -> X2, X1 -> X3
        [0, 0, 0],  # X2 has no children
        [0, 1, 0]   # X3 -> X2
    ], dtype=np.float32)
    
    print("\nAdjacency Matrix (Model 1):")
    print("     X1  X2  X3")
    for i, var in enumerate(['X1', 'X2', 'X3']):
        print(f"{var}   {int(A[i,0])}   {int(A[i,1])}   {int(A[i,2])}")
    print(f"\nEnergy test α = {alpha}")
    print(f"Max seeds to try: {max_seeds}")
    
    model1_dir = os.path.join(output_dir, 'ckpt', 'discovery_model1')
    os.makedirs(model1_dir, exist_ok=True)
    
    failed_checkpoints = []
    
    for seed in range(max_seeds):
        print(f"\n{'-'*60}")
        print(f"Model 1 - Seed {seed}")
        print(f"{'-'*60}")
        
        # Set seed
        seed_everything(seed)
        
        # Normalize data
        print(f"Normalizing data with QuantileTransformer (seed={seed})...")
        qt = QuantileTransformer(output_distribution="normal", 
                                random_state=seed,
                                n_quantiles=min(1000, n))
        data_norm = qt.fit_transform(data)
        data_norm_tensor = torch.tensor(data_norm, dtype=torch.float32)
        
        # Train models
        print(f"Training Model 1...")
        models = train_all_conditional_ddpms(
            train_data=data_norm_tensor,
            groups=groups,
            A=A,
            hidden_dims=hidden_dims,
            dim_t=dim_t,
            n_steps=n_steps,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            verbose=True
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(model1_dir, f'models_seed{seed}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(models, f)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Validate with Energy tests
        print(f"\nValidating with Energy tests...")
        
        # Generate unconditional samples
        print(f"  Generating {n} unconditional samples...")
        generated_norm = generate_conditional_samples(
            models=models,
            groups=groups,
            A=A,
            do_vars=None,
            sample_vars=None,
            n_samples=n
        )  # Shape: [n, 1, 3]
        
        generated_norm = generated_norm.squeeze(1).cpu().detach().numpy()  # [n, 3]
        generated = qt.inverse_transform(generated_norm)
        
        # Test 1: (X1, X2)
        X1_X2_orig = data[:, [0, 1]]
        X1_X2_gen = generated[:, [0, 1]]
        passed_12, p_val_12 = run_energy_test(X1_X2_orig, X1_X2_gen, alpha)
        status_12 = "PASS" if passed_12 else "FAIL"
        print(f"  Energy test (X1, X2): p-value = {p_val_12:.4f} [{status_12}]")
        
        # Test 2: (X1, X3)
        X1_X3_orig = data[:, [0, 2]]
        X1_X3_gen = generated[:, [0, 2]]
        passed_13, p_val_13 = run_energy_test(X1_X3_orig, X1_X3_gen, alpha)
        status_13 = "PASS" if passed_13 else "FAIL"
        print(f"  Energy test (X1, X3): p-value = {p_val_13:.4f} [{status_13}]")
        
        if passed_12 and passed_13:
            print(f"\n{'='*60}")
            print(f"SUCCESS! Model 1 found good model at seed {seed}")
            print(f"{'='*60}")
            
            # Save as best model
            best_model_path = os.path.join(model1_dir, 'models.pkl')
            best_qt_path = os.path.join(model1_dir, 'transformer.pkl')
            best_meta_path = os.path.join(model1_dir, 'metadata.pkl')
            
            with open(best_model_path, 'wb') as f:
                pickle.dump(models, f)
            with open(best_qt_path, 'wb') as f:
                pickle.dump(qt, f)
            with open(best_meta_path, 'wb') as f:
                pickle.dump({
                    'seed': seed,
                    'groups': groups,
                    'A': A,
                    'hidden_dims': hidden_dims,
                    'dim_t': dim_t,
                    'n_steps': n_steps
                }, f)
            
            # Delete checkpoints
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            for failed_path in failed_checkpoints:
                if os.path.exists(failed_path):
                    os.remove(failed_path)
            
            print(f"Saved: models.pkl, transformer.pkl, metadata.pkl")
            
            # Proceed to compute E[X2|X1,X3]
            break
        else:
            failed_checkpoints.append(checkpoint_path)
            print(f"Seed {seed} failed Energy tests, trying next...")
    else:
        # All seeds failed
        print(f"\n{'='*60}")
        print(f"FAILED! Model 1 - no good model found after {max_seeds} seeds")
        print(f"{'='*60}")
        
        # Cleanup
        for failed_path in failed_checkpoints:
            if os.path.exists(failed_path):
                os.remove(failed_path)
        
        return False, None
    
    print(f"\nModel 1 saved to: {model1_dir}")
    
    # Estimate conditional expectation: E[X2|X1,X3]
    print(f"\nEstimating conditional expectation with {mc_samples} MC samples...")
    print(f"  Estimating E[X2|X1,X3]...")
    
    do_vars_X2 = {
        0: data_norm[:, 0],  # X1
        2: data_norm[:, 2]   # X3
    }
    sample_vars_X2 = [1]  # X2
    
    X2_samples_norm = generate_conditional_samples(
        models=models,
        groups=groups,
        A=A,
        do_vars=do_vars_X2,
        sample_vars=sample_vars_X2,
        n_samples=mc_samples
    )  # Shape: [mc_samples, n, 1]
    
    # Average over MC samples to get E[X2|X1,X3]
    X2_conditional_mean_norm = X2_samples_norm.mean(dim=0).cpu().numpy()  # [n, 1]
    
    # Transform back to original space
    print(f"  Transforming back to original space...")
    
    # Need to pad with observed X1 and X3 for inverse transform
    full_data_norm = np.column_stack([
        data_norm[:, 0],  # X1 (observed)
        X2_conditional_mean_norm.flatten(),  # E[X2|X1,X3]
        data_norm[:, 2]   # X3 (observed)
    ])
    
    full_data_original = qt.inverse_transform(full_data_norm)
    E_X2_given_X1_X3 = full_data_original[:, 1]  # [n,]
    
    # Save results
    results_dir = os.path.join(output_dir, 'results', 'discovery_model1')
    os.makedirs(results_dir, exist_ok=True)
    
    results_df = pd.DataFrame({
        'E_X2_given_X1_X3': E_X2_given_X1_X3
    })
    results_path = os.path.join(results_dir, 'conditional_expectations.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\nConditional expectation saved to: {results_path}")
    print(f"  E[X2|X1,X3]: mean={E_X2_given_X1_X3.mean():.4f}, "
          f"std={E_X2_given_X1_X3.std():.4f}")
    
    return True, E_X2_given_X1_X3


def fit_model_2(data, hidden_dims, dim_t, n_steps, n_epochs, lr, device, 
                D, D_batch, output_dir, alpha=0.2, max_seeds=11):
    """
    Model 2: X1 -> X2 -> X3
    
    Train with Energy validation on (X1, X2) and (X2, X3).
    Test X1 ⊥ X3 | X2 using MCODEC.
    
    Args:
        alpha: significance level for Energy tests
        max_seeds: maximum seeds to try (0-10)
    
    Returns:
        success: bool
        tau_obs: float or None
        tau_samples: np.array or None
        p_value: float or None
    """
    print("\n" + "="*70)
    print("MODEL 2: X1 -> X2 -> X3")
    print("="*70)
    
    n = data.shape[0]
    
    # Group assignment
    groups = torch.tensor([0, 1, 2], dtype=torch.long)
    
    # Adjacency matrix: X1 -> X2 -> X3
    A = np.array([
        [0, 1, 0],  # X1 -> X2
        [0, 0, 1],  # X2 -> X3
        [0, 0, 0]   # X3 has no children
    ], dtype=np.float32)
    
    print("\nAdjacency Matrix (Model 2):")
    print("     X1  X2  X3")
    for i, var in enumerate(['X1', 'X2', 'X3']):
        print(f"{var}   {int(A[i,0])}   {int(A[i,1])}   {int(A[i,2])}")
    print(f"\nEnergy test α = {alpha}")
    print(f"Max seeds to try: {max_seeds}")
    
    model2_dir = os.path.join(output_dir, 'ckpt', 'discovery_model2')
    os.makedirs(model2_dir, exist_ok=True)
    
    failed_checkpoints = []
    
    for seed in range(max_seeds):
        print(f"\n{'-'*60}")
        print(f"Model 2 - Seed {seed}")
        print(f"{'-'*60}")
        
        # Set seed
        seed_everything(seed)
        
        # Normalize data
        print(f"Normalizing data with QuantileTransformer (seed={seed})...")
        qt = QuantileTransformer(output_distribution="normal",
                                random_state=seed,
                                n_quantiles=min(1000, n))
        data_norm = qt.fit_transform(data)
        data_norm_tensor = torch.tensor(data_norm, dtype=torch.float32)
        
        # Train models
        print(f"Training Model 2...")
        models = train_all_conditional_ddpms(
            train_data=data_norm_tensor,
            groups=groups,
            A=A,
            hidden_dims=hidden_dims,
            dim_t=dim_t,
            n_steps=n_steps,
            n_epochs=n_epochs,
            lr=lr,
            device=device,
            verbose=True
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(model2_dir, f'models_seed{seed}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(models, f)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Validate with Energy tests
        print(f"\nValidating with Energy tests...")
        
        # Generate unconditional samples
        print(f"  Generating {n} unconditional samples...")
        generated_norm = generate_conditional_samples(
            models=models,
            groups=groups,
            A=A,
            do_vars=None,
            sample_vars=None,
            n_samples=n
        )  # Shape: [n, 1, 3]
        
        generated_norm = generated_norm.squeeze(1).cpu().detach().numpy()  # [n, 3]
        generated = qt.inverse_transform(generated_norm)
        
        # Test 1: (X1, X2)
        X1_X2_orig = data[:, [0, 1]]
        X1_X2_gen = generated[:, [0, 1]]
        passed_12, p_val_12 = run_energy_test(X1_X2_orig, X1_X2_gen, alpha)
        status_12 = "PASS" if passed_12 else "FAIL"
        print(f"  Energy test (X1, X2): p-value = {p_val_12:.4f} [{status_12}]")
        
        # Test 2: (X2, X3)
        X2_X3_orig = data[:, [1, 2]]
        X2_X3_gen = generated[:, [1, 2]]
        passed_23, p_val_23 = run_energy_test(X2_X3_orig, X2_X3_gen, alpha)
        status_23 = "PASS" if passed_23 else "FAIL"
        print(f"  Energy test (X2, X3): p-value = {p_val_23:.4f} [{status_23}]")
        
        if passed_12 and passed_23:
            print(f"\n{'='*60}")
            print(f"SUCCESS! Model 2 found good model at seed {seed}")
            print(f"{'='*60}")
            
            # Save as best model
            best_model_path = os.path.join(model2_dir, 'models.pkl')
            best_qt_path = os.path.join(model2_dir, 'transformer.pkl')
            best_meta_path = os.path.join(model2_dir, 'metadata.pkl')
            
            with open(best_model_path, 'wb') as f:
                pickle.dump(models, f)
            with open(best_qt_path, 'wb') as f:
                pickle.dump(qt, f)
            with open(best_meta_path, 'wb') as f:
                pickle.dump({
                    'seed': seed,
                    'groups': groups,
                    'A': A,
                    'hidden_dims': hidden_dims,
                    'dim_t': dim_t,
                    'n_steps': n_steps
                }, f)
            
            # Delete checkpoints
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            for failed_path in failed_checkpoints:
                if os.path.exists(failed_path):
                    os.remove(failed_path)
            
            print(f"Saved: models.pkl, transformer.pkl, metadata.pkl")
            
            # Proceed to MCODEC test
            break
        else:
            failed_checkpoints.append(checkpoint_path)
            print(f"Seed {seed} failed Energy tests, trying next...")
    else:
        # All seeds failed
        print(f"\n{'='*60}")
        print(f"FAILED! Model 2 - no good model found after {max_seeds} seeds")
        print(f"{'='*60}")
        
        # Cleanup
        for failed_path in failed_checkpoints:
            if os.path.exists(failed_path):
                os.remove(failed_path)
        
        return False, None, None, None
    
    print(f"\nModel 2 saved to: {model2_dir}")
    
    # Conditional Independence Test: X1 ⊥ X3 | X2
    print(f"\n{'='*70}")
    print("CONDITIONAL INDEPENDENCE TEST: X1 ⊥ X3 | X2")
    print(f"{'='*70}")
    
    X1_data = data[:, 0:1]  # [n, 1]
    X2_data = data[:, 1:2]  # [n, 1]
    X3_data = data[:, 2:3]  # [n, 1]
    
    # Compute tau_observed = MCODEC(X3, X1, X2)
    print(f"\nComputing τ_obs = MCODEC(X3, X1, X2)...")
    tau_observed = mcodec(X3_data, X1_data, X2_data)
    print(f"  τ_obs = {tau_observed:.6f}")
    
    # Sample X3 D times given X2 and compute MCODEC
    print(f"\nSampling X3 {D} times from conditional model...")
    tau_samples = []
    
    mc_batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])
    
    for batch_idx, mc_batch_size in enumerate(mc_batch_sizes):
        print(f"  Batch {batch_idx+1}/{len(mc_batch_sizes)}: "
              f"Generating {mc_batch_size} samples...")
        
        # Condition on X1 and X2 (in normalized space)
        do_vars = {
            0: data_norm[:, 0],  # X1
            1: data_norm[:, 1]   # X2
        }
        sample_vars = [2]  # X3
        
        # Generate samples
        X3_sampled_batch_norm = generate_conditional_samples(
            models=models,
            groups=groups,
            A=A,
            do_vars=do_vars,
            sample_vars=sample_vars,
            n_samples=mc_batch_size
        )  # [mc_batch_size, n, 1]
        
        X3_sampled_batch_norm = X3_sampled_batch_norm.cpu().numpy()
        
        # Transform back and compute MCODEC
        for i in range(mc_batch_size):
            # Reconstruct full data
            full_data_norm = np.column_stack([
                data_norm[:, 0],  # X1
                data_norm[:, 1],  # X2
                X3_sampled_batch_norm[i, :, 0]  # X3 (sampled)
            ])
            
            full_data = qt.inverse_transform(full_data_norm)
            X3_sampled = full_data[:, 2:3]
            
            # Compute MCODEC(X3_sampled, X1, X2)
            tau_d = mcodec(X3_sampled, X1_data, X2_data)
            tau_samples.append(tau_d)
    
    tau_samples = np.array(tau_samples)
    
    print(f"\nGenerated {len(tau_samples)} τ^(d) values")
    print(f"  τ^(d) range: [{tau_samples.min():.6f}, {tau_samples.max():.6f}]")
    print(f"  τ^(d) mean:  {tau_samples.mean():.6f}")
    print(f"  τ^(d) std:   {tau_samples.std():.6f}")
    
    # Compute p-value
    num_larger_or_equal = np.sum(tau_samples >= tau_observed)
    p_value = (1 + num_larger_or_equal) / (D + 1)
    
    print(f"\nResults:")
    print(f"  τ_obs:               {tau_observed:.6f}")
    print(f"  # of τ^(d) ≥ τ_obs:  {num_larger_or_equal}")
    print(f"  P-value:             {p_value:.6f}")
    print(f"\nInterpretation at α=0.05:")
    if p_value <= 0.05:
        print(f"  → REJECT H0: Evidence against X1 ⊥ X3 | X2")
    else:
        print(f"  → FAIL TO REJECT H0: No evidence against X1 ⊥ X3 | X2")
    
    # Save results
    results_dir = os.path.join(output_dir, 'results', 'discovery_model2')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save tau samples
    tau_samples_df = pd.DataFrame({
        'sample_idx': range(1, len(tau_samples) + 1),
        'tau_value': tau_samples
    })
    tau_samples_df.to_csv(os.path.join(results_dir, 'tau_samples.csv'), index=False)
    
    # Save tau observed
    tau_obs_df = pd.DataFrame({'tau_observed': [tau_observed]})
    tau_obs_df.to_csv(os.path.join(results_dir, 'tau_observed.csv'), index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  - tau_samples.csv ({len(tau_samples)} samples)")
    print(f"  - tau_observed.csv")
    
    return True, tau_observed, tau_samples, p_value


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='Discovery: Causal Model Testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument('--data_path', type=str, 
                       default=os.path.join(script_dir, 'data', 'discovery.csv'),
                       help='Path to discovery data')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                       default=[256, 128, 128, 64],
                       help='Hidden dimensions for MLP')
    parser.add_argument('--dim_t', type=int, default=32,
                       help='Time embedding dimension')
    parser.add_argument('--n_steps', type=int, default=1000,
                       help='Number of diffusion steps')
    parser.add_argument('--n_epochs', type=int, default=3000,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    
    # Sampling parameters
    parser.add_argument('--mc_samples', type=int, default=400,
                       help='MC samples for conditional expectation (Model 1)')
    parser.add_argument('--D', type=int, default=500,
                       help='MC samples for MCODEC test (Model 2)')
    parser.add_argument('--D_batch', type=int, default=250,
                       help='Batch size for MC sampling')
    
    # Validation parameters
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Significance level for Energy tests')
    parser.add_argument('--max_seeds', type=int, default=11,
                       help='Maximum seeds to try (0 to max_seeds-1)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for data loading')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda:0, cpu). If None, auto-detect.')
    parser.add_argument('--output_dir', type=str, default=script_dir,
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DISCOVERY: CAUSAL MODEL TESTING")
    print("="*70)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Set device
    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\nUsing device: {device}")
    
    # Set seed
    seed_everything(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Load data
    print(f"\nLoading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    data = df[['X1', 'X2', 'X3']].values
    
    print(f"Data shape: {data.shape}")
    print(f"  X1: mean={data[:, 0].mean():.4f}, std={data[:, 0].std():.4f}")
    print(f"  X2: mean={data[:, 1].mean():.4f}, std={data[:, 1].std():.4f}")
    print(f"  X3: mean={data[:, 2].mean():.4f}, std={data[:, 2].std():.4f}")
    
    # Fit Model 1
    success_1, conditional_mean = fit_model_1(
        data=data,
        hidden_dims=args.hidden_dims,
        dim_t=args.dim_t,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        mc_samples=args.mc_samples,
        output_dir=args.output_dir,
        alpha=args.alpha,
        max_seeds=args.max_seeds
    )
    
    if not success_1:
        print("\n" + "="*70)
        print("STOPPING: Model 1 failed to train a good model")
        print("="*70)
        print("\nPlease consider:")
        print("  1. Increasing n_epochs")
        print("  2. Adjusting learning rate")
        print("  3. Increasing model capacity (hidden_dims, dim_t)")
        print("  4. Relaxing α (currently 0.2)")
        print("  5. Checking data quality")
        return
    
    # Fit Model 2
    success_2, tau_obs, tau_samples, p_value = fit_model_2(
        data=data,
        hidden_dims=args.hidden_dims,
        dim_t=args.dim_t,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        D=args.D,
        D_batch=args.D_batch,
        output_dir=args.output_dir,
        alpha=args.alpha,
        max_seeds=args.max_seeds
    )
    
    if not success_2:
        print("\n" + "="*70)
        print("STOPPING: Model 2 failed to train a good model")
        print("="*70)
        print("\nPlease consider:")
        print("  1. Increasing n_epochs")
        print("  2. Adjusting learning rate")
        print("  3. Increasing model capacity (hidden_dims, dim_t)")
        print("  4. Relaxing α (currently 0.2)")
        print("  5. Checking data quality")
        return
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nModel 1: X1 -> X2 <- X3, X1 -> X3")
    print(f"  Status: SUCCESS")
    print(f"  Conditional expectation saved")
    print(f"  E[X2|X1,X3]: mean={conditional_mean.mean():.4f}, std={conditional_mean.std():.4f}")
    
    print("\nModel 2: X1 -> X2 -> X3")
    print(f"  Status: SUCCESS")
    print(f"  Test: X1 ⊥ X3 | X2")
    print(f"  τ_obs:   {tau_obs:.6f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Decision: {'REJECT H0' if p_value <= 0.05 else 'FAIL TO REJECT H0'}")
    
    print("\n" + "="*70)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)


if __name__ == '__main__':
    main()
