#!/usr/bin/env python
"""
Assumption Check: SVD-based Conditional Independence Testing with Energy Validation

This script:
1. Generates data with latent factor H for two scenarios
2. Performs SVD-based rank-r approximation
3. Trains conditional diffusion models with Energy test validation
4. Tests conditional independence: X1 ⊥ Y1 | Z1 using Monte Carlo sampling
5. Saves all results for visualization

Usage:
    python Assumption_Check.py --n 500 --p 20 --range_H 2.0 --seed 1 --n_epochs 3000 --D 100
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import os
import sys
import random
import pickle
import argparse
import json
from datetime import datetime

# Add parent directory to path to import utils
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


def generate_X_two_scenarios(n, p, range_H, scenario='sc1', eps_sd=1.0, seed=None):
    """
    Generate data with latent factor H.
    
    Parameters:
    -----------
    n : int
        Number of samples
    p : int
        Number of variable pairs (total variables = 2p)
    range_H : float
        Range for uniform distribution of H: [-range_H, range_H]
    scenario : str
        'sc1' or 'sc2'
    eps_sd : float
        Standard deviation of noise
    seed : int or None
        Random seed
    
    Returns:
    --------
    dict with keys:
        'X' : ndarray of shape (n, 2p) - generated data
        'H' : ndarray of shape (n,) - latent factor
        'eps' : ndarray of shape (n, 2p) - noise terms
        'alpha' : ndarray of shape (p,) - scaling factors for first p variables
        'beta' : ndarray of shape (p,) - scaling factors for H in second p variables
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Latent factor H ~ Uniform(-range_H, range_H)
    H = np.random.uniform(-range_H, range_H, size=n)
    H2 = H ** 2
    
    # Noise for all 2p variables
    eps = np.random.normal(0, eps_sd, size=(n, 2 * p))
    
    # Sample alpha_i ~ Uniform[0.9, 1.1] for each of first p variables
    alpha = np.random.uniform(0.9, 1.1, size=p)
    
    # Sample beta_j ~ Uniform[0.9, 1.1] for each of second p variables
    beta = np.random.uniform(0.9, 1.1, size=p)
    
    # Initialize data matrix
    X = np.zeros((n, 2 * p))
    
    # First p variables: X_i = alpha_i * H² + ε_i
    for i in range(p):
        X[:, i] = alpha[i] * H2 + eps[:, i]
    
    # Next p variables: X_{p+i} depends on scenario
    for i in range(p, 2 * p):
        j = i - p  # corresponding first variable
        if scenario == 'sc1':
            # Scenario 1: X_i = H * (X_j - alpha_j * H²) + beta_j * H + ε_i
            X[:, i] = H * (X[:, j] - alpha[j] * H2) + beta[j] * H + eps[:, i]
        else:  # sc2
            # Scenario 2: X_i = X_j - alpha_j * H² + beta_j * H + ε_i
            X[:, i] = X[:, j] - alpha[j] * H2 + beta[j] * H + eps[:, i]
    
    return {
        'X': X,
        'H': H,
        'eps': eps,
        'alpha': alpha,
        'beta': beta
    }


def svd_rank_r_approx(X, r=2, center=False, scale=False):
    """
    Compute rank-r SVD approximation of X.
    
    Parameters:
    -----------
    X : ndarray of shape (n, p)
        Data matrix
    r : int
        Rank for approximation
    center : bool
        Whether to center the data
    scale : bool
        Whether to scale the data
    
    Returns:
    --------
    dict with keys:
        'Xc' : centered/scaled data
        'S' : rank-r approximation
        'U' : left singular vectors (first r)
        'D' : singular values (first r, for approximation)
        'D_all' : ALL singular values
        'V' : right singular vectors (first r)
    """
    # Standardize data
    if center or scale:
        scaler = StandardScaler(with_mean=center, with_std=scale)
        Xc = scaler.fit_transform(X)
    else:
        Xc = X.copy()
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    
    # Store all singular values
    s_all = s.copy()
    
    # Truncate to rank r
    r = min(r, len(s))
    Ur = U[:, :r]
    sr = s[:r]
    Vr = Vt[:r, :].T  # V, not V^T
    
    # Rank-r approximation: S = U_r @ diag(s_r) @ V_r^T
    S = Ur @ np.diag(sr) @ Vr.T
    
    return {
        'Xc': Xc,
        'S': S,
        'U': Ur,
        'D': sr,
        'D_all': s_all,
        'V': Vr
    }


def process_scenario(scenario_name, dat, S, S_result, p, n, seed, hidden_dims, dim_t, n_steps, 
                     n_epochs, lr, device, D, D_batch, use_standard_scaler, output_dir,
                     alpha=0.2, max_seeds=11):
    """
    Process a single scenario: prepare data, train models with Energy validation, and perform inference.
    
    Args:
        alpha: significance level for Energy tests
        max_seeds: maximum seeds to try (0 to max_seeds-1)
    """
    
    print(f"\n{'='*70}")
    print(f"PROCESSING SCENARIO: {scenario_name.upper()}")
    print(f"{'='*70}\n")
    
    # Define test variables
    print("1. Defining test variables...")
    Z1 = np.column_stack([dat['X'][:, 0], S[:, 0]])  # Shape: (n, 2)
    X1 = S[:, p].reshape(-1, 1)  # Shape: (n, 1)
    Y1 = (dat['X'][:, p] - X1.flatten()).reshape(-1, 1)  # Shape: (n, 1)
    
    print(f"   Z1 shape: {Z1.shape}")
    print(f"   X1 shape: {X1.shape}")
    print(f"   Y1 shape: {Y1.shape}")
    print(f"   Correlation(X1, Y1): {np.corrcoef(X1.flatten(), Y1.flatten())[0, 1]:.4f}")
    
    # Combine into single dataset: [Z1, X1, Y1]
    train_data = np.column_stack([Z1, X1, Y1])  # Shape: (n, 4)
    
    # Define groups
    d_in = 4
    groups = torch.zeros(d_in, dtype=torch.long)
    groups[0:2] = 0  # Z1
    groups[2] = 1    # X1
    groups[3] = 2    # Y1
    
    # Adjacency matrix for conditional independence testing
    A = np.array([
        [0, 1, 1],  # G0 (Z1) → G1 (X1), G0 → G2 (Y1)
        [0, 0, 0],  # G1 (X1) has no children (no X1 → Y1)
        [0, 0, 0]   # G2 (Y1) has no children
    ], dtype=np.float32)
    
    # Save training data and SVD results
    scenario_data_dir = os.path.join(output_dir, 'data', f'Assumption_Check_{scenario_name}')
    os.makedirs(scenario_data_dir, exist_ok=True)
    train_df = pd.DataFrame(train_data, columns=['Z1_1', 'Z1_2', 'X1', 'Y1'])
    train_df.to_csv(os.path.join(scenario_data_dir, 'train_data.csv'), index=False)
    
    # Save ALL SVD singular values for this scenario
    svd_values_df = pd.DataFrame({
        'component': range(1, len(S_result['D_all']) + 1),
        'singular_value': S_result['D_all']
    })
    svd_values_df.to_csv(os.path.join(scenario_data_dir, 'singular_values.csv'), index=False)
    
    print(f"\n2. Training data saved to {scenario_data_dir}/train_data.csv")
    print(f"   All SVD singular values ({len(S_result['D_all'])}) saved to {scenario_data_dir}/singular_values.csv")
    
    # Setup checkpoint directory
    scenario_ckpt_dir = os.path.join(output_dir, 'ckpt', f'Assumption_Check_{scenario_name}')
    os.makedirs(scenario_ckpt_dir, exist_ok=True)
    
    # Training with Energy validation
    print(f"\n3. Training conditional diffusion models with Energy validation...")
    print(f"   Energy test α = {alpha}")
    print(f"   Max seeds to try: {max_seeds}")
    print(f"   Device: {device}")
    print(f"   Epochs: {n_epochs}")
    print(f"   Learning rate: {lr}")
    
    failed_checkpoints = []
    
    for train_seed in range(max_seeds):
        print(f"\n{'-'*60}")
        print(f"{scenario_name.upper()} - Seed {train_seed}")
        print(f"{'-'*60}")
        
        # Set seed
        seed_everything(train_seed)
        
        # Normalize data
        print(f"Normalizing data (seed={train_seed})...")
        if use_standard_scaler:
            print("   Using StandardScaler")
            scaler = StandardScaler()
            train_data_norm = scaler.fit_transform(train_data)
        else:
            n_quantiles = min(1000, n)
            print(f"   Using QuantileTransformer with {n_quantiles} quantiles")
            qt = QuantileTransformer(output_distribution="normal", random_state=train_seed, 
                                    n_quantiles=n_quantiles,
                                    subsample=min(100000, n))
            train_data_norm = qt.fit_transform(train_data)
        
        train_data_norm_tensor = torch.tensor(train_data_norm, dtype=torch.float32)
        
        # Train models
        print(f"Training models...")
        models = train_all_conditional_ddpms(
            train_data=train_data_norm_tensor,
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
        checkpoint_path = os.path.join(scenario_ckpt_dir, f'models_seed{train_seed}.pkl')
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
        )  # Shape: [n, 1, 4]
        
        generated_norm = generated_norm.squeeze(1).cpu().detach().numpy()  # [n, 4]
        
        # Inverse transform
        scaler_obj = scaler if use_standard_scaler else qt
        generated = scaler_obj.inverse_transform(generated_norm)
        
        # Test 1: (Z1, X1) - columns [0, 1, 2]
        Z1_X1_orig = train_data[:, [0, 1, 2]]
        Z1_X1_gen = generated[:, [0, 1, 2]]
        passed_Z1_X1, p_val_Z1_X1 = run_energy_test(Z1_X1_orig, Z1_X1_gen, alpha)
        status_Z1_X1 = "PASS" if passed_Z1_X1 else "FAIL"
        print(f"  Energy test (Z1, X1): p-value = {p_val_Z1_X1:.4f} [{status_Z1_X1}]")
        
        # Test 2: (Z1, Y1) - columns [0, 1, 3]
        Z1_Y1_orig = train_data[:, [0, 1, 3]]
        Z1_Y1_gen = generated[:, [0, 1, 3]]
        passed_Z1_Y1, p_val_Z1_Y1 = run_energy_test(Z1_Y1_orig, Z1_Y1_gen, alpha)
        status_Z1_Y1 = "PASS" if passed_Z1_Y1 else "FAIL"
        print(f"  Energy test (Z1, Y1): p-value = {p_val_Z1_Y1:.4f} [{status_Z1_Y1}]")
        
        if passed_Z1_X1 and passed_Z1_Y1:
            print(f"\n{'='*60}")
            print(f"SUCCESS! {scenario_name.upper()} found good model at seed {train_seed}")
            print(f"{'='*60}")
            
            # Save as best model
            best_model_path = os.path.join(scenario_ckpt_dir, 'models.pkl')
            best_scaler_path = os.path.join(scenario_data_dir, 'scaler.pkl')
            best_meta_path = os.path.join(scenario_ckpt_dir, 'metadata.pkl')
            
            with open(best_model_path, 'wb') as f:
                pickle.dump(models, f)
            with open(best_scaler_path, 'wb') as f:
                pickle.dump(scaler_obj, f)
            with open(best_meta_path, 'wb') as f:
                pickle.dump({
                    'seed': train_seed,
                    'groups': groups,
                    'A': A,
                    'hidden_dims': hidden_dims,
                    'dim_t': dim_t,
                    'n_steps': n_steps,
                    'use_standard_scaler': use_standard_scaler
                }, f)
            
            # Delete checkpoints
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            for failed_path in failed_checkpoints:
                if os.path.exists(failed_path):
                    os.remove(failed_path)
            
            print(f"Saved: models.pkl, scaler.pkl, metadata.pkl")
            
            # Store scaler and models for MCODEC test
            final_scaler = scaler_obj
            final_models = models
            final_train_data_norm = train_data_norm
            break
        else:
            failed_checkpoints.append(checkpoint_path)
            print(f"Seed {train_seed} failed Energy tests, trying next...")
    else:
        # All seeds failed
        print(f"\n{'='*60}")
        print(f"FAILED! {scenario_name.upper()} - no good model found after {max_seeds} seeds")
        print(f"{'='*60}")
        
        # Cleanup
        for failed_path in failed_checkpoints:
            if os.path.exists(failed_path):
                os.remove(failed_path)
        
        return None
    
    print(f"\nModels saved to: {scenario_ckpt_dir}")
    
    # Conditional Independence Test
    print(f"\n4. Performing conditional independence test...")
    print(f"   Testing: X1 ⊥ Y1 | Z1")
    
    # Extract variables
    Z1_data = train_data[:, 0:2]
    X1_data = train_data[:, 2:3]
    Y1_data = train_data[:, 3:4]
    
    # Compute τ_obs
    print(f"\n   Computing τ_obs = MCODEC(Y1_observed, X1, Z1)...")
    tau_observed = mcodec(Y1_data, X1_data, Z1_data)
    print(f"   τ_obs = {tau_observed:.6f}")
    
    # Sample Y1 D times and compute τ^(d)
    print(f"\n   Sampling Y1 {D} times from conditional model...")
    tau_samples = []
    
    mc_batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])
    
    for batch_idx, mc_batch_size in enumerate(mc_batch_sizes):
        print(f"   Batch {batch_idx+1}/{len(mc_batch_sizes)}: Generating {mc_batch_size} samples...")
        
        # Prepare do_vars
        do_vars = {}
        do_vars[0] = final_train_data_norm[:, 0]  # Z1[0]
        do_vars[1] = final_train_data_norm[:, 1]  # Z1[1]
        do_vars[2] = final_train_data_norm[:, 2]  # X1
        
        sample_vars = [3]
        
        # Generate conditional samples
        Y1_sampled_batch_norm = generate_conditional_samples(
            models=final_models,
            groups=groups,
            A=A,
            do_vars=do_vars,
            sample_vars=sample_vars,
            n_samples=mc_batch_size
        )
        
        Y1_sampled_batch_norm = Y1_sampled_batch_norm.cpu().numpy()
        
        # Transform back and compute MCODEC
        for i in range(mc_batch_size):
            full_data_norm = np.zeros((n, 4))
            full_data_norm[:, 0:2] = final_train_data_norm[:, 0:2]
            full_data_norm[:, 2] = final_train_data_norm[:, 2]
            full_data_norm[:, 3] = Y1_sampled_batch_norm[i, :, 0]
            
            # Inverse transform
            full_data = final_scaler.inverse_transform(full_data_norm)
            
            Y1_sampled = full_data[:, 3:4]
            tau_d = mcodec(Y1_sampled, X1_data, Z1_data)
            tau_samples.append(tau_d)
    
    print(f"\n   Generated {len(tau_samples)} τ^(d) values")
    print(f"   τ^(d) range: [{min(tau_samples):.6f}, {max(tau_samples):.6f}]")
    print(f"   τ^(d) mean: {np.mean(tau_samples):.6f}")
    print(f"   τ^(d) std:  {np.std(tau_samples):.6f}")
    
    # Compute P-value
    num_larger_or_equal = sum(1 for tau_d in tau_samples if tau_d >= tau_observed)
    p_value = (1 + num_larger_or_equal) / (D + 1)
    
    print(f"\n6. Results:")
    print(f"   τ_obs:                       {tau_observed:.6f}")
    print(f"   # of τ^(d) ≥ τ_obs:          {num_larger_or_equal}")
    print(f"   P-value:                     {p_value:.6f}")
    print(f"\n   Interpretation at α=0.05:")
    if p_value <= 0.05:
        print(f"   → REJECT H0: Evidence against X1 ⊥ Y1 | Z1")
    else:
        print(f"   → FAIL TO REJECT H0: No evidence against X1 ⊥ Y1 | Z1")
    
    # Save results
    scenario_results_dir = os.path.join(output_dir, 'results', f'Assumption_Check_{scenario_name}')
    os.makedirs(scenario_results_dir, exist_ok=True)
    
    # Save tau samples as CSV
    tau_samples_df = pd.DataFrame({
        'sample_idx': range(1, len(tau_samples) + 1),
        'tau_value': tau_samples
    })
    tau_samples_df.to_csv(os.path.join(scenario_results_dir, 'tau_samples.csv'), index=False)
    
    # Save tau_observed separately
    tau_observed_df = pd.DataFrame({
        'tau_observed': [tau_observed]
    })
    tau_observed_df.to_csv(os.path.join(scenario_results_dir, 'tau_observed.csv'), index=False)
    
    print(f"\n7. Results saved to {scenario_results_dir}/")
    print(f"   - tau_samples.csv ({len(tau_samples)} samples)")
    print(f"   - tau_observed.csv")
    
    # Return summary for final display (not saved to file)
    return {
        'scenario': scenario_name,
        'tau_observed': tau_observed,
        'p_value': p_value,
        'num_larger_or_equal': num_larger_or_equal,
        'D': D
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='SVD-based Conditional Independence Testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation parameters
    parser.add_argument('--n', type=int, default=500, help='Number of samples')
    parser.add_argument('--p', type=int, default=20, help='Number of variable pairs')
    parser.add_argument('--range_H', type=float, default=2.0, help='Range for H distribution')
    parser.add_argument('--eps_sd', type=float, default=1.0, help='Noise standard deviation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    
    # SVD parameters
    parser.add_argument('--svd_rank', type=int, default=2, help='SVD rank for approximation')
    
    # Model parameters
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 128, 64],
                       help='Hidden dimensions for MLP')
    parser.add_argument('--dim_t', type=int, default=32, help='Time embedding dimension')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--n_epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--use_standard_scaler', action='store_true',
                       help='Use StandardScaler instead of QuantileTransformer')
    
    # Inference parameters
    parser.add_argument('--D', type=int, default=500, help='Number of Monte Carlo samples')
    parser.add_argument('--D_batch', type=int, default=250, help='Batch size for MC sampling')
    
    # Validation parameters
    parser.add_argument('--alpha', type=float, default=0.2,
                       help='Significance level for Energy tests')
    parser.add_argument('--max_seeds', type=int, default=11,
                       help='Maximum seeds to try (0 to max_seeds-1)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda:0, cpu). If None, auto-detect.')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=script_dir,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("ASSUMPTION CHECK: SVD-BASED CONDITIONAL INDEPENDENCE TESTING")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Set device
    if args.device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\nUsing device: {device}")
    
    if device.startswith('cuda') and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Set random seed
    seed_everything(args.seed)
    print(f"\nRandom seed set to: {args.seed}")
    
    # Generate data for both scenarios
    print(f"\n{'='*70}")
    print("DATA GENERATION")
    print(f"{'='*70}\n")
    
    print("Generating Scenario 1 (sc1)...")
    dat1 = generate_X_two_scenarios(
        n=args.n, p=args.p, range_H=args.range_H,
        scenario='sc1', eps_sd=args.eps_sd, seed=args.seed
    )
    print(f"  Data shape: {dat1['X'].shape}")
    
    print("\nGenerating Scenario 2 (sc2)...")
    dat2 = generate_X_two_scenarios(
        n=args.n, p=args.p, range_H=args.range_H,
        scenario='sc2', eps_sd=args.eps_sd, seed=args.seed
    )
    print(f"  Data shape: {dat2['X'].shape}")
    
    print(f"\nH statistics:")
    print(f"  Range: [{dat1['H'].min():.3f}, {dat1['H'].max():.3f}]")
    print(f"  Mean: {dat1['H'].mean():.3f}, Std: {dat1['H'].std():.3f}")
    
    # Compute SVD approximations
    print(f"\n{'='*70}")
    print("SVD APPROXIMATIONS")
    print(f"{'='*70}\n")
    
    print("Computing SVD for Scenario 1...")
    S1_result = svd_rank_r_approx(dat1['X'], r=args.svd_rank, center=False, scale=False)
    S1 = S1_result['S']
    print(f"  S1 shape: {S1.shape}")
    print(f"  Top {args.svd_rank} singular values: {S1_result['D']}")
    
    print("\nComputing SVD for Scenario 2...")
    S2_result = svd_rank_r_approx(dat2['X'], r=args.svd_rank, center=False, scale=False)
    S2 = S2_result['S']
    print(f"  S2 shape: {S2.shape}")
    print(f"  Top {args.svd_rank} singular values: {S2_result['D']}")
    
    # Save SVD results
    svd_results_dir = os.path.join(args.output_dir, 'results', 'Assumption_Check')
    os.makedirs(svd_results_dir, exist_ok=True)
    
    # Save ALL singular values for both scenarios
    svd_df = pd.DataFrame({
        'scenario': ['sc1'] * len(S1_result['D_all']) + ['sc2'] * len(S2_result['D_all']),
        'component': list(range(1, len(S1_result['D_all']) + 1)) + list(range(1, len(S2_result['D_all']) + 1)),
        'singular_value': np.concatenate([S1_result['D_all'], S2_result['D_all']])
    })
    svd_path = os.path.join(svd_results_dir, 'singular_values.csv')
    svd_df.to_csv(svd_path, index=False)
    print(f"\nAll singular values saved to: {svd_path}")
    
    # Save configuration
    config_dir = os.path.join(args.output_dir, 'results', 'Assumption_Check')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Process both scenarios
    summary_sc1 = process_scenario(
        scenario_name='sc1',
        dat=dat1,
        S=S1,
        S_result=S1_result,
        p=args.p,
        n=args.n,
        seed=args.seed,
        hidden_dims=args.hidden_dims,
        dim_t=args.dim_t,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        D=args.D,
        D_batch=args.D_batch,
        use_standard_scaler=args.use_standard_scaler,
        output_dir=args.output_dir,
        alpha=args.alpha,
        max_seeds=args.max_seeds
    )
    
    if summary_sc1 is None:
        print(f"\n{'='*70}")
        print("STOPPING: Scenario 1 failed to train a good model")
        print(f"{'='*70}")
        print("\nPlease consider:")
        print("  1. Increasing n_epochs")
        print("  2. Adjusting learning rate")
        print("  3. Increasing model capacity (hidden_dims, dim_t)")
        print("  4. Relaxing α (currently {})".format(args.alpha))
        print("  5. Checking data quality")
        return
    
    summary_sc2 = process_scenario(
        scenario_name='sc2',
        dat=dat2,
        S=S2,
        S_result=S2_result,
        p=args.p,
        n=args.n,
        seed=args.seed,
        hidden_dims=args.hidden_dims,
        dim_t=args.dim_t,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        D=args.D,
        D_batch=args.D_batch,
        use_standard_scaler=args.use_standard_scaler,
        output_dir=args.output_dir,
        alpha=args.alpha,
        max_seeds=args.max_seeds
    )
    
    if summary_sc2 is None:
        print(f"\n{'='*70}")
        print("STOPPING: Scenario 2 failed to train a good model")
        print(f"{'='*70}")
        print("\nPlease consider:")
        print("  1. Increasing n_epochs")
        print("  2. Adjusting learning rate")
        print("  3. Increasing model capacity (hidden_dims, dim_t)")
        print("  4. Relaxing α (currently {})".format(args.alpha))
        print("  5. Checking data quality")
        return
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Scenario 1:")
    print(f"  τ_obs:   {summary_sc1['tau_observed']:.6f}")
    print(f"  P-value: {summary_sc1['p_value']:.6f}")
    print(f"  Decision: {'REJECT H0' if summary_sc1['p_value'] <= 0.05 else 'FAIL TO REJECT H0'}")
    
    print(f"\nScenario 2:")
    print(f"  τ_obs:   {summary_sc2['tau_observed']:.6f}")
    print(f"  P-value: {summary_sc2['p_value']:.6f}")
    print(f"  Decision: {'REJECT H0' if summary_sc2['p_value'] <= 0.05 else 'FAIL TO REJECT H0'}")
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"  Data:    {args.output_dir}/data/Assumption_Check_{{sc1,sc2}}/")
    print(f"  Models:  {args.output_dir}/ckpt/Assumption_Check_{{sc1,sc2}}/")
    print(f"  Results: {args.output_dir}/results/Assumption_Check_{{sc1,sc2}}/")


if __name__ == '__main__':
    main()
