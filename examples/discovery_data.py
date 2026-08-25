#!/usr/bin/env python
"""
Discovery Data Generation

Generates synthetic data with three variables for causal discovery experiments:
- X1 ~ N(0, 1)
- X2 = X1^2 + epsilon2, where epsilon2 ~ t(df=1)
- X3 = cos(X2 - 1) + exp(X1) - 2 + tanh(X2 - X1) + epsilon3, where epsilon3 ~ N(0, 1)

Usage:
    python discovery_data.py --n 500 --seed 42
"""

import numpy as np
import pandas as pd
import argparse
import os


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)


def generate_discovery_data(n, seed=None):
    """
    Generate discovery dataset.
    
    Parameters:
    -----------
    n : int
        Number of samples
    seed : int or None
        Random seed for reproducibility
    
    Returns:
    --------
    ndarray of shape (n, 3) with columns [X1, X2, X3]
    """
    if seed is not None:
        seed_everything(seed)
    
    # X1 ~ N(0, 1) - standard normal
    X1 = np.random.normal(0, 1, size=n)
    
    # X2 = X1^2 + epsilon2, where epsilon2 ~ t(df=1) (Cauchy distribution)
    epsilon2 = np.random.standard_t(df=1, size=n)
    X2 = X1**2 + epsilon2
    
    # X3 = cos(X2 - 1) + exp(X1) - 2 + tanh(X2 - X1) + epsilon3
    # where epsilon3 ~ N(0, 1)
    epsilon3 = np.random.normal(0, 1, size=n)
    X3 = np.cos(X2 - 1) + np.exp(X1) - 2 + np.tanh(X2 - X1) + epsilon3
    
    # Combine into matrix
    data = np.column_stack([X1, X2, X3])
    
    return data


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description='Generate discovery dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--n', type=int, default=500, 
                       help='Number of samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default=script_dir,
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("DISCOVERY DATA GENERATION")
    print("="*70)
    print(f"\nParameters:")
    print(f"  n (samples):  {args.n}")
    print(f"  seed:         {args.seed}")
    
    # Generate data
    print(f"\nGenerating data...")
    data = generate_discovery_data(n=args.n, seed=args.seed)
    
    print(f"  X1: mean={data[:, 0].mean():.4f}, std={data[:, 0].std():.4f}")
    print(f"  X2: mean={data[:, 1].mean():.4f}, std={data[:, 1].std():.4f}")
    print(f"  X3: mean={data[:, 2].mean():.4f}, std={data[:, 2].std():.4f}")
    
    # Save data
    data_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    output_path = os.path.join(data_dir, 'discovery.csv')
    df = pd.DataFrame(data, columns=['X1', 'X2', 'X3'])
    df.to_csv(output_path, index=False)
    
    print(f"\nData saved to: {output_path}")
    print(f"Shape: {data.shape}")
    
    # Display correlations
    print(f"\nCorrelations:")
    corr_matrix = np.corrcoef(data.T)
    print(f"  Corr(X1, X2): {corr_matrix[0, 1]:.4f}")
    print(f"  Corr(X1, X3): {corr_matrix[0, 2]:.4f}")
    print(f"  Corr(X2, X3): {corr_matrix[1, 2]:.4f}")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
