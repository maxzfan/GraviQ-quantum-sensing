"""
Fast Gzz grid generation using analytical approximation instead of full quantum simulation.
This is ~1000x faster than Qiskit simulation.
"""
import numpy as np
import os
from tqdm import tqdm


def gzz_approximation(density_grid, t_evolution=30e-6):
    """
    Fast analytical approximation of Gzz without quantum simulation.
    
    Based on the expected behavior:
    Gzz ≈ exp(-ρ * t) * cos(some phase)
    
    For Ramsey with phase damping:
    Gzz ≈ exp(-ρ * t)
    
    This is orders of magnitude faster than Qiskit simulation.
    
    Args:
        density_grid: (N, M) density values
        t_evolution: Evolution time
    
    Returns:
        gzz_grid: (N, M) Gzz values
    """
    # Simple exponential decay model
    gzz_grid = np.exp(-density_grid * t_evolution * 1e6)  # Scale for realistic values
    
    # Add small random noise to simulate quantum measurement uncertainty
    noise = np.random.normal(0, 0.02, gzz_grid.shape)
    gzz_grid = np.clip(gzz_grid + noise, -1, 1)
    
    return gzz_grid


def process_training_data_fast(data_dir='training_data', t_evolution=30e-6):
    """
    Generate Gzz grids for all density grids using fast approximation.
    
    Args:
        data_dir: Directory containing density_grid_*.npy files
        t_evolution: Evolution time for approximation
    """
    # Find all density grid files
    density_files = sorted([f for f in os.listdir(data_dir) 
                           if f.startswith('density_grid_') and f.endswith('.npy')])
    
    print(f"Found {len(density_files)} density grids in {data_dir}/")
    print(f"Using FAST approximation mode (no Qiskit simulation)")
    print(f"Settings: t_evolution={t_evolution:.2e}s")
    print(f"Estimated time: ~{len(density_files) * 0.1:.1f} seconds")
    print()
    
    for i, filename in enumerate(tqdm(density_files, desc="Processing grids")):
        sample_id = filename.replace('density_grid_', '').replace('.npy', '')
        density_path = os.path.join(data_dir, filename)
        gzz_path = os.path.join(data_dir, f'gzz_grid_{sample_id}.npy')
        
        # Skip if already exists
        if os.path.exists(gzz_path):
            continue
        
        # Load density grid
        density_grid = np.load(density_path)
        
        # Fast approximation
        gzz_grid = gzz_approximation(density_grid, t_evolution)
        
        # Save Gzz grid
        np.save(gzz_path, gzz_grid)
    
    print(f"\nAll Gzz grids generated using fast approximation!")
    print(f"Saved to {data_dir}/gzz_grid_*.npy")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast Gzz grid generation (approximation)')
    parser.add_argument('--data_dir', default='training_data', help='Directory with density grids')
    parser.add_argument('--t_evolution', type=float, default=30e-6, help='Evolution time (seconds)')
    parser.add_argument('--sample', type=int, default=None, help='Process single sample ID only')
    
    args = parser.parse_args()
    
    if args.sample is not None:
        # Process single sample
        sample_id = f"{args.sample:03d}"
        density_path = os.path.join(args.data_dir, f'density_grid_{sample_id}.npy')
        gzz_path = os.path.join(args.data_dir, f'gzz_grid_{sample_id}.npy')
        
        print(f"Processing single sample {sample_id} (FAST mode)...")
        density_grid = np.load(density_path)
        gzz_grid = gzz_approximation(density_grid, args.t_evolution)
        np.save(gzz_path, gzz_grid)
        print(f"Saved to {gzz_path}")
        print(f"Gzz range: [{gzz_grid.min():.4f}, {gzz_grid.max():.4f}]")
    else:
        # Process all samples
        process_training_data_fast(args.data_dir, args.t_evolution)
