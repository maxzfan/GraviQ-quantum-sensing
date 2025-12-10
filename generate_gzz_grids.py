"""
Generate Gzz (quantum gravity correlation) grids for all density grids in training_data.
Uses Ramsey pulse sequence simulation with phase damping.
"""
import numpy as np
import os
from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
try:
    # Try newer Qiskit Aer API
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, phase_damping_error
except ImportError:
    # Fall back to older API
    from qiskit.providers.aer import AerSimulator
    from qiskit.providers.aer.noise import NoiseModel
    from qiskit.providers.aer.noise.errors import phase_damping_error


def ramsey_circuit(t):
    """Create Ramsey pulse sequence circuit"""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.delay(int(t * 1e9), 0, unit="ns")
    qc.h(0)
    qc.measure(0, 0)
    return qc


def damping_probability(rho, t):
    """Convert density to phase damping probability"""
    return 1 - np.exp(-rho * t)


def compute_Gzz_at(rho_value, t, backend, shots=1000):
    """
    Compute Gzz for a single grid point using Qiskit simulation.
    Reduced shots for faster computation.
    """
    # Build noise channel
    p = damping_probability(rho_value, t)
    p = np.clip(p, 0, 1)  # Ensure valid probability
    dephasing = phase_damping_error(p)
    
    # Create noise model
    noise_model = NoiseModel()
    # Add error to qubit 0 for id and delay gates
    noise_model.add_quantum_error(dephasing, ["id", "delay"], [0])
    
    # Build circuit
    qc = ramsey_circuit(t)
    qc = transpile(qc, backend)
    
    # Run simulation
    result = backend.run(qc, noise_model=noise_model, shots=shots).result()
    counts = result.get_counts()
    
    # Compute expectation value <Z> = P(0) - P(1)
    P0 = counts.get('0', 0) / shots
    P1 = counts.get('1', 0) / shots
    Gzz = P0 - P1
    
    return Gzz


def density_to_gzz_grid(density_grid, t_evolution=30e-6, shots=500):
    """
    Convert entire density grid to Gzz grid using Qiskit simulation.
    
    Args:
        density_grid: (N, M) array of density values
        t_evolution: Evolution time in seconds
        shots: Number of quantum shots per pixel (lower = faster but noisier)
    
    Returns:
        Gzz_grid: (N, M) array of Gzz values
    """
    N, M = density_grid.shape
    Gzz_grid = np.zeros((N, M))
    
    # Create backend once
    backend = AerSimulator()
    
    # Progress bar for entire grid
    total_pixels = N * M
    with tqdm(total=total_pixels, desc="Computing Gzz", unit="px") as pbar:
        for i in range(N):
            for j in range(M):
                rho = density_grid[i, j]
                Gzz_grid[i, j] = compute_Gzz_at(rho, t_evolution, backend, shots)
                pbar.update(1)
    
    return Gzz_grid


def process_training_data(data_dir='training_data', t_evolution=30e-6, shots=500):
    """
    Generate Gzz grids for all density grids in training_data directory.
    
    Args:
        data_dir: Directory containing density_grid_*.npy files
        t_evolution: Evolution time for Ramsey sequence
        shots: Quantum shots per pixel
    """
    # Find all density grid files
    density_files = sorted([f for f in os.listdir(data_dir) if f.startswith('density_grid_') and f.endswith('.npy')])
    
    print(f"Found {len(density_files)} density grids in {data_dir}/")
    print(f"Settings: t_evolution={t_evolution:.2e}s, shots={shots}")
    print(f"Estimated time: ~{len(density_files) * 60 * 150 * shots / 10000:.1f} seconds")
    print()
    
    for i, filename in enumerate(density_files):
        sample_id = filename.replace('density_grid_', '').replace('.npy', '')
        density_path = os.path.join(data_dir, filename)
        gzz_path = os.path.join(data_dir, f'gzz_grid_{sample_id}.npy')
        
        # Skip if already exists
        if os.path.exists(gzz_path):
            print(f"[{i+1}/{len(density_files)}] Skipping {sample_id} (already exists)")
            continue
        
        print(f"[{i+1}/{len(density_files)}] Processing sample {sample_id}...")
        
        # Load density grid
        density_grid = np.load(density_path)
        
        # Convert to Gzz
        gzz_grid = density_to_gzz_grid(density_grid, t_evolution, shots)
        
        # Save Gzz grid
        np.save(gzz_path, gzz_grid)
        print(f"  Saved to {gzz_path}")
        print(f"  Gzz range: [{gzz_grid.min():.4f}, {gzz_grid.max():.4f}]")
        print()
    
    print("All Gzz grids generated!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Gzz grids from density grids')
    parser.add_argument('--data_dir', default='training_data', help='Directory with density grids')
    parser.add_argument('--t_evolution', type=float, default=30e-6, help='Evolution time (seconds)')
    parser.add_argument('--shots', type=int, default=500, help='Quantum shots per pixel (lower=faster)')
    parser.add_argument('--sample', type=int, default=None, help='Process single sample ID only')
    
    args = parser.parse_args()
    
    if args.sample is not None:
        # Process single sample
        sample_id = f"{args.sample:03d}"
        density_path = os.path.join(args.data_dir, f'density_grid_{sample_id}.npy')
        gzz_path = os.path.join(args.data_dir, f'gzz_grid_{sample_id}.npy')
        
        print(f"Processing single sample {sample_id}...")
        density_grid = np.load(density_path)
        gzz_grid = density_to_gzz_grid(density_grid, args.t_evolution, args.shots)
        np.save(gzz_path, gzz_grid)
        print(f"Saved to {gzz_path}")
    else:
        # Process all samples
        process_training_data(args.data_dir, args.t_evolution, args.shots)
