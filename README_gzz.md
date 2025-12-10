# Gzz Grid Generation

Generate quantum gravity correlation (Gzz) grids from density grids using Ramsey pulse sequence simulation.

## What is Gzz?

Gzz represents the gravitational correlation measured via quantum sensing:
- Uses Ramsey interferometry pulse sequence (H-delay-H)
- Phase damping noise models gravitational effects
- Higher density → stronger phase damping → lower Gzz signal

## Quick Start

### 1. Install Qiskit

```bash
pip install qiskit qiskit-aer
```

Or in your virtual environment:

```bash
venv\Scripts\activate
pip install qiskit qiskit-aer
```

### 2. Generate Gzz Grids for Training Data

**Fast mode** (200 shots, ~30-60 seconds per grid):
```bash
python generate_gzz_grids.py --shots 200
```

**High quality** (1000 shots, ~2-5 minutes per grid):
```bash
python generate_gzz_grids.py --shots 1000
```

**Single sample** (for testing):
```bash
python generate_gzz_grids.py --sample 0 --shots 200
```

### 3. View in Web Demo

The web app automatically:
- Generates Gzz grids on-the-fly for random samples
- Loads pre-computed Gzz grids for uploaded training data
- Displays Gzz alongside density grids

```bash
python app.py
```

Open http://localhost:5000 - you'll see 4 panels:
1. **Density Grid** - Input
2. **Gzz Grid** - Quantum correlation
3. **Tunnel Probability** - ML prediction
4. **Detected Tunnels** - Binary result

## File Structure

After generation, `training_data/` will contain:

```
training_data/
├── density_grid_000.npy    # Input density
├── gzz_grid_000.npy        # Generated Gzz
├── tunnel_mask_000.npy     # Ground truth
├── metadata_000.json       # Labels
└── ...
```

## How It Works

### Physics Model

```python
# 1. Ramsey pulse sequence
H - delay(t) - H - measure

# 2. Phase damping from density
p(ρ, t) = 1 - exp(-ρ × t)

# 3. Gzz expectation value
Gzz = P(0) - P(1)
```

### Parameters

- **t_evolution**: Evolution time (default: 30 μs)
  - Higher = more sensitivity to density changes
  - Too high = saturates to noise
  
- **shots**: Quantum circuit repetitions (default: 500)
  - Higher = less noisy, slower
  - Lower = faster, more noise
  - 200 shots: good for demos
  - 1000+ shots: publication quality

## Performance

For 60×150 grids (9000 pixels):

| Shots | Time per Grid | Quality |
|-------|---------------|---------|
| 100   | ~15 seconds   | Noisy   |
| 200   | ~30 seconds   | Good    |
| 500   | ~1 minute     | Better  |
| 1000  | ~2 minutes    | Best    |
| 5000  | ~10 minutes   | Overkill|

With 500 training samples at 200 shots: **~4 hours total**

## Batch Processing Tips

### Process in Parallel

Split samples across multiple processes:

```bash
# Terminal 1: Process samples 0-99
python generate_gzz_grids.py --shots 200 &

# Terminal 2: Manually process 100-199 by modifying script
# etc.
```

### Resume Interrupted Runs

The script automatically skips existing Gzz grids, so you can safely re-run:

```bash
python generate_gzz_grids.py --shots 200
# Interrupted? Just run again - picks up where it left off
```

### Check Progress

```bash
# Count generated Gzz grids
ls training_data/gzz_grid_*.npy | wc -l

# Check a sample
python -c "import numpy as np; g = np.load('training_data/gzz_grid_000.npy'); print(g.shape, g.min(), g.max())"
```

## Interpretation

**Gzz Values:**
- **~1.0**: No damping (low density, void/tunnel)
- **~0.0**: Strong damping (high density, rock/ore)
- **~0.5**: Moderate damping (medium density)

**Expected patterns:**
- Tunnels: Gzz ≈ 0.8-1.0 (bright in viridis colormap)
- Rock: Gzz ≈ 0.2-0.5 (medium)
- Ore: Gzz ≈ 0.0-0.2 (dark)

## Troubleshooting

**"No module named 'qiskit'"**
```bash
pip install qiskit qiskit-aer
```

**Slow performance**
- Reduce `--shots` to 100-200
- Process fewer samples first
- Consider using approximation instead of full simulation

**Memory issues**
- Reduce shots
- Process one sample at a time with `--sample N`

**Gzz values all the same**
- Check t_evolution (try 10e-6 to 100e-6)
- Verify density grid has variation
- Increase shots for less noise

## Advanced: Approximation Mode

For very fast generation without Qiskit simulation:

```python
# Simple approximation (no quantum simulation)
def approx_gzz(density, t=30e-6):
    return np.exp(-density * t)

gzz_grid = approx_gzz(density_grid)
```

This is **much faster** but less physically accurate. Good for:
- Rapid prototyping
- Testing visualization
- Cases where exact quantum simulation isn't needed

## Citation

If you use this Gzz generation method:

```
Quantum gravity sensing via Ramsey interferometry with phase damping
noise model. Evolution time: 30 μs, Qiskit Aer simulator.
```
