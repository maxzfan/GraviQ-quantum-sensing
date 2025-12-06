import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt

N_x = 150
N_z = 60

rho_rock = 1.0
rho_void = 0.0
rho_ore  = 2.5


def draw_tunnel(grid, start_x, start_z, length, thickness, curvature=0.0, rng=None):
    """
    Draw a tunnel as a continuous void path.
    curvature >0 bends right; <0 bends left.
    """
    x, z = start_x, start_z

    for step in range(length):
        # random walk in depth
        z += rng.randint(-1, 2)
        z = np.clip(z, 0, N_z-1)

        # curvature makes tunnel drift horizontally
        x += curvature + rng.uniform(-0.5, 0.5)
        x = int(np.clip(x, 0, N_x-1))

        # carve out tunnel
        for dz in range(-thickness, thickness+1):
            for dx in range(-thickness*2, thickness*2+1):
                zz = np.clip(z+dz, 0, N_z-1)
                xx = np.clip(x+dx, 0, N_x-1)
                grid[zz, xx] = rho_void


def make_grid(seed):
    rng = np.random.RandomState(seed)
    grid = np.full((N_z, N_x), rho_rock)

    # --- add ore blobs ---
    for _ in range(5):
        cx = rng.randint(0, N_x)
        cz = rng.randint(int(N_z*0.2), N_z)
        w = rng.randint(5, 20)
        h = rng.randint(3, 10)
        grid[cz:cz+h, max(0, cx-w//2):min(N_x, cx+w//2)] = rho_ore

    # --- add void pockets (caves) ---
    for _ in range(3):
        cx = rng.randint(0, N_x)
        cz = rng.randint(0, N_z)
        w = rng.randint(8, 20)
        h = rng.randint(3, 7)
        grid[cz:cz+h, max(0, cx-w//2):min(N_x, cx+w//2)] = rho_void

    # --- add tunnels ---
    num_tunnels = rng.randint(1, 3)
    for _ in range(num_tunnels):
        start_x = rng.randint(0, N_x)
        start_z = rng.randint(5, N_z//2)
        length   = rng.randint(40, 120)
        thickness = rng.randint(1, 3)
        curvature = rng.uniform(-0.3, 0.3)
        draw_tunnel(grid, start_x, start_z, length, thickness, curvature, rng)

    return grid


import os

# --- Generate and save 50 training maps ---
output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)

num_samples = 50

print(f"Generating {num_samples} density grids...")

for i in range(num_samples):
    grid = make_grid(seed=i)
    
    # Save as numpy array
    np.save(os.path.join(output_dir, f"density_grid_{i:03d}.npy"), grid)
    
    # Save visualization as PNG
    plt.figure(figsize=(7, 3))
    plt.imshow(grid, origin='upper', cmap='inferno')
    plt.colorbar(label="density")
    plt.title(f"Density Grid #{i:03d}")
    plt.xlabel("x index")
    plt.ylabel("z index (depth)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"density_grid_{i:03d}.png"), dpi=100)
    plt.close()
    
    if (i + 1) % 10 == 0:
        print(f"  Generated {i + 1}/{num_samples}")

print(f"Done! Saved {num_samples} grids to '{output_dir}/'")
print(f"  - .npy files: raw numpy arrays ({N_z}x{N_x})")
print(f"  - .png files: visualizations")
