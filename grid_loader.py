import numpy as np
import matplotlib.pyplot as plt

# Load a single grid
grid = np.load("training_data/density_grid_000.npy")

plt.figure(figsize=(7, 3))
plt.imshow(grid, origin='upper', cmap='inferno')
plt.colorbar(label="density")
plt.xlabel("x index")
plt.ylabel("z index (depth)")
plt.title("Density Grid")
plt.show()

# Load all grids
grids = [np.load(f"training_data/density_grid_{i:03d}.npy") for i in range(50)]