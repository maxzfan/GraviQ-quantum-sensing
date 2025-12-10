import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader


class TunnelDataset(Dataset):
    """
    Dataset for tunnel detection/segmentation from density grids.
    
    Returns:
        - density_grid: (1, H, W) tensor - input
        - tunnel_mask: (1, H, W) tensor - ground truth segmentation
        - metadata: dict with labels
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing density_grid_*.npy, tunnel_mask_*.npy, metadata_*.json
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Find all density grid files
        self.samples = []
        for filename in sorted(os.listdir(data_dir)):
            if filename.startswith('density_grid_') and filename.endswith('.npy'):
                sample_id = filename.replace('density_grid_', '').replace('.npy', '')
                self.samples.append(sample_id)
        
        print(f"Found {len(self.samples)} samples in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        
        # Load density grid (input)
        density_path = os.path.join(self.data_dir, f'density_grid_{sample_id}.npy')
        density_grid = np.load(density_path).astype(np.float32)
        
        # Load tunnel mask (ground truth)
        mask_path = os.path.join(self.data_dir, f'tunnel_mask_{sample_id}.npy')
        tunnel_mask = np.load(mask_path).astype(np.float32)
        
        # Load metadata
        metadata_path = os.path.join(self.data_dir, f'metadata_{sample_id}.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        density_grid = density_grid[np.newaxis, ...]
        tunnel_mask = tunnel_mask[np.newaxis, ...]
        
        # Convert to tensors
        density_grid = torch.from_numpy(density_grid)
        tunnel_mask = torch.from_numpy(tunnel_mask)
        
        if self.transform:
            density_grid, tunnel_mask = self.transform(density_grid, tunnel_mask)
        
        return {
            'input': density_grid,
            'mask': tunnel_mask,
            'has_tunnel': metadata['has_tunnel'],
            'num_tunnels': metadata['num_tunnels'],
            'sample_id': sample_id
        }


def get_dataloaders(data_dir, batch_size=8, train_split=0.8, num_workers=0):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory with training data
        batch_size: Batch size
        train_split: Fraction of data for training (rest for validation)
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    full_dataset = TunnelDataset(data_dir)
    
    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = TunnelDataset("training_data")
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Check first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Input shape: {sample['input'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Has tunnel: {sample['has_tunnel']}")
    print(f"  Num tunnels: {sample['num_tunnels']}")
    
    # Test dataloader
    train_loader, val_loader = get_dataloaders("training_data", batch_size=4)
    
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Input: {batch['input'].shape}")
    print(f"  Mask: {batch['mask'].shape}")
    print(f"  Has tunnel: {batch['has_tunnel']}")
