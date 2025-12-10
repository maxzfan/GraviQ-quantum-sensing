import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_dataloaders

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("WARNING: TensorBoard not available. Training will proceed without logging.")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combination of BCE and Dice loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate IoU, Dice, Precision, Recall"""
    # Apply sigmoid and threshold
    preds = (torch.sigmoid(predictions) > threshold).float()
    
    # Flatten
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Calculate metrics
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (preds.sum() + targets.sum() + 1e-7)
    
    # Precision and Recall
    true_positive = intersection
    predicted_positive = preds.sum()
    actual_positive = targets.sum()
    
    precision = (true_positive + 1e-7) / (predicted_positive + 1e-7)
    recall = (true_positive + 1e-7) / (actual_positive + 1e-7)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0
    epoch_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        metrics = calculate_metrics(outputs, targets)
        
        # Update running stats
        epoch_loss += loss.item()
        for key in epoch_metrics:
            epoch_metrics[key] += metrics[key]
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'dice': metrics['dice']})
    
    # Average over batches
    num_batches = len(train_loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    
    epoch_loss = 0
    epoch_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch in pbar:
            inputs = batch['input'].to(device)
            targets = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, targets)
            
            # Update running stats
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'dice': metrics['dice']})
    
    # Average over batches
    num_batches = len(val_loader)
    epoch_loss /= num_batches
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches
    
    return epoch_loss, epoch_metrics


def save_predictions(model, val_loader, device, save_dir, num_samples=5):
    """Save sample predictions for visualization"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    batch = next(iter(val_loader))
    inputs = batch['input'].to(device)
    targets = batch['mask'].to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.sigmoid(outputs)
    
    # Move to CPU and convert to numpy
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = preds.cpu().numpy()
    
    for i in range(min(num_samples, inputs.shape[0])):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Input density grid
        axes[0].imshow(inputs[i, 0], cmap='inferno', origin='upper')
        axes[0].set_title('Input Density Grid')
        axes[0].axis('off')
        
        # Ground truth
        axes[1].imshow(targets[i, 0], cmap='binary', origin='upper', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(preds[i, 0], cmap='binary', origin='upper', vmin=0, vmax=1)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'), dpi=100)
        plt.close()


def train(
    data_dir='training_data',
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='checkpoints',
    log_dir='runs'
):
    """Main training function"""
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard (if available)
    writer = SummaryWriter(log_dir) if TENSORBOARD_AVAILABLE else None
    if writer:
        print(f"TensorBoard logging enabled. Run: tensorboard --logdir={log_dir}")
    else:
        print("TensorBoard not available. Metrics will only be printed to console.")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_split=0.8,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = get_model(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_dice = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to tensorboard (if available)
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/train', train_metrics['dice'], epoch)
            writer.add_scalar('Dice/val', val_metrics['dice'], epoch)
            writer.add_scalar('IoU/val', val_metrics['iou'], epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (Dice: {best_val_dice:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            
            # Save sample predictions
            save_predictions(model, val_loader, device, 
                           os.path.join(save_dir, f'predictions_epoch_{epoch}'))
    
    if writer:
        writer.close()
    print(f"\nTraining complete! Best validation Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    train(
        data_dir='training_data',
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-3
    )
