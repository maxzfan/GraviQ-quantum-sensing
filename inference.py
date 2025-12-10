import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import UNet


def load_model(checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained model from checkpoint"""
    model = UNet(in_channels=1, out_channels=1)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_dice' in checkpoint:
        print(f"  Val Dice: {checkpoint['val_dice']:.4f}")
    
    return model


def predict(model, density_grid, device='cuda' if torch.cuda.is_available() else 'cpu', threshold=0.5):
    """
    Run inference on a single density grid.
    
    Args:
        model: Trained UNet model
        density_grid: (H, W) numpy array
        device: torch device
        threshold: Probability threshold for binary mask
    
    Returns:
        prediction_prob: (H, W) probability map
        prediction_binary: (H, W) binary mask
        has_tunnel: boolean
    """
    # Prepare input
    input_tensor = torch.from_numpy(density_grid).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output)
    
    # Convert to numpy
    prob_map = prob_map.squeeze().cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    # Determine if tunnel exists (if any pixel > threshold)
    has_tunnel = np.any(binary_mask)
    
    return prob_map, binary_mask, has_tunnel


def visualize_prediction(density_grid, ground_truth, prediction_prob, prediction_binary, 
                         save_path=None, show=True, verbose=True):
    """Visualize input, ground truth, and prediction"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Input density grid
    im0 = axes[0, 0].imshow(density_grid, cmap='inferno', origin='upper')
    axes[0, 0].set_title('Input Density Grid')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], label='Density')
    
    # Ground truth
    im1 = axes[0, 1].imshow(ground_truth, cmap='binary', origin='upper', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth Tunnel Mask')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Prediction probability
    im2 = axes[1, 0].imshow(prediction_prob, cmap='hot', origin='upper', vmin=0, vmax=1)
    axes[1, 0].set_title('Prediction Probability')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], label='Probability')
    
    # Prediction binary
    im3 = axes[1, 1].imshow(prediction_binary, cmap='binary', origin='upper', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction Binary (threshold=0.5)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if verbose:
            print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def evaluate_dataset(model, data_dir, device='cuda' if torch.cuda.is_available() else 'cpu', 
                     save_dir='evaluation_results', max_samples_per_category=10):
    """Evaluate model on entire dataset and save results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subdirectories for successful and unsuccessful predictions
    success_dir = os.path.join(save_dir, 'successful_dice_above_0.8')
    failure_dir = os.path.join(save_dir, 'unsuccessful_dice_below_0.8')
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)
    
    from dataset import TunnelDataset
    dataset = TunnelDataset(data_dir)
    
    total_correct = 0
    total_samples = 0
    
    dice_scores = []
    iou_scores = []
    
    # Track saved samples
    num_success_saved = 0
    num_failure_saved = 0
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        density_grid = sample['input'].squeeze().numpy()
        ground_truth = sample['mask'].squeeze().numpy()
        has_tunnel_gt = sample['has_tunnel']
        sample_id = sample['sample_id']
        
        # Predict
        prob_map, binary_mask, has_tunnel_pred = predict(
            model, density_grid, device, threshold=0.5
        )
        
        # Calculate metrics
        intersection = (binary_mask * ground_truth).sum()
        union = binary_mask.sum() + ground_truth.sum() - intersection
        
        iou = intersection / (union + 1e-7)
        dice = (2 * intersection) / (binary_mask.sum() + ground_truth.sum() + 1e-7)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
        
        # Classification accuracy
        if has_tunnel_pred == has_tunnel_gt:
            total_correct += 1
        total_samples += 1
        
        # Save visualization based on Dice score
        if dice >= 0.8 and num_success_saved < max_samples_per_category:
            visualize_prediction(
                density_grid, ground_truth, prob_map, binary_mask,
                save_path=os.path.join(success_dir, f'sample_{sample_id}_dice_{dice:.3f}.png'),
                show=False, verbose=False
            )
            num_success_saved += 1
        elif dice < 0.8 and num_failure_saved < max_samples_per_category:
            visualize_prediction(
                density_grid, ground_truth, prob_map, binary_mask,
                save_path=os.path.join(failure_dir, f'sample_{sample_id}_dice_{dice:.3f}.png'),
                show=False, verbose=False
            )
            num_failure_saved += 1
    
    # Count successful and unsuccessful samples
    num_successful = sum(1 for d in dice_scores if d >= 0.8)
    num_unsuccessful = sum(1 for d in dice_scores if d < 0.8)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Classification Accuracy: {100*total_correct/total_samples:.2f}% ({total_correct}/{total_samples})")
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Mean IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    print(f"\nDice Score Distribution:")
    print(f"  Successful (≥ 0.8): {num_successful}/{total_samples} ({100*num_successful/total_samples:.1f}%)")
    print(f"  Unsuccessful (< 0.8): {num_unsuccessful}/{total_samples} ({100*num_unsuccessful/total_samples:.1f}%)")
    print(f"\nSaved Visualizations:")
    print(f"  {success_dir}: {num_success_saved} samples")
    print(f"  {failure_dir}: {num_failure_saved} samples")
    print(f"{'='*60}\n")
    
    # Save metrics
    results = {
        'classification_accuracy': total_correct / total_samples,
        'mean_dice': float(np.mean(dice_scores)),
        'std_dice': float(np.std(dice_scores)),
        'mean_iou': float(np.mean(iou_scores)),
        'std_iou': float(np.std(iou_scores)),
        'num_successful_dice_above_0.8': num_successful,
        'num_unsuccessful_dice_below_0.8': num_unsuccessful,
        'success_rate': num_successful / total_samples,
    }
    
    import json
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('checkpoints/best_model.pth', device)
    
    # Evaluate on dataset
    # Save up to 10 successful and 10 unsuccessful samples
    evaluate_dataset(model, 'training_data', device, 
                    save_dir='evaluation_results',
                    max_samples_per_category=10)
    
    # Example: Single prediction
    print("\n" + "="*60)
    print("SINGLE SAMPLE PREDICTION")
    print("="*60)
    
    # Load a single sample
    density_grid = np.load('training_data/density_grid_000.npy')
    ground_truth = np.load('training_data/tunnel_mask_000.npy')
    
    prob_map, binary_mask, has_tunnel = predict(model, density_grid, device)
    
    print(f"Detected tunnel: {has_tunnel}")
    print(f"Max probability: {prob_map.max():.4f}")
    print(f"Tunnel pixels: {binary_mask.sum()}")
    
    visualize_prediction(density_grid, ground_truth, prob_map, binary_mask, 
                        save_path='single_prediction.png', show=False)
