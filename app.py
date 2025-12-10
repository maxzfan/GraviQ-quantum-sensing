from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from model import UNet
import os
from generate_gzz_fast import gzz_approximation

app = Flask(__name__)

# Load model at startup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")
model = UNet(in_channels=1, out_channels=1)

if os.path.exists('checkpoints/best_model.pth'):
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded! (Epoch {checkpoint.get('epoch', 'N/A')}, Val Dice: {checkpoint.get('val_dice', 'N/A'):.4f})")
else:
    print("WARNING: No trained model found. Please train first!")

model = model.to(device)
model.eval()


def predict_tunnel(density_grid, threshold=0.5):
    """Run inference on density grid"""
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
    
    has_tunnel = np.any(binary_mask)
    confidence = float(prob_map.max())
    tunnel_pixels = int(binary_mask.sum())
    
    return prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels


def create_visualization(density_grid, gzz_grid, prob_map, binary_mask):
    """Create visualization as base64 encoded image"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # Input density grid
    im0 = axes[0].imshow(density_grid, cmap='inferno', origin='upper')
    axes[0].set_title('Density Grid', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='Density', fraction=0.046)
    
    # Gzz grid
    im1 = axes[1].imshow(gzz_grid, cmap='viridis', origin='upper')
    axes[1].set_title('Gzz Grid (Quantum)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Gzz', fraction=0.046)
    
    # Prediction probability
    im2 = axes[2].imshow(prob_map, cmap='hot', origin='upper', vmin=0, vmax=1)
    axes[2].set_title('Tunnel Probability', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='Prob', fraction=0.046)
    
    # Binary prediction
    im3 = axes[3].imshow(binary_mask, cmap='binary', origin='upper', vmin=0, vmax=1)
    axes[3].set_title('Detected Tunnels', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def generate_random_grid():
    """Generate a random density grid using the same logic as training data"""
    from density_grid_generator import make_grid
    import random
    seed = random.randint(0, 10000)
    grid, tunnel_mask, metadata = make_grid(seed)
    return grid, metadata


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a random density grid and run inference"""
    try:
        # Generate random grid
        density_grid, metadata = generate_random_grid()
        
        # Generate Gzz grid using fast approximation (instant!)
        gzz_grid = gzz_approximation(density_grid, t_evolution=30e-6)
        
        # Run inference
        prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels = predict_tunnel(density_grid)
        
        # Create visualization
        img_base64 = create_visualization(density_grid, gzz_grid, prob_map, binary_mask)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'has_tunnel': bool(has_tunnel),
            'confidence': float(confidence),
            'tunnel_pixels': int(tunnel_pixels),
            'ground_truth': {
                'has_tunnel': metadata['has_tunnel'],
                'num_tunnels': metadata['num_tunnels']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload():
    """Upload a .npy density grid file and run inference"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.npy'):
            return jsonify({'success': False, 'error': 'Please upload a .npy file'}), 400
        
        # Load density grid from uploaded file
        density_grid = np.load(file)
        
        # Validate shape
        if density_grid.shape != (60, 150):
            return jsonify({
                'success': False, 
                'error': f'Invalid grid shape {density_grid.shape}. Expected (60, 150)'
            }), 400
        
        # Check if corresponding Gzz grid exists in training_data
        # Try to extract sample ID from filename (e.g., density_grid_042.npy)
        gzz_grid = None
        if file.filename.startswith('density_grid_'):
            sample_id = file.filename.replace('density_grid_', '').replace('.npy', '')
            gzz_path = os.path.join('training_data', f'gzz_grid_{sample_id}.npy')
            
            if os.path.exists(gzz_path):
                print(f"✓ Loading pre-computed Gzz grid: {gzz_path}")
                gzz_grid = np.load(gzz_path)
        
        # Generate Gzz grid if not found (using fast approximation)
        if gzz_grid is None:
            print("Generating Gzz grid using fast approximation...")
            gzz_grid = gzz_approximation(density_grid, t_evolution=30e-6)
        
        # Run inference
        prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels = predict_tunnel(density_grid)
        
        # Create visualization
        img_base64 = create_visualization(density_grid, gzz_grid, prob_map, binary_mask)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'has_tunnel': bool(has_tunnel),
            'confidence': float(confidence),
            'tunnel_pixels': int(tunnel_pixels)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Run inference with custom threshold"""
    try:
        data = request.get_json()
        threshold = float(data.get('threshold', 0.5))
        
        # This would need the grid stored in session or re-uploaded
        # For simplicity, just return current threshold
        return jsonify({'success': True, 'threshold': threshold})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("GraviQ Tunnel Detection Demo Server")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: U-Net (380K params)")
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
