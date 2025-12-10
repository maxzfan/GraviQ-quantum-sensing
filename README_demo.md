# GraviQ Tunnel Detection Web Demo

A simple web interface for demoing the tunnel detection AI model.

## Features

- 🎲 **Generate Random Grids**: Create random density grids with/without tunnels
- 📤 **Upload Custom Grids**: Upload your own `.npy` density grid files
- 🎯 **Real-time Detection**: See tunnel detection results instantly
- 📊 **Visual Results**: View input, probability map, and binary detection
- ✅ **Ground Truth Comparison**: For generated samples, see actual vs predicted

## Quick Start

### 1. Install Flask (if not already installed)

```bash
pip install flask
```

Or with your virtual environment:

```bash
venv\Scripts\activate
pip install flask
```

### 2. Make sure you have a trained model

The demo needs `checkpoints/best_model.pth` to exist. If you haven't trained yet:

```bash
python train.py
```

### 3. Start the web server

```bash
python app.py
```

You should see:

```
============================================================
GraviQ Tunnel Detection Demo Server
============================================================
Device: cpu
Model: U-Net (380K params)

Starting server at http://localhost:5000
Press Ctrl+C to stop
============================================================
```

### 4. Open in browser

Navigate to: **http://localhost:5000**

The page will auto-load with a random sample!

## Usage

### Generate Random Grids

Click **"🎲 Generate Random Grid"** to:
- Create a new random density grid
- Automatically run inference
- Show detection results
- Display ground truth (if tunnel exists)

### Upload Your Own Grids

1. Click **"📤 Upload .npy File"**
2. Select a `.npy` file (must be shape `(60, 150)`)
3. Results will display automatically

Valid files:
- Any `.npy` file from `training_data/density_grid_*.npy`
- Custom density grids (60×150 array)

### Understanding Results

**Metrics Display:**
- **Tunnel Detected**: YES/NO (green for yes, red for no)
- **Confidence**: Max probability from the model (0-100%)
- **Tunnel Pixels**: Number of pixels classified as tunnel

**Visualizations:**
1. **Input Density Grid**: Raw input data (colored by density)
2. **Tunnel Probability Map**: Model's confidence per pixel (0-1)
3. **Detected Tunnels (Binary)**: Final binary prediction (threshold = 0.5)

**Ground Truth** (for generated samples only):
- Shows actual tunnel presence and count
- Helps evaluate model accuracy

## API Endpoints

### Generate Random Grid
```http
POST /generate
```

Returns:
```json
{
  "success": true,
  "image": "base64_encoded_image",
  "has_tunnel": true,
  "confidence": 0.987,
  "tunnel_pixels": 1247,
  "ground_truth": {
    "has_tunnel": true,
    "num_tunnels": 2
  }
}
```

### Upload Grid
```http
POST /upload
Content-Type: multipart/form-data

file: density_grid.npy
```

Returns same format as `/generate` (without ground_truth)

## File Structure

```
GraviQ-quantum-sensing/
├── app.py                  # Flask server
├── templates/
│   └── index.html         # Web interface
├── static/
│   └── style.css          # Styling
├── checkpoints/
│   └── best_model.pth     # Trained model (required)
└── training_data/         # Sample grids for testing
    ├── density_grid_*.npy
    └── ...
```

## Customization

### Change Detection Threshold

Edit in `app.py`:
```python
def predict_tunnel(density_grid, threshold=0.5):  # Change 0.5 to your value
```

Or use the slider in the web UI (currently visual only, needs backend integration).

### Change Port

Edit in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change 5000
```

### Styling

Edit `static/style.css` to customize colors, fonts, layout, etc.

## Troubleshooting

**"No trained model found"**
- Run `python train.py` first to create `checkpoints/best_model.pth`

**"Invalid grid shape"**
- Uploaded `.npy` must be exactly (60, 150)
- Check with: `np.load('your_file.npy').shape`

**Port already in use**
- Change port in `app.py` or kill process on port 5000:
  ```bash
  # Windows
  netstat -ano | findstr :5000
  taskkill /PID <PID> /F
  ```

**Can't connect to server**
- Make sure server is running (`python app.py`)
- Check firewall isn't blocking port 5000
- Try http://127.0.0.1:5000 instead of localhost

## Next Steps

- Deploy to cloud (Heroku, AWS, etc.)
- Add batch upload for multiple grids
- Export predictions as JSON/CSV
- Add model comparison (different checkpoints)
- Real-time threshold adjustment
- Download results as images

## Demo Tips

**Show off your model:**
1. Click "Generate Random" multiple times
2. Watch it correctly identify tunnels
3. Note the high confidence on positive detections
4. Upload edge cases from `unsuccessful_dice_below_0.8/`

**Test robustness:**
- Upload grids with no tunnels
- Upload grids with multiple tunnels
- Check probability maps for uncertainty
