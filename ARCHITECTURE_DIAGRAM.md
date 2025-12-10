# GraviQ System Architecture
*Visual diagrams and flowcharts*

---

## 🏗️ High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GRAVIQ SYSTEM OVERVIEW                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│   OFFLINE    │  Data Generation & Training
│   PIPELINE   │
└──────────────┘
      │
      ├─► [1] Generate Synthetic Data
      │    ├─ Density Grids (geological features)
      │    ├─ Tunnel Masks (ground truth)
      │    └─ Metadata (labels)
      │    
      ├─► [2] Compute Quantum Sensing
      │    └─ Gzz Grids (Ramsey interferometry)
      │    
      ├─► [3] Train ML Model
      │    ├─ U-Net architecture
      │    ├─ Dice + BCE loss
      │    └─ Checkpoint best model
      │    
      └─► [4] Evaluate Performance
           ├─ Validation metrics
           └─ Success/failure visualization

┌──────────────┐
│    ONLINE    │  Production Inference
│   PIPELINE   │
└──────────────┘
      │
      ├─► [5] Web API (Flask)
      │    ├─ Random generation endpoint
      │    └─ File upload endpoint
      │    
      ├─► [6] Real-Time Inference
      │    ├─ Load density grid
      │    ├─ Compute/load Gzz grid
      │    ├─ ML prediction
      │    └─ Visualization
      │    
      └─► [7] User Interface
           └─ Interactive web demo
```

---

## 🔄 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                         │
└─────────────────────────────────────────────────────────────────┘

START: density_grid_generator.py
│
├─► Generate Random Parameters
│   ├─ num_tunnels ∈ {0, 1, 2, 3}
│   ├─ tunnel_width, height, rotation
│   ├─ ore_count ∈ {2, 3, 4, 5}
│   └─ void_count ∈ {1, 2, 3}
│
├─► Create Density Grid (60 × 150)
│   ├─ Base rock: ρ = 1.0
│   ├─ Add ore blobs: ρ = 2.5
│   ├─ Add void pockets: ρ = 0.0
│   └─ Add tunnels: ρ = 0.0
│
├─► Create Tunnel Mask (60 × 150)
│   └─ Binary: 1 = tunnel, 0 = not tunnel
│
├─► Save Outputs
│   ├─ density_grid_{id}.npy    (input)
│   ├─ tunnel_mask_{id}.npy     (label)
│   ├─ metadata_{id}.json       (info)
│   └─ visualization_{id}.png   (QC)
│
└─► FOR EACH GRID: generate_gzz_fast.py
    │
    ├─► Load Density Grid
    │
    ├─► Compute Gzz
    │   └─ Gzz = exp(-density × t × 10^6)
    │       + noise ~ N(0, 0.02)
    │
    └─► Save Gzz Grid
        └─ gzz_grid_{id}.npy

┌───────────────────────────────────┐
│  TRAINING DATA READY (500 samples) │
└───────────────────────────────────┘

NEXT: train.py
│
├─► Load Dataset
│   ├─ dataset.py → TunnelDataset
│   └─ DataLoader (batch_size=16)
│
├─► Initialize Model
│   └─ model.py → UNet(in=1, out=1)
│
├─► Training Loop (100 epochs)
│   │
│   ├─► Forward Pass
│   │   ├─ Input: density grid (1, 60, 150)
│   │   └─ Output: logits (1, 60, 150)
│   │
│   ├─► Compute Loss
│   │   ├─ Dice Loss (overlap metric)
│   │   ├─ BCE Loss (pixel-wise)
│   │   └─ Combined: 0.5 × Dice + 0.5 × BCE
│   │
│   ├─► Backward Pass
│   │   └─ Adam optimizer step
│   │
│   ├─► Validation
│   │   ├─ Compute metrics (Dice, IoU)
│   │   └─ Learning rate scheduling
│   │
│   └─► Checkpoint
│       └─ Save if val_dice improved
│
└─► Best Model Saved
    └─ checkpoints/best_model.pth

┌───────────────────────────────────┐
│    MODEL READY FOR DEPLOYMENT      │
└───────────────────────────────────┘

DEPLOY: app.py (Flask web server)
│
├─► Load Model at Startup
│   └─ model.load_state_dict(checkpoint)
│
├─► API Endpoint: POST /generate
│   │
│   ├─► Generate random density grid
│   ├─► Compute Gzz (fast mode)
│   ├─► Run inference
│   ├─► Create 4-panel visualization
│   └─► Return JSON + base64 image
│
├─► API Endpoint: POST /upload
│   │
│   ├─► Load uploaded .npy file
│   ├─► Load or compute Gzz
│   ├─► Run inference
│   ├─► Create visualization
│   └─► Return JSON + base64 image
│
└─► Serve UI: GET /
    └─ templates/index.html

┌───────────────────────────────────┐
│   WEB DEMO LIVE (localhost:5000)   │
└───────────────────────────────────┘
```

---

## 🧠 U-Net Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     U-NET ARCHITECTURE                           │
│                                                                  │
│  Input: (1, 60, 150) - Single channel density grid              │
└─────────────────────────────────────────────────────────────────┘

ENCODER (Downsampling Path)
│
├─► DoubleConv(1 → 64)         [60×150]
│   ├─ Conv2d(1, 64, 3×3) + BN + ReLU
│   └─ Conv2d(64, 64, 3×3) + BN + ReLU
│
├─► MaxPool2d(2×2)              [30×75]
│
├─► DoubleConv(64 → 128)        [30×75]
│   ├─ Conv2d(64, 128, 3×3) + BN + ReLU
│   └─ Conv2d(128, 128, 3×3) + BN + ReLU
│
├─► MaxPool2d(2×2)              [15×37]
│
├─► DoubleConv(128 → 256)       [15×37]
│   ├─ Conv2d(128, 256, 3×3) + BN + ReLU
│   └─ Conv2d(256, 256, 3×3) + BN + ReLU
│
├─► MaxPool2d(2×2)              [7×18]
│
└─► DoubleConv(256 → 512)       [7×18]
    ├─ Conv2d(256, 512, 3×3) + BN + ReLU
    └─ Conv2d(512, 512, 3×3) + BN + ReLU

BOTTLENECK
│
└─► MaxPool2d(2×2)              [3×9]
    │
    └─► DoubleConv(512 → 1024)   [3×9]
        ├─ Conv2d(512, 1024, 3×3) + BN + ReLU
        └─ Conv2d(1024, 1024, 3×3) + BN + ReLU

DECODER (Upsampling Path + Skip Connections)
│
├─► Up(1024 → 512)              [7×18]
│   ├─ UpConv(1024, 512)
│   ├─ Concat with skip from encoder [512+512=1024]
│   └─ DoubleConv(1024 → 512)
│
├─► Up(512 → 256)               [15×37]
│   ├─ UpConv(512, 256)
│   ├─ Concat with skip [256+256=512]
│   └─ DoubleConv(512 → 256)
│
├─► Up(256 → 128)               [30×75]
│   ├─ UpConv(256, 128)
│   ├─ Concat with skip [128+128=256]
│   └─ DoubleConv(256 → 128)
│
└─► Up(128 → 64)                [60×150]
    ├─ UpConv(128, 64)
    ├─ Concat with skip [64+64=128]
    └─ DoubleConv(128 → 64)

OUTPUT HEAD
│
└─► Conv2d(64 → 1, 1×1)         [60×150]
    │
    └─► Sigmoid (during inference)
        │
        └─► Binary Mask (threshold @ 0.5)

┌─────────────────────────────────────────────────────────────────┐
│  Output: (1, 60, 150) - Tunnel probability map                  │
│  Total Parameters: ~380,000                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚛️ Quantum Sensing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  QUANTUM SENSING (Gzz) PIPELINE                  │
└─────────────────────────────────────────────────────────────────┘

INPUT: Density Grid ρ(x,z)  [60 × 150 array]

┌─────────────────────────────────────────────┐
│         FOR EACH PIXEL (i, j):              │
└─────────────────────────────────────────────┘

METHOD A: Fast Approximation (Default)
│
├─► Extract density value: ρ
│
├─► Apply exponential decay model
│   └─ Gzz = exp(-ρ × t × 10^6)
│       where t = 30 μs
│
├─► Add quantum noise
│   └─ Gzz += N(0, 0.02)
│
├─► Clip to physical range
│   └─ Gzz ∈ [-1, 1]
│
└─► Time: ~0.001 ms per pixel
    Total: ~0.1 s per grid

METHOD B: Full Quantum Simulation (Research)
│
├─► Extract density value: ρ
│
├─► Compute damping probability
│   └─ p = 1 - exp(-ρ × t)
│
├─► Create quantum circuit
│   │
│   └─► Ramsey Sequence:
│       ┌───┐        ┌───┐┌─┐
│       │ H │─[delay]─│ H ││M│
│       └───┘    t    └───┘└─┘
│       |0⟩ state, measured in Z basis
│
├─► Add phase damping noise
│   └─ NoiseModel with p(ρ, t)
│
├─► Run Qiskit simulation
│   └─ shots = 200 (fast) or 1000 (quality)
│
├─► Measure outcomes
│   ├─ Count |0⟩ and |1⟩ results
│   └─ Compute expectation
│       Gzz = P(0) - P(1)
│
└─► Time: ~0.15 s per pixel (200 shots)
    Total: ~30-120 s per grid

OUTPUT: Gzz Grid  [60 × 150 array]

┌─────────────────────────────────────────────┐
│  Physical Interpretation:                   │
│  • Gzz ≈ +1.0  →  Low ρ (tunnel/void)       │
│  • Gzz ≈  0.5  →  Med ρ (rock)              │
│  • Gzz ≈  0.0  →  High ρ (ore)              │
└─────────────────────────────────────────────┘
```

---

## 🌐 Web Demo Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WEB DEMO ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────────┘

CLIENT (Browser)
│
├─► templates/index.html
│   ├─ Generate Random button → POST /generate
│   ├─ Upload File button → POST /upload
│   ├─ Threshold slider (0.0 - 1.0)
│   └─ Results display area
│
└─► JavaScript Functions
    ├─ showLoading() / hideLoading()
    ├─ displayResults(data)
    ├─ generateRandom()
    └─ uploadFile()

        ▼ HTTP Request

SERVER (Flask - app.py)
│
├─► Route: GET /
│   └─ return render_template('index.html')
│
├─► Route: POST /generate
│   │
│   ├─► [1] Generate Random Grid
│   │   └─ make_grid() → density, metadata
│   │
│   ├─► [2] Compute Gzz
│   │   └─ gzz_approximation(density)
│   │
│   ├─► [3] Run ML Inference
│   │   ├─ Preprocess: to tensor
│   │   ├─ model(tensor) → logits
│   │   └─ sigmoid(logits) → prob_map
│   │
│   ├─► [4] Post-process
│   │   ├─ Binary mask (threshold)
│   │   ├─ has_tunnel = any(mask)
│   │   ├─ confidence = max(prob_map)
│   │   └─ tunnel_pixels = sum(mask)
│   │
│   ├─► [5] Create Visualization
│   │   ├─ 4-panel matplotlib figure
│   │   ├─ Save to BytesIO buffer
│   │   └─ base64 encode PNG
│   │
│   └─► [6] Return JSON
│       └─ {success, image, metrics, ground_truth}
│
└─► Route: POST /upload
    │
    ├─► [1] Load .npy file
    │   └─ np.load(uploaded_file)
    │
    ├─► [2] Check for pre-computed Gzz
    │   ├─ If exists: load from training_data/
    │   └─ Else: compute on-the-fly
    │
    ├─► [3-6] Same as /generate
    │   (inference, visualization, return)
    │
    └─► Return JSON (no ground_truth)

        ▼ HTTP Response

CLIENT (Browser)
│
└─► Display Results
    ├─ Show metrics (tunnel: YES/NO, confidence, pixels)
    ├─ Show ground truth (if available)
    └─ Display 4-panel image (base64 decoded)

┌─────────────────────────────────────────────┐
│  Performance:                                │
│  • Random generation: ~500 ms               │
│  • File upload: ~200 ms                     │
│  • Bottleneck: Gzz computation + viz        │
└─────────────────────────────────────────────┘
```

---

## 📊 Training Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

INITIALIZATION
│
├─► Load Dataset
│   ├─ TunnelDataset (density, mask pairs)
│   ├─ Train/Val split: 400/100
│   └─ DataLoader (batch_size=16, shuffle)
│
├─► Initialize Model
│   ├─ UNet(in_channels=1, out_channels=1)
│   └─ Move to device (CPU/GPU)
│
├─► Initialize Optimizer
│   ├─ Adam(lr=1e-3)
│   └─ ReduceLROnPlateau scheduler
│
└─► Initialize Logging
    ├─ TensorBoard writer (optional)
    └─ Best model tracker

TRAINING LOOP (100 epochs)
│
├─► FOR EACH EPOCH:
│   │
│   ├─► TRAINING PHASE
│   │   │
│   │   ├─► FOR EACH BATCH:
│   │   │   │
│   │   │   ├─► Forward Pass
│   │   │   │   ├─ density → model → logits
│   │   │   │   └─ sigmoid(logits) → probs
│   │   │   │
│   │   │   ├─► Compute Loss
│   │   │   │   ├─ dice_loss = DiceLoss(probs, mask)
│   │   │   │   ├─ bce_loss = BCELoss(logits, mask)
│   │   │   │   └─ total = 0.5×dice + 0.5×bce
│   │   │   │
│   │   │   ├─► Backward Pass
│   │   │   │   ├─ loss.backward()
│   │   │   │   └─ optimizer.step()
│   │   │   │
│   │   │   └─► Accumulate Metrics
│   │   │       ├─ Running loss
│   │   │       └─ Running Dice score
│   │   │
│   │   └─► Average over batches
│   │       └─ train_loss, train_dice
│   │
│   ├─► VALIDATION PHASE
│   │   │
│   │   ├─► FOR EACH BATCH (no_grad):
│   │   │   │
│   │   │   ├─► Forward Pass
│   │   │   │   └─ density → model → probs
│   │   │   │
│   │   │   ├─► Compute Metrics
│   │   │   │   ├─ Dice score
│   │   │   │   ├─ IoU
│   │   │   │   └─ Pixel accuracy
│   │   │   │
│   │   │   └─► Accumulate
│   │   │
│   │   └─► Average over batches
│   │       └─ val_loss, val_dice, val_iou
│   │
│   ├─► LOGGING
│   │   ├─ Print to console
│   │   └─ TensorBoard (if available)
│   │       ├─ Loss curves
│   │       ├─ Dice/IoU curves
│   │       └─ Learning rate
│   │
│   ├─► CHECKPOINTING
│   │   └─ IF val_dice improved:
│   │       └─ Save best_model.pth
│   │           ├─ model_state_dict
│   │           ├─ optimizer_state_dict
│   │           ├─ epoch
│   │           └─ best_val_dice
│   │
│   └─► LR SCHEDULING
│       └─ ReduceLROnPlateau(val_loss)
│           └─ Reduce lr if no improvement
│
└─► CONVERGENCE
    └─ Best model @ Epoch 52
        ├─ val_dice = 0.9913
        └─ val_iou = 0.9828

┌─────────────────────────────────────────────┐
│  Final Model: checkpoints/best_model.pth    │
│  Training Time: ~3 minutes (CPU)            │
│  Ready for deployment!                      │
└─────────────────────────────────────────────┘
```

---

## 🔄 Inference Flow (Single Sample)

```
┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE (SINGLE SAMPLE)             │
└─────────────────────────────────────────────────────────────────┘

INPUT: density_grid.npy  [60 × 150]

PREPROCESSING
│
├─► Load Data
│   └─ density = np.load(file)
│
├─► Convert to Tensor
│   └─ tensor = torch.FloatTensor(density)
│       .unsqueeze(0)    # Add batch dim → (1, 60, 150)
│       .unsqueeze(0)    # Add channel dim → (1, 1, 60, 150)
│
└─► Move to Device
    └─ tensor = tensor.to(device)

INFERENCE
│
├─► Model Forward Pass (no_grad)
│   └─ logits = model(tensor)  # (1, 1, 60, 150)
│
├─► Apply Sigmoid
│   └─ prob_map = torch.sigmoid(logits)
│       .squeeze()              # (60, 150)
│       .cpu().numpy()
│
└─► Threshold to Binary
    └─ binary_mask = (prob_map > threshold).astype(uint8)

POST-PROCESSING
│
├─► Extract Predictions
│   ├─ has_tunnel = np.any(binary_mask)
│   ├─ confidence = float(prob_map.max())
│   └─ tunnel_pixels = int(binary_mask.sum())
│
└─► Compute Metrics (if ground truth available)
    ├─ dice_score = 2×|pred∩true| / (|pred|+|true|)
    ├─ iou_score = |pred∩true| / |pred∪true|
    └─ pixel_accuracy = correct_pixels / total_pixels

VISUALIZATION
│
├─► Create 4-Panel Figure
│   ├─ Panel 1: Input density (inferno)
│   ├─ Panel 2: Gzz grid (viridis)
│   ├─ Panel 3: Probability map (hot)
│   └─ Panel 4: Binary mask (binary)
│
├─► Save to Buffer / File
│   └─ plt.savefig() → PNG
│
└─► Encode (for web)
    └─ base64.b64encode(png_bytes)

OUTPUT
│
└─► Results Dictionary
    ├─ has_tunnel: bool
    ├─ confidence: float
    ├─ tunnel_pixels: int
    ├─ prob_map: (60, 150) array
    ├─ binary_mask: (60, 150) array
    └─ visualization: base64 PNG

┌─────────────────────────────────────────────┐
│  Inference Time: ~50 ms                     │
│  Accuracy: 99%+ Dice score                  │
└─────────────────────────────────────────────┘
```

---

## 📦 File Organization Map

```
GraviQ-quantum-sensing/
│
├── 📁 Core Scripts (Offline Pipeline)
│   ├── density_grid_generator.py   ← Generate synthetic data
│   ├── generate_gzz_fast.py        ← Fast Gzz computation
│   ├── generate_gzz_grids.py       ← Full quantum simulation
│   ├── gridToGzz.py                ← Reference implementation
│   ├── dataset.py                  ← PyTorch Dataset
│   ├── model.py                    ← U-Net architecture
│   ├── train.py                    ← Training loop
│   └── inference.py                ← Evaluation script
│
├── 📁 Web Demo (Online Pipeline)
│   ├── app.py                      ← Flask backend
│   └── templates/
│       └── index.html              ← Frontend UI
│
├── 📁 Generated Data (Not in repo)
│   ├── training_data/
│   │   ├── density_grid_000.npy    ← Input samples (500×)
│   │   ├── gzz_grid_000.npy        ← Quantum data (500×)
│   │   ├── tunnel_mask_000.npy     ← Ground truth (500×)
│   │   ├── metadata_000.json       ← Labels (500×)
│   │   └── visualization_000.png   ← QC images (500×)
│   │
│   ├── checkpoints/
│   │   └── best_model.pth          ← Trained weights
│   │
│   ├── runs/                       ← TensorBoard logs
│   │
│   └── evaluation_results/
│       ├── successful_dice_above_0.8/
│       ├── unsuccessful_dice_below_0.8/
│       └── evaluation_metrics.json
│
├── 📁 Documentation
│   ├── README_training.md          ← Training guide
│   ├── README_demo.md              ← Web demo guide
│   ├── README_gzz.md               ← Quantum sensing guide
│   ├── TECHNICAL_SUMMARY.md        ← Full technical doc
│   ├── PRESENTATION_OVERVIEW.md    ← Executive summary
│   ├── QUICK_REFERENCE.md          ← Cheat sheet
│   └── ARCHITECTURE_DIAGRAM.md     ← This file
│
└── 📄 Config Files
    ├── requirements.txt            ← Dependencies
    └── .gitignore                  ← Git exclusions
```

---

## 🎯 Decision Tree: Which Script to Use?

```
START
│
├─ Need training data?
│  YES → Run density_grid_generator.py
│         └─ Generates 500 samples with tunnels/masks
│
├─ Need Gzz grids?
│  ├─ Fast demo? → generate_gzz_fast.py (13s total)
│  └─ Research?  → generate_gzz_grids.py (4hrs total)
│
├─ Need to train model?
│  YES → Run train.py
│         └─ Trains U-Net, saves best_model.pth
│
├─ Need to evaluate model?
│  YES → Run inference.py
│         └─ Generates visualizations, metrics
│
├─ Need web demo?
│  YES → Run app.py
│         └─ Launch Flask server on :5000
│
└─ Need to modify?
   ├─ Architecture → model.py
   ├─ Loss/Training → train.py
   ├─ Data generation → density_grid_generator.py
   ├─ Quantum sensing → generate_gzz_*.py
   └─ Web interface → app.py, templates/index.html
```

---

**Last Updated:** December 2025  
**For:** Technical presentations and system understanding
