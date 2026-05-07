# Traffic Sign Recognition using Deep Learning

A deep learning project that classifies German traffic signs using a
Convolutional Neural Network (CNN) built with Julia and Flux.jl.

---

## Results Summary

| Metric | Value |
|---|---|
| Training Accuracy | 83.6% |
| Test Accuracy | 76.4% |
| Number of Classes | 43 |
| Training Images | 39,209 |
| Test Images | 12,630 |
| Epochs | 30 |
| Optimizer | Adam with learning rate scheduling |
| Input Size | 32x32 pixels |

---

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**
- Source: [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- ~51,000 images across 43 traffic sign classes
- Split into 39,209 training and 12,630 test images

> The dataset is not included in this repository due to its size.
> Download from Kaggle and place contents into the `data/` folder.

---

## Project Structure

traffic-sign-recognition/
│
├── data/                        # Dataset (not tracked by Git)
│   ├── Train/                   # Training images (43 subfolders)
│   ├── Test/                    # Test images
│   ├── Meta/                    # Sign metadata
│   ├── Train.csv                # Training labels
│   └── Test.csv                 # Test labels
│
├── src/
│   ├── data_loading.jl          # Load and decode images
│   ├── preprocessing.jl         # Normalize and batch data
│   ├── model.jl                 # CNN architecture
│   ├── train.jl                 # Training loop
│   └── evaluate.jl              # Evaluation and bias analysis
│
├── results/
│   ├── metrics/
│   │   ├── trained_model_v2.bson    # Saved trained model
│   │   ├── per_class_accuracy.csv   # Per-class results
│   │   └── bias_report.txt          # Bias analysis report
│   └── plots/
│       ├── training_history.png     # Loss and accuracy curves
│       └── demo_predictions.png     # Sample predictions
│
├── main.jl                      # Run full pipeline
├── demo.jl                      # Live demo script
├── report.md                    # Full written report
├── Project.toml                 # Julia dependencies
├── Manifest.toml                # Julia environment lock
└── README.md                    # This file

---

## How to Run

### 1. Install Julia
Download from [julialang.org](https://julialang.org/downloads)

### 2. Clone the repository
```bash
git clone https://github.com/[your-username]/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 3. Install dependencies
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 4. Download the dataset
- Go to [Kaggle GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Download and extract into the `data/` folder

### 5. Run the full pipeline
```julia
include("main.jl")
```

### 6. Load saved model and run demo
```julia
using Pkg
Pkg.activate(".")
include("src/model.jl")
include("src/data_loading.jl")
include("src/preprocessing.jl")
include("src/evaluate.jl")
include("demo.jl")
using BSON: @load
@load "results/metrics/trained_model_v2.bson" model
run_demo(model, 5)
```

---

## Model Architecture

A 4-block CNN implemented using Flux.jl:

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv(3x3, 3→32) + BatchNorm + MaxPool | 16x16x32 | 896 |
| Conv(3x3, 32→64) + BatchNorm + MaxPool | 8x8x64 | 18,496 |
| Conv(3x3, 64→128) + BatchNorm + MaxPool | 4x4x128 | 73,856 |
| Conv(3x3, 128→256) + BatchNorm | 4x4x256 | 295,168 |
| Flatten | 4,096 | 0 |
| Dense(4096→512, relu) + Dropout(0.3) | 512 | 2,097,664 |
| Dense(512→128, relu) + Dropout(0.2) | 128 | 65,664 |
| Dense(128→43) | 43 | 5,547 |

---

## Key Findings

**Best performing classes:**
| Class | Sign | Accuracy |
|---|---|---|
| 32 | End restrictions | 100.0% |
| 16 | No vehicles >3.5t | 99.33% |
| 33 | Turn right ahead | 96.67% |
| 20 | Dangerous curve right | 95.56% |
| 13 | Yield | 93.19% |

**Worst performing classes:**
| Class | Sign | Accuracy |
|---|---|---|
| 0 | Speed limit 20 | 6.67% |
| 39 | Keep left | 35.56% |
| 30 | Ice/snow | 44.0% |
| 3 | Speed limit 60 | 45.11% |
| 19 | Dangerous curve left | 46.67% |

**Bias finding:** Classes with fewer than 100 test samples averaged
64.58% accuracy compared to 77.9% for larger classes — a 13.3% gap
indicating class imbalance bias.

---

## Tools and Libraries

| Tool | Purpose |
|---|---|
| Julia 1.12 | Programming language |
| Flux.jl | Deep learning framework |
| Images.jl | Image loading |
| ImageTransformations.jl | Image resizing |
| CSV.jl | Data handling |
| DataFrames.jl | Data manipulation |
| Plots.jl | Visualisation |
| BSON.jl | Model saving |

---

## Course Information

- **Course:** Machine Learning
- **Assessment:** 100% project based
- **Student:** Alhagie kijera 