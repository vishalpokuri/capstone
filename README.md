# Few-Shot Learning with Prototypical Networks

This repository contains implementations of various few-shot learning approaches using Prototypical Networks with different backbone architectures.

## Project Structure

```
capstone/
├── dataset/                          # Original YOLO format dataset
├── clean_dataset/                    # Processed classification dataset
│   ├── train/                        # Training data
│   ├── val/                          # Validation data
│   └── test/                         # Test data
├── Baseline1_ResNet18_ProtoNet_FewShot.ipynb    # ResNet18 + ProtoNet
├── Baseline2_ResNet50_ProtoNet_FewShot.ipynb    # ResNet50 + ProtoNet
├── Baseline3_VGG16_ProtoNet_FewShot_.ipynb      # VGG16 + ProtoNet
├── Baseline4_MS_ProtoNet_ResNet18.ipynb         # Multi-Scale ProtoNet (ResNet18)
├── Baseline5_MS_ProtoNet_ResNet50.ipynb         # Multi-Scale ProtoNet (ResNet50)
├── comparison.ipynb                              # Results comparison
├── cleaning_script.py                            # YOLO to classification converter
└── Identifier_cleaning_script.py                 # Dataset preprocessing
```

## Dataset

The dataset contains 4 classes:

- `-K` (Potassium deficiency)
- `-N` (Nitrogen deficiency)
- `-P` (Phosphorus deficiency)
- `FN` (False Negative/Normal)

The data is organized in a classification format with separate train/val/test splits.

## Setup

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Verify Dataset Structure

Ensure the `clean_dataset` folder has the following structure:

```
clean_dataset/
├── train/
│   ├── -K/
│   ├── -N/
│   ├── -P/
│   └── FN/
├── val/
│   ├── -K/
│   ├── -N/
│   ├── -P/
│   └── FN/
└── test/
    ├── -K/
    ├── -N/
    ├── -P/
    └── FN/
```

### 3. GPU Setup (Recommended)

For faster training, ensure PyTorch can access your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

## Models

### Baseline Models

1. **Baseline 1: ResNet18 + ProtoNet**
   - Single-scale feature extraction from ResNet18's final layer
   - Standard Prototypical Network approach

2. **Baseline 2: ResNet50 + ProtoNet**
   - Deeper ResNet50 backbone
   - Single-scale features

3. **Baseline 3: VGG16 + ProtoNet**
   - VGG16 convolutional features
   - Global average pooling

### Proposed Models (Multi-Scale)

4. **Baseline 4: MS-ProtoNet-18**
   - Multi-scale feature extraction from ResNet18 (layer2, layer3, layer4)
   - Feature fusion with dimension projection
   - **Feature caching** for efficiency

5. **Baseline 5: MS-ProtoNet-50**
   - Multi-scale features from ResNet50
   - Enhanced representation learning

## Few-Shot Learning Settings

All models are evaluated under:

- **1-shot**: 1 support image per class
- **5-shot**: 5 support images per class
- **10-shot**: 10 support images per class

## Training

Open any baseline notebook and run all cells:

```bash
jupyter notebook Baseline1_ResNet18_ProtoNet_FewShot.ipynb
```

Each notebook includes:

- Data loading and preprocessing
- Model definition
- Training loop (episodic sampling)
- Evaluation
- Results visualization (confusion matrix)

## Results Comparison

Run `comparison.ipynb` to visualize accuracy comparisons across all baselines.

## Key Features

- **Episodic Sampling**: N-way K-shot Q-query episode generation
- **Prototypical Networks**: Distance-based classification using class prototypes
- **Multi-Scale Features**: Novel approach extracting features from multiple network layers
- **Feature Caching**: Optimization technique to reduce redundant computations

## Notes

- All models use ImageNet pre-trained weights
- Image size: 224×224
- Normalization: ImageNet statistics
- 4-way classification (N=4)
- 100 episodes for training
- 20 episodes for testing

## Citation

If you use this code, please cite appropriately.
