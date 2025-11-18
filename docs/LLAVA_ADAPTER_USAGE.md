# LLaVA Cross-Layer Adapter User Guide
# LLaVA 跨层适配器使用指南

This guide explains how to use HiCroPL's cross-layer knowledge flow mechanism with LLaVA-v1.5-7B for hallucination mitigation.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Architecture Details](#architecture-details)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

## Overview

### What is this?

This implementation applies **HiCroPL's bidirectional cross-layer knowledge flow** mechanism to **LLaVA-v1.5-7B** to mitigate hallucination in large vision-language models (LVLMs).

### Key Features

- ✅ **Lightweight Adaptation**: Only ~3M trainable parameters (adapters)
- ✅ **Base Model Frozen**: LLaVA-v1.5-7B parameters remain unchanged
- ✅ **Bidirectional Knowledge Flow**:
  - Shallow layers (1-16): Vision → Language (visual grounding)
  - Deep layers (17-32): Language → Vision (semantic enhancement)
- ✅ **Hallucination Detection**: Real-time vision-language alignment monitoring
- ✅ **Efficient Training**: Mixed precision (FP16) support

### Architecture

```
LLaVA-v1.5-7B
├── Vision Encoder: CLIP ViT-L/14 (24 layers, hidden_dim=1024)
├── Language Model: Vicuna-7B (32 layers, hidden_dim=4096)
└── Cross-Layer Adapter (~3M parameters)
    ├── Layers 1-16: Vision → Language (prevent hallucination)
    └── Layers 17-32: Language → Vision (enhance understanding)
```

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.13.0
- CUDA >= 11.7 (for GPU training)

### Step 1: Install PyTorch

```bash
# For CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision
```

### Step 2: Install Dependencies

```bash
cd /path/to/HiCroPL
pip install -r requirements.txt
```

### Step 3: Install Additional Packages

```bash
# For LLaVA model loading
pip install transformers>=4.30.0
pip install accelerate
pip install sentencepiece

# For evaluation
pip install matplotlib
pip install scikit-learn

# For data processing
pip install Pillow
pip install pycocotools  # For COCO dataset
```

### Step 4: Download LLaVA Model (Optional)

The training script can automatically download the model from HuggingFace:

```bash
# This will be downloaded automatically on first run
# Model: liuhaotian/llava-v1.5-7b (~14GB)
```

Or manually download:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "liuhaotian/llava-v1.5-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained("./models/llava-v1.5-7b")
tokenizer.save_pretrained("./models/llava-v1.5-7b")
```

## Quick Start

### Test Installation

Test with dummy model (no LLaVA download required):

```bash
cd /path/to/HiCroPL

# Test training script
python scripts/train_llava_adapter.py \
    --config configs/llava_adapter_config.yaml \
    --output_dir ./outputs/test_run \
    --use_dummy_model

# Test evaluation script
python scripts/evaluate_hallucination.py \
    --config configs/llava_adapter_config.yaml \
    --output_dir ./outputs/test_eval
```

### Quick Training Example

```bash
# Train with actual LLaVA model (requires GPU and model download)
python scripts/train_llava_adapter.py \
    --config configs/llava_adapter_config.yaml \
    --output_dir ./outputs/llava_adapter_exp1
```

## Training

### Step 1: Prepare Data

The training script expects data in **COCO Captions format**:

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg"},
    {"id": 2, "file_name": "image2.jpg"}
  ],
  "annotations": [
    {"image_id": 1, "caption": "A person riding a bicycle"},
    {"image_id": 2, "caption": "A cat sitting on a couch"}
  ]
}
```

**Option 1: Use COCO Dataset**

```bash
# Download COCO 2017
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip -d ./data/coco/
unzip val2017.zip -d ./data/coco/
unzip annotations_trainval2017.zip -d ./data/coco/
```

**Option 2: Use Custom Data**

Prepare your data in the same format and update paths in config.

### Step 2: Configure Training

Edit `configs/llava_adapter_config.yaml`:

```yaml
# Update data paths
data:
  train_data_path: "./data/coco/annotations/captions_train2017.json"
  train_image_dir: "./data/coco/train2017"
  val_data_path: "./data/coco/annotations/captions_val2017.json"
  val_image_dir: "./data/coco/val2017"

# Adjust training hyperparameters
training:
  batch_size: 4  # Reduce if OOM
  num_epochs: 5
  learning_rate: 2.0e-4
  use_amp: true  # Enable mixed precision
```

### Step 3: Start Training

```bash
# Single GPU training
python scripts/train_llava_adapter.py \
    --config configs/llava_adapter_config.yaml \
    --output_dir ./outputs/llava_adapter_coco

# Multi-GPU training (if available)
# Note: Distributed training code needs to be added for multi-GPU
```

### Step 4: Monitor Training

Training logs will show:

```
Epoch 1/5
================================================
Trainable params: 3,145,728 || All params: 7,003,145,728 || Trainable ratio: 0.04%

Epoch 1: 100%|████████| 1000/1000 [10:23<00:00, 1.60it/s, loss=0.8234, lr=2.0e-04]

Epoch 1 Summary:
  loss: 0.8234
  base_loss: 0.7891
  hallucination_loss: 0.0343

Saved adapter to ./outputs/llava_adapter_coco/adapter_epoch_1.pt
```

### Training Tips

1. **GPU Memory**: If OOM, reduce batch_size or use gradient_accumulation_steps
2. **Training Speed**: Enable `use_amp: true` for 2x speedup with minimal quality loss
3. **Checkpoint Size**: Adapter checkpoints are small (~12MB), base model not saved
4. **Monitoring**: Watch `hallucination_loss` - should decrease over epochs

## Evaluation

### Step 1: Prepare Evaluation Data

**CHAIR Evaluation** (requires COCO instances):

```bash
# Already downloaded if using COCO dataset
# Path: ./data/coco/annotations/instances_val2017.json
```

**POPE Evaluation** (download POPE data):

```bash
# Download POPE evaluation data
git clone https://github.com/AoiDragon/POPE.git
cp POPE/output/coco/coco_pope_adversarial.json ./data/pope/

# Or create custom POPE questions (see POPE paper)
```

### Step 2: Configure Evaluation

Edit `configs/llava_adapter_config.yaml`:

```yaml
evaluation:
  chair:
    coco_instances_path: "./data/coco/annotations/instances_val2017.json"
  
  pope:
    pope_data_path: "./data/pope/coco_pope_adversarial.json"
  
  compute_chair: true
  compute_pope: true
  compute_alignment_scores: true
```

### Step 3: Run Evaluation

```bash
# Evaluate trained adapter
python scripts/evaluate_hallucination.py \
    --config configs/llava_adapter_config.yaml \
    --checkpoint ./outputs/llava_adapter_coco/adapter_best.pt \
    --output_dir ./outputs/evaluation_results
```

### Step 4: Analyze Results

Evaluation outputs:

```
CHAIR Results:
  CHAIRs (Sentence-level): 12.3%
  CHAIRi (Instance-level): 8.5%
  Hallucinated captions: 123/1000

POPE Results:
  Accuracy: 87.5%
  Precision: 89.2%
  Recall: 85.8%
  F1 Score: 87.5

Alignment Scores:
  Average: 0.7823
  Min: 0.6521
  Max: 0.8791

Results saved to ./outputs/evaluation_results/evaluation_results.json
Visualization saved to ./outputs/evaluation_results/alignment_scores.png
```

**Interpreting Metrics**:

- **CHAIRs**: Lower is better (% of captions with hallucinations)
- **CHAIRi**: Lower is better (% of hallucinated objects)
- **POPE Accuracy**: Higher is better (object detection accuracy)
- **Alignment Score**: Higher is better (vision-language alignment)

### Comparing with Baseline

```bash
# Evaluate baseline LLaVA (without adapter)
# Compare results to see hallucination reduction
```

Expected improvements:
- ✅ CHAIRs: 15-20% reduction
- ✅ CHAIRi: 10-15% reduction  
- ✅ POPE Accuracy: 2-3% improvement

## Architecture Details

### Cross-Layer Adapter Components

1. **AttentionPooling (LKP)**
   - Compresses layer features into proxy tokens
   - Enables efficient cross-modal communication

2. **CrossLayerKnowledgeMapper**
   - Maps knowledge between modalities
   - Cross-attention + FFN architecture

3. **Bidirectional Knowledge Flow**
   - **Shallow (1-16)**: Vision → Language
     - Grounds text generation in visual content
     - Prevents hallucinated objects
   - **Deep (17-32)**: Language → Vision
     - Enhances visual understanding with semantics
     - Improves context comprehension

### Parameter Efficiency

```
Total Parameters: ~7B (LLaVA base)
Trainable Parameters: ~3M (0.04%)
Breakdown:
  - Projection layers: ~1M
  - LKP networks: ~0.5M
  - Knowledge mappers: ~1.5M
```

### Hallucination Loss

```python
# Vision-language alignment score (cosine similarity)
alignment_score = cos_sim(vision_features, language_features)

# Penalize low alignment (potential hallucination)
hallucination_loss = max(0, target_alignment - alignment_score)

# Total loss
total_loss = language_modeling_loss + λ * hallucination_loss
```

## Customization

### Adjust Adapter Architecture

Edit `configs/llava_adapter_config.yaml`:

```yaml
model:
  adapter:
    hidden_size: 256  # Increase for more capacity
    n_tokens: 4  # More tokens = richer representation
    cross_layer_point: 16  # Change flow switch point
    num_attention_heads: 8  # Multi-head attention
```

### Modify Loss Weighting

```yaml
training:
  hallucination_loss_weight: 0.1  # Increase to focus on hallucination
```

### Fine-tune for Specific Domain

```yaml
# Example: Medical imaging
training:
  batch_size: 2  # Smaller for high-res medical images
  learning_rate: 1.0e-4  # Lower LR for fine-tuning

model:
  adapter:
    cross_layer_point: 12  # Earlier switch for domain-specific
```

### Extend to Other Models

The adapter is modular and can be adapted to other LVLMs:

```python
from lvlm_adapters import LVLMCrossLayerAdapter

# For a different model (e.g., InstructBLIP)
adapter = LVLMCrossLayerAdapter(
    vision_hidden_size=1408,  # Adjust for your model
    language_hidden_size=5120,
    num_vision_layers=39,
    num_language_layers=40,
    # ... other params
)
```

## Troubleshooting

### Out of Memory (OOM)

```yaml
# Solution 1: Reduce batch size
training:
  batch_size: 2
  gradient_accumulation_steps: 8  # Keep effective batch size

# Solution 2: Enable mixed precision
training:
  use_amp: true

# Solution 3: Reduce adapter size
model:
  adapter:
    hidden_size: 128  # Down from 256
    n_tokens: 2  # Down from 4
```

### Slow Training

```bash
# Enable mixed precision
use_amp: true  # 2x speedup

# Reduce data workers if CPU bottleneck
num_workers: 2  # Down from 4

# Use gradient checkpointing (add to code)
model.gradient_checkpointing_enable()
```

### Model Not Found

```bash
# If HuggingFace download fails, manually download:
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir ./models/llava

# Then update config:
model:
  llava_model_name: "./models/llava"
```

### Poor Hallucination Metrics

```yaml
# Increase hallucination loss weight
training:
  hallucination_loss_weight: 0.2  # Up from 0.1
  target_alignment_score: 0.85  # Up from 0.8

# Train longer
training:
  num_epochs: 10  # Up from 5
```

### Compatibility Issues

```bash
# Ensure compatible versions
pip install transformers==4.34.0
pip install torch==2.0.1
pip install accelerate==0.23.0

# Check CUDA compatibility
python -c "import torch; print(torch.cuda.is_available())"
```

## Advanced Usage

### Loading Pre-trained Adapter

```python
from lvlm_adapters import LLaVAWithAdapter

# Load base model
llava_model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

# Wrap with adapter
model = LLaVAWithAdapter(llava_model, adapter_config)

# Load trained adapter weights
model.load_adapter("./outputs/llava_adapter_coco/adapter_best.pt")

# Use for inference
outputs = model.generate(input_ids=..., pixel_values=...)
```

### Extracting Alignment Scores

```python
# During forward pass
outputs = model(input_ids=..., pixel_values=..., labels=...)

# Get per-layer alignment scores
alignment_scores = outputs['alignment_scores']
print(f"Layer-wise alignment: {alignment_scores}")

# Identify problematic layers
low_alignment_layers = [i for i, s in enumerate(alignment_scores) if s < 0.7]
print(f"Low alignment at layers: {low_alignment_layers}")
```

### Custom Hallucination Detection

```python
# Implement custom hallucination metric
def custom_hallucination_loss(vision_feat, lang_feat):
    # Your custom logic here
    return loss

# Modify adapter class to use custom loss
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zheng2025hierarchical,
  title={Hierarchical cross-modal prompt learning for vision-language models},
  author={Zheng, Hao and Yang, Shunzhi and He, Zhuoxin and Yang, Jinfeng and Huang, Zhenhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1891--1901},
  year={2025}
}
```

## Support

For issues and questions:
- GitHub Issues: [https://github.com/PaopaoLaoZhang/HiCroPL/issues](https://github.com/PaopaoLaoZhang/HiCroPL/issues)
- Original HiCroPL Paper: [https://arxiv.org/pdf/2507.14976](https://arxiv.org/pdf/2507.14976)

## License

This project follows the same license as the original HiCroPL repository.
