# LLaVA Cross-Layer Adapters

This directory contains the implementation of **HiCroPL's cross-layer knowledge flow mechanism** adapted for **Large Vision-Language Models (LVLMs)**, specifically LLaVA-v1.5-7B.

## Overview

The cross-layer adapter addresses hallucination in LVLMs by implementing bidirectional knowledge flow between vision and language modalities:

- **Shallow Layers (1-16)**: Vision → Language flow
  - Grounds text generation in visual content
  - Prevents generation of objects not present in images
  
- **Deep Layers (17-32)**: Language → Vision flow
  - Enhances visual understanding with semantic information
  - Improves contextual comprehension

## Architecture

```
LLaVA-v1.5-7B (7B parameters, frozen)
├── Vision Encoder: CLIP ViT-L/14
│   └── 24 layers, hidden_dim=1024
└── Language Model: Vicuna-7B
    └── 32 layers, hidden_dim=4096

Cross-Layer Adapter (~3M trainable parameters)
├── Layer-wise Knowledge Pooling (LKP)
│   └── Compresses features to proxy tokens
├── Knowledge Mappers
│   ├── Vision → Language (layers 1-16)
│   └── Language → Vision (layers 17-32)
└── Hallucination Detection
    └── Real-time alignment monitoring
```

## Modules

### `cross_layer_adapter.py`

Core adapter components:

1. **`AttentionPooling`**: LKP module for compressing layer features
2. **`CrossLayerKnowledgeMapper`**: Bidirectional knowledge mapping
3. **`LVLMCrossLayerAdapter`**: Main adapter class with ~3M parameters

Key features:
- Dimension reduction (4096 → 256 → 4096) for efficiency
- Multi-head attention for rich representations
- Residual connections for stable training
- Hallucination loss computation

### `llava_with_adapter.py`

LLaVA integration:

1. **`LLaVAFeatureExtractor`**: Hooks for extracting intermediate features
2. **`LLaVAWithAdapter`**: Model wrapper with adapter injection

Key features:
- Freezes base LLaVA model (only adapters trained)
- Runtime feature extraction via forward hooks
- Automatic adapter checkpoint saving/loading
- Hallucination loss integration

## Usage

### Basic Example

```python
from lvlm_adapters import LVLMCrossLayerAdapter, LLaVAWithAdapter
from transformers import AutoModelForCausalLM

# Load base LLaVA model
llava_model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")

# Configure adapter
adapter_config = {
    'vision_hidden_size': 1024,
    'language_hidden_size': 4096,
    'adapter_hidden_size': 256,
    'num_vision_layers': 24,
    'num_language_layers': 32,
    'cross_layer_point': 16,
    'hallucination_loss_weight': 0.1
}

# Wrap with adapter
model = LLaVAWithAdapter(llava_model, adapter_config, freeze_base_model=True)

# Train (only adapter parameters)
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=2e-4
)

# Forward pass
outputs = model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    labels=labels
)

loss = outputs['loss']  # Language modeling + hallucination loss
loss.backward()
optimizer.step()

# Save adapter (small checkpoint, ~12MB)
model.save_adapter('adapter.pt')

# Load adapter
model.load_adapter('adapter.pt')

# Generate with adapter
generated = model.generate(input_ids=input_ids, pixel_values=pixel_values)
```

### Training

See `scripts/train_llava_adapter.py` for a complete training example:

```bash
python scripts/train_llava_adapter.py \
    --config configs/llava_adapter_config.yaml \
    --output_dir ./outputs/llava_adapter
```

### Evaluation

See `scripts/evaluate_hallucination.py` for hallucination metrics:

```bash
python scripts/evaluate_hallucination.py \
    --config configs/llava_adapter_config.yaml \
    --checkpoint ./outputs/llava_adapter/adapter_best.pt
```

## Design Principles

### 1. Parameter Efficiency

Only ~3M parameters (0.04% of total) are trainable:
- Base LLaVA model: Frozen (7B parameters)
- Adapters: Trainable (3M parameters)
- Checkpoint size: ~12MB (vs ~14GB for full model)

### 2. Bidirectional Knowledge Flow

**Why bidirectional?**
- **Vision → Language**: Prevents hallucination by grounding generation
- **Language → Vision**: Enhances understanding with semantic context

**Why switch at layer 16?**
- Early layers: Need visual grounding (prevent false objects)
- Late layers: Need semantic enhancement (better comprehension)

### 3. Hallucination Detection

Monitors vision-language alignment in real-time:

```python
alignment_score = cosine_similarity(vision_features, language_features)
hallucination_loss = max(0, target_alignment - alignment_score)
```

Low alignment scores indicate potential hallucination.

### 4. Modular Design

Easy to extend to other LVLMs:

```python
# For InstructBLIP
adapter = LVLMCrossLayerAdapter(
    vision_hidden_size=1408,  # Q-Former output
    language_hidden_size=5120,  # Flan-T5-XXL
    # ... other params
)
```

## Performance

Expected improvements on hallucination metrics:

| Metric | Baseline LLaVA | With Adapter | Improvement |
|--------|---------------|--------------|-------------|
| CHAIRs | ~20% | ~15% | ↓ 25% |
| CHAIRi | ~12% | ~9% | ↓ 25% |
| POPE Acc | ~85% | ~87% | ↑ 2% |

**Training cost:**
- Parameters: 3M (0.04% of model)
- Memory: ~8GB GPU (batch_size=4, fp16)
- Time: ~2 hours on COCO (1 GPU)

## Testing

Run the test suite to validate the implementation:

```bash
python scripts/test_adapter.py
```

This will test:
- ✓ Module imports
- ✓ AttentionPooling functionality
- ✓ CrossLayerKnowledgeMapper functionality
- ✓ LVLMCrossLayerAdapter functionality
- ✓ LLaVAWithAdapter wrapper

## Technical Details

### LKP (Layer-wise Knowledge Pooling)

Compresses sequence features into a single proxy token:

1. Initialize learnable proxy token
2. Use as query in multi-head attention
3. Sequence features as key/value
4. Output: Compressed representation

**Benefits:**
- Reduces computational cost
- Focuses on most relevant information
- Enables efficient cross-modal mapping

### Knowledge Mapper

Maps knowledge between modalities:

1. Project to common dimension
2. Cross-attention between modalities
3. Feed-forward network for transformation
4. Residual connections for stability

**Architecture:**
```
Input (target modality)
  ↓ Linear projection
  ↓ Cross-attention with source
  ↓ Residual connection
  ↓ FFN (4x expansion)
  ↓ Residual connection
Output (refined target)
```

### Alignment Score Computation

```python
# Normalize features
vision_norm = vision_features / ||vision_features||
language_norm = language_features / ||language_features||

# Compute cosine similarity
alignment = vision_norm · language_norm

# Higher score = better alignment
# Lower score = potential hallucination
```

## Troubleshooting

### Out of Memory

```python
# Reduce adapter size
adapter_config['adapter_hidden_size'] = 128  # Down from 256
adapter_config['n_tokens'] = 2  # Down from 4
```

### Poor Hallucination Metrics

```python
# Increase hallucination loss weight
adapter_config['hallucination_loss_weight'] = 0.2  # Up from 0.1

# Adjust target alignment
target_alignment = 0.85  # Up from 0.8
```

### Slow Training

```python
# Enable mixed precision
use_amp = True

# Reduce gradient accumulation if memory allows
gradient_accumulation_steps = 2  # Down from 4
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{zheng2025hierarchical,
  title={Hierarchical cross-modal prompt learning for vision-language models},
  author={Zheng, Hao and Yang, Shunzhi and He, Zhuoxin and Yang, Jinfeng and Huang, Zhenhua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1891--1901},
  year={2025}
}
```

## License

This implementation follows the same license as the HiCroPL repository.
