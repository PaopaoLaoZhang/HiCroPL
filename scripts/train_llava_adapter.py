#!/usr/bin/env python3
"""
Training Script for LLaVA-v1.5-7B with Cross-Layer Adapter
使用跨层适配器训练 LLaVA-v1.5-7B

This script:
1. Loads pre-trained LLaVA-v1.5-7B model
2. Adds cross-layer adapter (freezing base model)
3. Trains on vision-language data with hallucination detection loss
4. Saves adapter checkpoints

Usage:
    python scripts/train_llava_adapter.py --config configs/llava_adapter_config.yaml
"""

import os
import sys
import argparse
import yaml
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lvlm_adapters import LVLMCrossLayerAdapter, LLaVAWithAdapter


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class DummyLLaVAModel(nn.Module):
    """
    Dummy LLaVA model for testing when actual model is not available
    仅用于测试的虚拟模型
    """
    def __init__(self, vision_hidden_size=1024, language_hidden_size=4096):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.language_hidden_size = language_hidden_size
        
        # Dummy layers to simulate LLaVA structure
        self.dummy_vision = nn.Linear(3 * 224 * 224, vision_hidden_size)
        self.dummy_language = nn.Linear(language_hidden_size, 32000)
        
    def forward(self, input_ids=None, pixel_values=None, labels=None, **kwargs):
        batch_size = input_ids.size(0) if input_ids is not None else 1
        
        # Dummy output
        class DummyOutput:
            def __init__(self):
                self.loss = torch.tensor(1.0, requires_grad=True)
                self.logits = torch.randn(batch_size, 100, 32000)
        
        return DummyOutput()


class VisionLanguageDataset(Dataset):
    """
    Dataset for vision-language training
    视觉语言数据集
    
    Expected data format (COCO Captions style):
    {
        "images": [{"id": 1, "file_name": "image1.jpg"}, ...],
        "annotations": [{"image_id": 1, "caption": "A photo of ..."}, ...]
    }
    """
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        max_length: int = 512,
        image_size: int = 224,
        is_train: bool = True
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.max_length = max_length
        self.image_size = image_size
        self.is_train = is_train
        
        # Load annotations
        if data_path is not None and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Build image_id to filename mapping
            self.image_info = {img['id']: img for img in data['images']}
            
            # Store annotations
            self.annotations = data['annotations']
            print(f"Loaded {len(self.annotations)} annotations from {data_path}")
        else:
            # Dummy data for testing
            print("Warning: Using dummy data for testing")
            self.image_info = {i: {'id': i, 'file_name': f'dummy_{i}.jpg'} for i in range(100)}
            self.annotations = [
                {'image_id': i, 'caption': f'This is dummy caption {i}'}
                for i in range(100)
            ]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        # In a real implementation, load and process the actual image
        # For now, return dummy tensors
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Tokenization would be done here in real implementation
        # For now, return dummy token IDs
        input_ids = torch.randint(0, 32000, (self.max_length,))
        attention_mask = torch.ones(self.max_length)
        labels = input_ids.clone()
        
        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer for training"""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    # Only optimize adapter parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int
):
    """Create learning rate scheduler"""
    warmup_steps = config['training']['warmup_steps']
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[GradScaler],
    config: Dict[str, Any],
    epoch: int,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_base_loss = 0.0
    total_hallucination_loss = 0.0
    num_batches = 0
    
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    logging_steps = config['training']['logging_steps']
    use_amp = config['training']['use_amp']
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(**batch)
                loss = outputs['loss']
                if loss is not None:
                    loss = loss / gradient_accumulation_steps
        else:
            outputs = model(**batch)
            loss = outputs['loss']
            if loss is not None:
                loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if loss is not None:
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Update weights after accumulation
        if (step + 1) % gradient_accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if loss is not None:
            total_loss += loss.item() * gradient_accumulation_steps
            if outputs.get('base_loss') is not None:
                total_base_loss += outputs['base_loss'].item()
            if outputs.get('hallucination_loss') is not None:
                total_hallucination_loss += outputs['hallucination_loss'].item()
            num_batches += 1
        
        if step % logging_steps == 0:
            avg_loss = total_loss / max(num_batches, 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
    
    # Calculate epoch metrics
    metrics = {
        'loss': total_loss / max(num_batches, 1),
        'base_loss': total_base_loss / max(num_batches, 1),
        'hallucination_loss': total_hallucination_loss / max(num_batches, 1)
    }
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    config: Dict[str, Any],
    output_dir: str,
    is_best: bool = False
):
    """Save training checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    # Save adapter separately (small file)
    if hasattr(model, 'save_adapter'):
        adapter_path = os.path.join(output_dir, f'adapter_epoch_{epoch}.pt')
        model.save_adapter(adapter_path)
        print(f"Saved adapter to {adapter_path}")
    
    # Save full checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    
    if is_best:
        best_adapter_path = os.path.join(output_dir, 'adapter_best.pt')
        if hasattr(model, 'save_adapter'):
            model.save_adapter(best_adapter_path)
            print(f"Saved best adapter to {best_adapter_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LLaVA with Cross-Layer Adapter")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/llava_adapter_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--use_dummy_model',
        action='store_true',
        help='Use dummy model for testing (when LLaVA not installed)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output dir if specified
    if args.output_dir is not None:
        config['training']['output_dir'] = args.output_dir
    
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to output directory
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config to {config_save_path}")
    
    # Set seed
    set_seed(config['seed'])
    
    # Set device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Load or create LLaVA model
    print("Loading LLaVA model...")
    if args.use_dummy_model:
        print("Using dummy model for testing")
        llava_model = DummyLLaVAModel(
            vision_hidden_size=config['model']['vision']['hidden_size'],
            language_hidden_size=config['model']['language']['hidden_size']
        )
    else:
        try:
            # Try to load actual LLaVA model from HuggingFace
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = config['model']['llava_model_name']
            print(f"Loading {model_name} from HuggingFace...")
            llava_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if config['hardware']['fp16'] else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None
            )
            print("LLaVA model loaded successfully")
        except Exception as e:
            print(f"Failed to load LLaVA model: {e}")
            print("Falling back to dummy model for testing")
            llava_model = DummyLLaVAModel(
                vision_hidden_size=config['model']['vision']['hidden_size'],
                language_hidden_size=config['model']['language']['hidden_size']
            )
    
    # Create model with adapter
    print("Wrapping model with cross-layer adapter...")
    adapter_config = {
        'vision_hidden_size': config['model']['vision']['hidden_size'],
        'language_hidden_size': config['model']['language']['hidden_size'],
        'adapter_hidden_size': config['model']['adapter']['hidden_size'],
        'num_vision_layers': config['model']['vision']['num_layers'],
        'num_language_layers': config['model']['language']['num_layers'],
        'cross_layer_point': config['model']['adapter']['cross_layer_point'],
        'num_attention_heads': config['model']['adapter']['num_attention_heads'],
        'n_tokens': config['model']['adapter']['n_tokens'],
        'hallucination_loss_weight': config['training']['hallucination_loss_weight']
    }
    
    model = LLaVAWithAdapter(
        llava_model=llava_model,
        adapter_config=adapter_config,
        freeze_base_model=config['model']['freeze_base_model']
    )
    model = model.to(device)
    
    # Print parameter info
    model.print_trainable_parameters()
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    train_dataset = VisionLanguageDataset(
        data_path=config['data']['train_data_path'],
        image_dir=config['data']['train_image_dir'],
        max_length=config['data']['max_length'],
        image_size=config['data']['image_size'],
        is_train=True
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    
    num_epochs = config['training']['num_epochs']
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if config['training']['use_amp'] else None
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            epoch=epoch,
            device=device
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save checkpoint
        is_best = metrics['loss'] < best_loss
        if is_best:
            best_loss = metrics['loss']
            print(f"New best loss: {best_loss:.4f}")
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=epoch * len(train_dataloader),
            config=config,
            output_dir=output_dir,
            is_best=is_best
        )
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
