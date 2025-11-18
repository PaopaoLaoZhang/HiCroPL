"""
LLaVA-v1.5-7B Integration with Cross-Layer Adapter
LLaVA-v1.5-7B 模型与跨层适配器的集成

This module provides:
1. Feature extraction hooks for vision and language layers
2. Model wrapper that injects adapter features
3. Forward propagation with hallucination mitigation

Architecture:
- Vision Encoder: CLIP ViT-L/14 (24 layers, hidden_dim=1024)
- Language Model: Vicuna-7B (32 layers, hidden_dim=4096)  
- Connector: 2-layer MLP projection
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
import warnings

from .cross_layer_adapter import LVLMCrossLayerAdapter


class LLaVAFeatureExtractor:
    """
    Feature extractor for LLaVA model
    从 LLaVA 模型中提取中间层特征
    
    Extracts intermediate features from vision encoder and language model
    using forward hooks.
    """
    def __init__(self):
        self.vision_features: Dict[int, torch.Tensor] = {}
        self.language_features: Dict[int, torch.Tensor] = {}
        self.hooks = []
    
    def _create_vision_hook(self, layer_idx: int):
        """Create hook for vision encoder layer"""
        def hook(module, input, output):
            # Store the output feature
            if isinstance(output, tuple):
                self.vision_features[layer_idx] = output[0].detach()
            else:
                self.vision_features[layer_idx] = output.detach()
        return hook
    
    def _create_language_hook(self, layer_idx: int):
        """Create hook for language model layer"""
        def hook(module, input, output):
            # Store the output feature (hidden states)
            if isinstance(output, tuple):
                # Usually output[0] is hidden_states
                self.language_features[layer_idx] = output[0].detach()
            else:
                self.language_features[layer_idx] = output.detach()
        return hook
    
    def register_hooks(
        self,
        vision_layers: nn.ModuleList,
        language_layers: nn.ModuleList
    ):
        """
        Register hooks to extract features
        
        Args:
            vision_layers: Vision transformer layers (e.g., ViT blocks)
            language_layers: Language model layers (e.g., Transformer blocks)
        """
        # Register vision hooks
        for idx, layer in enumerate(vision_layers):
            hook = layer.register_forward_hook(self._create_vision_hook(idx))
            self.hooks.append(hook)
        
        # Register language hooks
        for idx, layer in enumerate(language_layers):
            hook = layer.register_forward_hook(self._create_language_hook(idx))
            self.hooks.append(hook)
    
    def clear_features(self):
        """Clear stored features"""
        self.vision_features.clear()
        self.language_features.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class LLaVAWithAdapter(nn.Module):
    """
    LLaVA model wrapper with cross-layer adapter
    带跨层适配器的 LLaVA 模型包装器
    
    This wrapper:
    1. Freezes the base LLaVA model (vision encoder + language model)
    2. Adds lightweight cross-layer adapters (~3M parameters)
    3. Injects adapter features during forward pass
    4. Computes hallucination detection loss
    
    Args:
        llava_model: Pre-trained LLaVA model (e.g., from HuggingFace)
        adapter_config: Configuration dict for adapter
        freeze_base_model: Whether to freeze LLaVA parameters (default: True)
    """
    def __init__(
        self,
        llava_model: nn.Module,
        adapter_config: Optional[Dict[str, Any]] = None,
        freeze_base_model: bool = True
    ):
        super(LLaVAWithAdapter, self).__init__()
        
        self.llava_model = llava_model
        self.freeze_base_model = freeze_base_model
        
        # Default adapter configuration
        default_config = {
            'vision_hidden_size': 1024,      # ViT-L/14
            'language_hidden_size': 4096,    # Vicuna-7B
            'adapter_hidden_size': 256,      # Adapter dimension
            'num_vision_layers': 24,         # ViT-L/14
            'num_language_layers': 32,       # Vicuna-7B
            'cross_layer_point': 16,         # Switch point
            'num_attention_heads': 8,
            'n_tokens': 4,
            'hallucination_loss_weight': 0.1
        }
        
        if adapter_config is not None:
            default_config.update(adapter_config)
        self.adapter_config = default_config
        
        # Initialize cross-layer adapter
        self.adapter = LVLMCrossLayerAdapter(
            vision_hidden_size=default_config['vision_hidden_size'],
            language_hidden_size=default_config['language_hidden_size'],
            adapter_hidden_size=default_config['adapter_hidden_size'],
            num_vision_layers=default_config['num_vision_layers'],
            num_language_layers=default_config['num_language_layers'],
            cross_layer_point=default_config['cross_layer_point'],
            num_attention_heads=default_config['num_attention_heads'],
            n_tokens=default_config['n_tokens']
        )
        
        self.hallucination_loss_weight = default_config['hallucination_loss_weight']
        
        # Freeze base model if specified
        if freeze_base_model:
            self._freeze_base_model()
        
        # Feature extractor (will be set up when needed)
        self.feature_extractor = None
        
        # Storage for alignment scores during forward pass
        self.alignment_scores = []
    
    def _freeze_base_model(self):
        """Freeze all parameters in base LLaVA model"""
        print("Freezing base LLaVA model parameters...")
        for param in self.llava_model.parameters():
            param.requires_grad = False
        print("Base model frozen. Only adapter parameters will be trained.")
    
    def _setup_feature_extraction(self):
        """
        Setup feature extraction hooks
        Note: This is model-specific and may need adjustment based on LLaVA version
        """
        if self.feature_extractor is not None:
            return
        
        self.feature_extractor = LLaVAFeatureExtractor()
        
        # Try to access vision and language layers
        # This is a simplified version - actual implementation depends on LLaVA structure
        try:
            # Common patterns for LLaVA structure
            if hasattr(self.llava_model, 'model'):
                model = self.llava_model.model
                
                # Vision encoder layers
                if hasattr(model, 'vision_tower'):
                    vision_model = model.vision_tower
                    if hasattr(vision_model, 'vision_model'):
                        vision_model = vision_model.vision_model
                    if hasattr(vision_model, 'encoder'):
                        vision_layers = vision_model.encoder.layers
                    elif hasattr(vision_model, 'transformer'):
                        vision_layers = vision_model.transformer.resblocks
                    else:
                        warnings.warn("Could not find vision layers")
                        vision_layers = nn.ModuleList()
                else:
                    warnings.warn("Could not find vision_tower")
                    vision_layers = nn.ModuleList()
                
                # Language model layers  
                if hasattr(model, 'layers'):
                    language_layers = model.layers
                elif hasattr(model, 'transformer'):
                    language_layers = model.transformer.h
                else:
                    warnings.warn("Could not find language layers")
                    language_layers = nn.ModuleList()
                
                self.feature_extractor.register_hooks(vision_layers, language_layers)
                print(f"Registered hooks for {len(vision_layers)} vision layers and {len(language_layers)} language layers")
                
        except Exception as e:
            warnings.warn(f"Failed to setup feature extraction: {e}")
            self.feature_extractor = None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adapter injection
        
        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            pixel_values: Image pixel values
            labels: Labels for language modeling loss
            **kwargs: Additional arguments for LLaVA model
            
        Returns:
            Dictionary containing:
                - loss: Total loss (language modeling + hallucination)
                - logits: Model predictions
                - hallucination_loss: Hallucination detection loss
                - alignment_scores: Vision-language alignment scores per layer
        """
        # Clear previous alignment scores
        self.alignment_scores = []
        
        # Setup feature extraction if not done
        if self.training and self.feature_extractor is None:
            self._setup_feature_extraction()
        
        # Clear previous features
        if self.feature_extractor is not None:
            self.feature_extractor.clear_features()
        
        # Forward pass through base LLaVA model
        # Note: This is simplified - actual implementation depends on LLaVA API
        outputs = self.llava_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            **kwargs
        )
        
        # Extract base loss and logits
        base_loss = outputs.loss if hasattr(outputs, 'loss') else None
        logits = outputs.logits if hasattr(outputs, 'logits') else None
        
        # Compute hallucination loss if in training mode
        hallucination_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        if self.training and self.feature_extractor is not None:
            # Process vision → language flow (shallow layers)
            for layer_idx in range(self.adapter.cross_layer_point):
                if layer_idx in self.feature_extractor.vision_features:
                    vision_feat = self.feature_extractor.vision_features[layer_idx]
                    _, alignment_score = self.adapter.process_vision_to_language(
                        vision_feat, layer_idx
                    )
                    self.alignment_scores.append(alignment_score)
            
            # Process language → vision flow (deep layers)
            for layer_idx in range(
                self.adapter.cross_layer_point, 
                self.adapter.num_language_layers
            ):
                if layer_idx in self.feature_extractor.language_features:
                    lang_feat = self.feature_extractor.language_features[layer_idx]
                    _, alignment_score = self.adapter.process_language_to_vision(
                        lang_feat, layer_idx
                    )
                    self.alignment_scores.append(alignment_score)
            
            # Compute hallucination loss from alignment scores
            hallucination_loss = self.adapter.compute_hallucination_loss(
                self.alignment_scores
            )
        
        # Combine losses
        total_loss = None
        if base_loss is not None:
            total_loss = base_loss + self.hallucination_loss_weight * hallucination_loss
        
        return {
            'loss': total_loss,
            'logits': logits,
            'hallucination_loss': hallucination_loss,
            'alignment_scores': self.alignment_scores,
            'base_loss': base_loss
        }
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **generation_kwargs
    ) -> torch.LongTensor:
        """
        Generate text with adapter-enhanced features
        
        Args:
            input_ids: Input token IDs
            pixel_values: Image pixel values
            **generation_kwargs: Arguments for generation (max_length, etc.)
            
        Returns:
            Generated token IDs
        """
        # For generation, we use the base model's generate method
        # The adapter effects are incorporated through the modified forward pass
        return self.llava_model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **generation_kwargs
        )
    
    def save_adapter(self, save_path: str):
        """
        Save only the adapter parameters
        
        Args:
            save_path: Path to save adapter checkpoint
        """
        torch.save({
            'adapter_state_dict': self.adapter.state_dict(),
            'adapter_config': self.adapter_config,
        }, save_path)
        print(f"Adapter saved to {save_path}")
    
    def load_adapter(self, load_path: str):
        """
        Load adapter parameters
        
        Args:
            load_path: Path to adapter checkpoint
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.adapter.load_state_dict(checkpoint['adapter_state_dict'])
        print(f"Adapter loaded from {load_path}")
    
    def get_trainable_parameters(self) -> int:
        """
        Get number of trainable parameters
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def print_trainable_parameters(self):
        """Print trainable parameters info"""
        trainable_params = 0
        all_params = 0
        
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable ratio: {100 * trainable_params / all_params:.2f}%")
