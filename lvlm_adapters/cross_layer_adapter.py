"""
Cross-Layer Adapter Module for LVLMs (Large Vision-Language Models)
基于 HiCroPL 的跨层知识流通机制

This module implements:
1. AttentionPooling (LKP - Layer-wise Knowledge Pooling)
2. CrossLayerKnowledgeMapper for bidirectional knowledge flow
3. LVLMCrossLayerAdapter as the main adapter class

Architecture:
- Shallow layers (1-16): Vision → Language knowledge flow (prevent hallucination)
- Deep layers (17-32): Language → Vision knowledge flow (enhance understanding)
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Optional, Tuple


class QuickGELU(nn.Module):
    """Quick GELU activation (used in CLIP)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class AttentionPooling(nn.Module):
    """
    LKP: Layer-wise Knowledge Pooling
    层级知识池化模块 - 将序列特征压缩为代理token
    
    This module compresses sequence features into a proxy token using attention mechanism.
    Used to create lightweight representations for efficient cross-modal communication.
    
    Args:
        hidden_size: Dimension of the hidden features
        num_attention_heads: Number of attention heads
    """
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super(AttentionPooling, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_attention_heads,
            batch_first=False
        )
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(
        self, 
        token_query: torch.Tensor,      # [1, hidden_size] or [batch, 1, hidden_size]
        sequence_key: torch.Tensor,     # [seq_len, hidden_size] or [batch, seq_len, hidden_size]
        sequence_value: torch.Tensor    # [seq_len, hidden_size] or [batch, seq_len, hidden_size]
    ) -> torch.Tensor:
        """
        Compress sequence into a single proxy token via attention pooling
        
        Args:
            token_query: Proxy token to be refined
            sequence_key: Key sequence for attention
            sequence_value: Value sequence for attention
            
        Returns:
            Refined proxy token with same shape as token_query
        """
        # Apply layer norm and attention
        normed_query = self.ln_1(token_query)
        normed_key = self.ln_1(sequence_key)
        normed_value = self.ln_1(sequence_value)
        
        attn_output, _ = self.attn(
            normed_query, 
            normed_key, 
            normed_value, 
            need_weights=False
        )
        
        # Residual connection + layer norm
        token_query = token_query + attn_output
        token_query = self.ln_2(token_query)
        
        return token_query


class CrossLayerKnowledgeMapper(nn.Module):
    """
    Multi-scale Knowledge Mapper
    跨层知识映射器 - 在不同模态间传递知识
    
    Maps knowledge from one modality to another with dimension adaptation.
    Implements cross-attention with FFN for deep feature transformation.
    
    Args:
        target_dim: Target modality dimension (query dimension)
        source_dim: Source modality dimension (key/value dimension)
        num_attention_heads: Number of attention heads
    """
    def __init__(
        self, 
        target_dim: int, 
        source_dim: int, 
        num_attention_heads: int
    ):
        super(CrossLayerKnowledgeMapper, self).__init__()
        
        # Cross-attention components
        self.attn = nn.MultiheadAttention(
            embed_dim=target_dim, 
            num_heads=num_attention_heads,
            batch_first=False
        )
        
        # Projection layers for dimension alignment
        self.linear_q = nn.Linear(target_dim, target_dim)
        self.linear_k = nn.Linear(source_dim, target_dim)
        self.linear_v = nn.Linear(source_dim, target_dim)
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(target_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(target_dim, target_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(target_dim * 4, target_dim))
        ]))
        self.ln_2 = nn.LayerNorm(target_dim)

    def forward(
        self, 
        target: torch.Tensor,   # Target modality features
        source: torch.Tensor,   # Source modality features  
    ) -> torch.Tensor:
        """
        Map knowledge from source modality to target modality
        
        Args:
            target: Target modality features [seq_len, target_dim] or [batch, seq_len, target_dim]
            source: Source modality features [seq_len, source_dim] or [batch, seq_len, source_dim]
            
        Returns:
            Updated target features with knowledge from source
        """
        # Project to aligned space
        q_proj = self.linear_q(target)
        k_proj = self.linear_k(source)
        v_proj = self.linear_v(source)
        
        # Cross-attention with residual
        normed_q = self.ln_1(q_proj)
        normed_k = self.ln_1(k_proj)
        normed_v = self.ln_1(v_proj)
        
        attn_output, _ = self.attn(normed_q, normed_k, normed_v, need_weights=False)
        q_proj = q_proj + attn_output
        
        # FFN with residual
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        
        return q_proj


class LVLMCrossLayerAdapter(nn.Module):
    """
    Main Cross-Layer Adapter for LVLMs
    大型视觉语言模型的跨层适配器
    
    Implements bidirectional knowledge flow between vision and language modalities:
    - Shallow layers: Vision → Language (visual grounding to prevent hallucination)
    - Deep layers: Language → Vision (semantic enhancement for better understanding)
    
    Args:
        vision_hidden_size: Vision encoder hidden dimension (e.g., 1024 for ViT-L/14)
        language_hidden_size: Language model hidden dimension (e.g., 4096 for Vicuna-7B)
        adapter_hidden_size: Adapter internal dimension (for efficiency, e.g., 256)
        num_vision_layers: Total number of vision layers (e.g., 24)
        num_language_layers: Total number of language layers (e.g., 32)
        cross_layer_point: Layer index to switch flow direction (e.g., 16)
        num_attention_heads: Number of attention heads (e.g., 8)
        n_tokens: Number of adapter tokens per layer (e.g., 4)
    """
    def __init__(
        self,
        vision_hidden_size: int = 1024,      # ViT-L/14 hidden size
        language_hidden_size: int = 4096,    # Vicuna-7B hidden size  
        adapter_hidden_size: int = 256,      # Adapter dimension
        num_vision_layers: int = 24,         # ViT-L/14 layers
        num_language_layers: int = 32,       # Vicuna-7B layers
        cross_layer_point: int = 16,         # Switch point for bidirectional flow
        num_attention_heads: int = 8,
        n_tokens: int = 4,
        dtype: torch.dtype = torch.float32
    ):
        super(LVLMCrossLayerAdapter, self).__init__()
        
        self.vision_hidden_size = vision_hidden_size
        self.language_hidden_size = language_hidden_size
        self.adapter_hidden_size = adapter_hidden_size
        self.num_vision_layers = num_vision_layers
        self.num_language_layers = num_language_layers
        self.cross_layer_point = cross_layer_point
        self.n_tokens = n_tokens
        self.dtype = dtype
        
        # Down-projection layers (to adapter dimension)
        self.vision_down_proj = nn.Linear(vision_hidden_size, adapter_hidden_size)
        self.language_down_proj = nn.Linear(language_hidden_size, adapter_hidden_size)
        
        # Up-projection layers (back to original dimension)
        self.vision_up_proj = nn.Linear(adapter_hidden_size, vision_hidden_size)
        self.language_up_proj = nn.Linear(adapter_hidden_size, language_hidden_size)
        
        # ===== Vision → Language pathway (layers 0 to cross_layer_point) =====
        # This prevents hallucination by grounding language generation in visual content
        
        # Vision adapter tokens (learnable parameters)
        self.vision_adapter_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(n_tokens, adapter_hidden_size, dtype=dtype))
            for _ in range(cross_layer_point)
        ])
        
        # Vision proxy tokens for LKP compression
        self.vision_proxy_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, adapter_hidden_size, dtype=dtype))
            for _ in range(cross_layer_point)
        ])
        
        # LKP networks for vision features
        self.vision_lkp_nets = nn.ModuleList([
            AttentionPooling(adapter_hidden_size, num_attention_heads)
            for _ in range(cross_layer_point)
        ])
        
        # Vision → Language knowledge mappers
        self.vision2language_mappers = nn.ModuleList([
            CrossLayerKnowledgeMapper(
                target_dim=adapter_hidden_size,
                source_dim=adapter_hidden_size,
                num_attention_heads=num_attention_heads
            )
            for _ in range(cross_layer_point)
        ])
        
        # ===== Language → Vision pathway (layers cross_layer_point to num_language_layers) =====
        # This enhances visual understanding with semantic information
        
        # Language adapter tokens
        remaining_layers = num_language_layers - cross_layer_point
        self.language_adapter_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(n_tokens, adapter_hidden_size, dtype=dtype))
            for _ in range(remaining_layers)
        ])
        
        # Language proxy tokens for LKP compression
        self.language_proxy_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, adapter_hidden_size, dtype=dtype))
            for _ in range(remaining_layers)
        ])
        
        # LKP networks for language features
        self.language_lkp_nets = nn.ModuleList([
            AttentionPooling(adapter_hidden_size, num_attention_heads)
            for _ in range(remaining_layers)
        ])
        
        # Language → Vision knowledge mappers
        self.language2vision_mappers = nn.ModuleList([
            CrossLayerKnowledgeMapper(
                target_dim=adapter_hidden_size,
                source_dim=adapter_hidden_size,
                num_attention_heads=num_attention_heads
            )
            for _ in range(remaining_layers)
        ])
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize learnable parameters with small random values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_vision_adapter_tokens(
        self, 
        layer_idx: int, 
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Get vision adapter tokens for a specific layer
        
        Args:
            layer_idx: Layer index (0-indexed)
            batch_size: Batch size for expansion
            
        Returns:
            Adapter tokens [batch_size, n_tokens, adapter_hidden_size]
        """
        if layer_idx >= self.cross_layer_point:
            return None
        
        tokens = self.vision_adapter_tokens[layer_idx]
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_language_adapter_tokens(
        self, 
        layer_idx: int, 
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Get language adapter tokens for a specific layer
        
        Args:
            layer_idx: Layer index (0-indexed)
            batch_size: Batch size for expansion
            
        Returns:
            Adapter tokens [batch_size, n_tokens, adapter_hidden_size]
        """
        if layer_idx < self.cross_layer_point:
            return None
        
        adjusted_idx = layer_idx - self.cross_layer_point
        if adjusted_idx >= len(self.language_adapter_tokens):
            return None
            
        tokens = self.language_adapter_tokens[adjusted_idx]
        return tokens.unsqueeze(0).expand(batch_size, -1, -1)
    
    def process_vision_to_language(
        self,
        vision_features: torch.Tensor,  # [batch, seq_len, vision_hidden_size]
        layer_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Process Vision → Language knowledge flow for shallow layers
        
        Args:
            vision_features: Vision features from encoder
            layer_idx: Current layer index
            
        Returns:
            language_adapter_output: Adapter features to inject into language model
            alignment_score: Vision-language alignment score (for hallucination detection)
        """
        if layer_idx >= self.cross_layer_point:
            return None, 0.0
        
        batch_size = vision_features.shape[0]
        
        # Project to adapter dimension
        vision_feat_proj = self.vision_down_proj(vision_features)  # [batch, seq_len, adapter_dim]
        
        # LKP: Compress vision features to proxy token
        proxy_token = self.vision_proxy_tokens[layer_idx].unsqueeze(0).expand(batch_size, -1, -1)
        # Reshape for attention: [batch, seq_len, dim] -> [seq_len, batch, dim]
        vision_feat_proj_t = vision_feat_proj.transpose(0, 1)
        proxy_token_t = proxy_token.transpose(0, 1)
        
        compressed_vision = self.vision_lkp_nets[layer_idx](
            token_query=proxy_token_t,
            sequence_key=vision_feat_proj_t,
            sequence_value=vision_feat_proj_t
        )  # [1, batch, adapter_dim]
        compressed_vision = compressed_vision.transpose(0, 1)  # [batch, 1, adapter_dim]
        
        # Get language adapter tokens
        lang_adapter_tokens = self.vision_adapter_tokens[layer_idx].unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, n_tokens, adapter_dim]
        
        # Map vision knowledge to language adapter tokens
        lang_adapter_tokens_t = lang_adapter_tokens.transpose(0, 1)  # [n_tokens, batch, adapter_dim]
        compressed_vision_expanded = compressed_vision.expand(-1, self.n_tokens, -1).transpose(0, 1)
        
        updated_tokens = self.vision2language_mappers[layer_idx](
            target=lang_adapter_tokens_t,
            source=compressed_vision_expanded
        )  # [n_tokens, batch, adapter_dim]
        updated_tokens = updated_tokens.transpose(0, 1)  # [batch, n_tokens, adapter_dim]
        
        # Project back to language dimension
        language_adapter_output = self.language_up_proj(updated_tokens)
        
        # Compute alignment score (cosine similarity)
        with torch.no_grad():
            vision_norm = compressed_vision / (compressed_vision.norm(dim=-1, keepdim=True) + 1e-8)
            lang_norm = updated_tokens.mean(dim=1, keepdim=True)
            lang_norm = lang_norm / (lang_norm.norm(dim=-1, keepdim=True) + 1e-8)
            alignment_score = (vision_norm * lang_norm).sum(dim=-1).mean().item()
        
        return language_adapter_output, alignment_score
    
    def process_language_to_vision(
        self,
        language_features: torch.Tensor,  # [batch, seq_len, language_hidden_size]
        layer_idx: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Process Language → Vision knowledge flow for deep layers
        
        Args:
            language_features: Language features from model
            layer_idx: Current layer index
            
        Returns:
            vision_adapter_output: Adapter features to inject into vision
            alignment_score: Language-vision alignment score
        """
        if layer_idx < self.cross_layer_point:
            return None, 0.0
        
        adjusted_idx = layer_idx - self.cross_layer_point
        if adjusted_idx >= len(self.language_adapter_tokens):
            return None, 0.0
        
        batch_size = language_features.shape[0]
        
        # Project to adapter dimension
        lang_feat_proj = self.language_down_proj(language_features)  # [batch, seq_len, adapter_dim]
        
        # LKP: Compress language features to proxy token
        proxy_token = self.language_proxy_tokens[adjusted_idx].unsqueeze(0).expand(batch_size, -1, -1)
        lang_feat_proj_t = lang_feat_proj.transpose(0, 1)
        proxy_token_t = proxy_token.transpose(0, 1)
        
        compressed_language = self.language_lkp_nets[adjusted_idx](
            token_query=proxy_token_t,
            sequence_key=lang_feat_proj_t,
            sequence_value=lang_feat_proj_t
        )  # [1, batch, adapter_dim]
        compressed_language = compressed_language.transpose(0, 1)  # [batch, 1, adapter_dim]
        
        # Get vision adapter tokens
        vision_adapter_tokens = self.language_adapter_tokens[adjusted_idx].unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, n_tokens, adapter_dim]
        
        # Map language knowledge to vision adapter tokens
        vision_adapter_tokens_t = vision_adapter_tokens.transpose(0, 1)
        compressed_language_expanded = compressed_language.expand(-1, self.n_tokens, -1).transpose(0, 1)
        
        updated_tokens = self.language2vision_mappers[adjusted_idx](
            target=vision_adapter_tokens_t,
            source=compressed_language_expanded
        )  # [n_tokens, batch, adapter_dim]
        updated_tokens = updated_tokens.transpose(0, 1)  # [batch, n_tokens, adapter_dim]
        
        # Project back to vision dimension
        vision_adapter_output = self.vision_up_proj(updated_tokens)
        
        # Compute alignment score
        with torch.no_grad():
            lang_norm = compressed_language / (compressed_language.norm(dim=-1, keepdim=True) + 1e-8)
            vision_norm = updated_tokens.mean(dim=1, keepdim=True)
            vision_norm = vision_norm / (vision_norm.norm(dim=-1, keepdim=True) + 1e-8)
            alignment_score = (lang_norm * vision_norm).sum(dim=-1).mean().item()
        
        return vision_adapter_output, alignment_score
    
    def compute_hallucination_loss(
        self,
        alignment_scores: List[float],
        target_alignment: float = 0.8
    ) -> torch.Tensor:
        """
        Compute hallucination penalty based on alignment scores
        
        Lower alignment scores indicate potential hallucination (language drifting from vision)
        
        Args:
            alignment_scores: List of alignment scores from each layer
            target_alignment: Target alignment score (default: 0.8)
            
        Returns:
            Hallucination loss tensor
        """
        if not alignment_scores:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        scores_tensor = torch.tensor(alignment_scores, device=next(self.parameters()).device)
        # Penalize low alignment (potential hallucination)
        loss = torch.clamp(target_alignment - scores_tensor, min=0.0).mean()
        return loss
