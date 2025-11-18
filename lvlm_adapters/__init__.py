"""
LLaVA Cross-Layer Adapters for Hallucination Mitigation
Based on HiCroPL's bidirectional knowledge flow mechanism
"""

from .cross_layer_adapter import (
    AttentionPooling,
    CrossLayerKnowledgeMapper,
    LVLMCrossLayerAdapter
)
from .llava_with_adapter import LLaVAWithAdapter

__all__ = [
    'AttentionPooling',
    'CrossLayerKnowledgeMapper', 
    'LVLMCrossLayerAdapter',
    'LLaVAWithAdapter'
]
