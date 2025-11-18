#!/usr/bin/env python3
"""
Simple test script for validating the cross-layer adapter implementation
验证跨层适配器实现的测试脚本

Usage:
    python scripts/test_adapter.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from lvlm_adapters import (
            AttentionPooling,
            CrossLayerKnowledgeMapper,
            LVLMCrossLayerAdapter,
            LLaVAWithAdapter
        )
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_attention_pooling():
    """Test AttentionPooling module"""
    print("\nTesting AttentionPooling...")
    try:
        import torch
        from lvlm_adapters import AttentionPooling
        
        # Create module
        pooling = AttentionPooling(hidden_size=256, num_attention_heads=8)
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        token_query = torch.randn(1, batch_size, 256)
        sequence = torch.randn(seq_len, batch_size, 256)
        
        output = pooling(token_query, sequence, sequence)
        
        assert output.shape == token_query.shape, f"Output shape mismatch: {output.shape}"
        print(f"✓ AttentionPooling works correctly (output shape: {output.shape})")
        return True
    except Exception as e:
        print(f"✗ AttentionPooling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_mapper():
    """Test CrossLayerKnowledgeMapper module"""
    print("\nTesting CrossLayerKnowledgeMapper...")
    try:
        import torch
        from lvlm_adapters import CrossLayerKnowledgeMapper
        
        # Create mapper (vision to language)
        mapper = CrossLayerKnowledgeMapper(
            target_dim=512,
            source_dim=768,
            num_attention_heads=8
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        target = torch.randn(seq_len, batch_size, 512)
        source = torch.randn(seq_len, batch_size, 768)
        
        output = mapper(target, source)
        
        assert output.shape == target.shape, f"Output shape mismatch: {output.shape}"
        print(f"✓ CrossLayerKnowledgeMapper works correctly (output shape: {output.shape})")
        return True
    except Exception as e:
        print(f"✗ CrossLayerKnowledgeMapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter():
    """Test LVLMCrossLayerAdapter module"""
    print("\nTesting LVLMCrossLayerAdapter...")
    try:
        import torch
        from lvlm_adapters import LVLMCrossLayerAdapter
        
        # Create adapter
        adapter = LVLMCrossLayerAdapter(
            vision_hidden_size=1024,
            language_hidden_size=4096,
            adapter_hidden_size=256,
            num_vision_layers=24,
            num_language_layers=32,
            cross_layer_point=16
        )
        
        num_params = sum(p.numel() for p in adapter.parameters())
        print(f"  Adapter has {num_params:,} parameters")
        
        # Test vision to language flow
        batch_size = 2
        seq_len = 10
        vision_feat = torch.randn(batch_size, seq_len, 1024)
        
        for layer_idx in [0, 8, 15]:
            output, score = adapter.process_vision_to_language(vision_feat, layer_idx)
            assert output is not None, f"Output should not be None for layer {layer_idx}"
            assert output.shape[0] == batch_size, "Batch size mismatch"
            print(f"  ✓ Vision→Language at layer {layer_idx}: output {output.shape}, score {score:.4f}")
        
        # Test language to vision flow
        lang_feat = torch.randn(batch_size, seq_len, 4096)
        
        for layer_idx in [16, 24, 31]:
            output, score = adapter.process_language_to_vision(lang_feat, layer_idx)
            assert output is not None, f"Output should not be None for layer {layer_idx}"
            assert output.shape[0] == batch_size, "Batch size mismatch"
            print(f"  ✓ Language→Vision at layer {layer_idx}: output {output.shape}, score {score:.4f}")
        
        # Test hallucination loss
        alignment_scores = [0.75, 0.82, 0.68, 0.91, 0.77]
        loss = adapter.compute_hallucination_loss(alignment_scores, target_alignment=0.8)
        print(f"  ✓ Hallucination loss computed: {loss.item():.4f}")
        
        print("✓ LVLMCrossLayerAdapter works correctly")
        return True
    except Exception as e:
        print(f"✗ LVLMCrossLayerAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llava_wrapper():
    """Test LLaVAWithAdapter wrapper"""
    print("\nTesting LLaVAWithAdapter...")
    try:
        import torch
        import torch.nn as nn
        from lvlm_adapters import LLaVAWithAdapter
        
        # Create dummy LLaVA model
        class DummyLLaVA(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Linear(10, 10)
            
            def forward(self, input_ids=None, pixel_values=None, labels=None, **kwargs):
                class DummyOutput:
                    def __init__(self):
                        self.loss = torch.tensor(1.0, requires_grad=True)
                        self.logits = torch.randn(2, 10, 100)
                return DummyOutput()
        
        dummy_llava = DummyLLaVA()
        
        # Wrap with adapter
        model = LLaVAWithAdapter(
            llava_model=dummy_llava,
            adapter_config={
                'vision_hidden_size': 1024,
                'language_hidden_size': 4096,
                'adapter_hidden_size': 256,
                'num_vision_layers': 24,
                'num_language_layers': 32,
                'cross_layer_point': 16,
                'hallucination_loss_weight': 0.1
            },
            freeze_base_model=True
        )
        
        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
        
        # Test forward pass
        batch_size = 2
        outputs = model(
            input_ids=torch.randint(0, 1000, (batch_size, 10)),
            pixel_values=torch.randn(batch_size, 3, 224, 224),
            labels=torch.randint(0, 1000, (batch_size, 10))
        )
        
        assert 'loss' in outputs, "Output should contain 'loss'"
        assert 'hallucination_loss' in outputs, "Output should contain 'hallucination_loss'"
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Base loss: {outputs['base_loss'].item():.4f}")
        print(f"  ✓ Hallucination loss: {outputs['hallucination_loss'].item():.4f}")
        
        # Test save/load adapter
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        model.save_adapter(temp_path)
        print(f"  ✓ Adapter saved to {temp_path}")
        
        model.load_adapter(temp_path)
        print(f"  ✓ Adapter loaded from {temp_path}")
        
        import os
        os.unlink(temp_path)
        
        print("✓ LLaVAWithAdapter works correctly")
        return True
    except Exception as e:
        print(f"✗ LLaVAWithAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Cross-Layer Adapter Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("AttentionPooling", test_attention_pooling()))
    results.append(("CrossLayerKnowledgeMapper", test_knowledge_mapper()))
    results.append(("LVLMCrossLayerAdapter", test_adapter()))
    results.append(("LLaVAWithAdapter", test_llava_wrapper()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
