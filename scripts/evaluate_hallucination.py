#!/usr/bin/env python3
"""
Evaluation Script for Hallucination Metrics
幻觉指标评估脚本

This script evaluates:
1. CHAIR (Caption Hallucination Assessment with Image Relevance)
   - CHAIRs: Sentence-level hallucination rate
   - CHAIRi: Instance-level hallucination rate
2. POPE (Polling-based Object Probing Evaluation)
   - Accuracy, Precision, Recall, F1
3. Alignment Scores (Vision-Language alignment per layer)

Usage:
    python scripts/evaluate_hallucination.py --config configs/llava_adapter_config.yaml --checkpoint outputs/llava_adapter/adapter_best.pt
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lvlm_adapters import LLaVAWithAdapter


class CHAIREvaluator:
    """
    CHAIR (Caption Hallucination Assessment with Image Relevance) Evaluator
    CHAIR 幻觉评估器
    
    Evaluates object hallucination in image captions by comparing
    generated objects with ground truth COCO object annotations.
    """
    def __init__(self, coco_instances_path: str):
        """
        Args:
            coco_instances_path: Path to COCO instances annotations
        """
        self.coco_instances_path = coco_instances_path
        self.image_id_to_objects = {}
        
        if coco_instances_path and os.path.exists(coco_instances_path):
            self._load_coco_objects()
        else:
            print("Warning: COCO instances file not found, using dummy data")
            self._create_dummy_objects()
    
    def _load_coco_objects(self):
        """Load COCO object annotations"""
        print(f"Loading COCO instances from {self.coco_instances_path}")
        with open(self.coco_instances_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build category mapping
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Build image_id to objects mapping
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            category_name = self.categories[ann['category_id']]
            
            if image_id not in self.image_id_to_objects:
                self.image_id_to_objects[image_id] = set()
            self.image_id_to_objects[image_id].add(category_name)
        
        print(f"Loaded objects for {len(self.image_id_to_objects)} images")
    
    def _create_dummy_objects(self):
        """Create dummy objects for testing"""
        dummy_objects = ['person', 'car', 'dog', 'cat', 'tree', 'building']
        for i in range(100):
            self.image_id_to_objects[i] = set(np.random.choice(
                dummy_objects, size=3, replace=False
            ))
    
    def extract_objects(self, caption: str) -> List[str]:
        """
        Extract object nouns from caption
        (Simplified version - in practice, use spaCy or similar)
        """
        # This is a simplified version. In real implementation, use NLP tools
        # to extract nouns properly
        words = caption.lower().split()
        
        # Simple noun list (in practice, use POS tagging)
        common_objects = [
            'person', 'people', 'man', 'woman', 'child', 'boy', 'girl',
            'car', 'truck', 'bus', 'motorcycle', 'bicycle',
            'dog', 'cat', 'horse', 'bird', 'sheep', 'cow',
            'tree', 'building', 'house', 'road', 'sky', 'water',
            'table', 'chair', 'sofa', 'bed', 'laptop', 'phone'
        ]
        
        extracted = [word for word in words if word in common_objects]
        return list(set(extracted))  # Remove duplicates
    
    def evaluate(
        self, 
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute CHAIR metrics
        
        Args:
            predictions: List of dicts with keys:
                - 'image_id': int
                - 'caption': str (generated caption)
        
        Returns:
            Dictionary with CHAIRs and CHAIRi scores
        """
        num_hallucinated_captions = 0
        total_hallucinated_objects = 0
        total_generated_objects = 0
        total_captions = len(predictions)
        
        for pred in predictions:
            image_id = pred['image_id']
            caption = pred['caption']
            
            # Get ground truth objects
            gt_objects = self.image_id_to_objects.get(image_id, set())
            
            # Extract generated objects
            gen_objects = self.extract_objects(caption)
            
            if len(gen_objects) == 0:
                continue
            
            # Count hallucinations
            hallucinated_objects = [obj for obj in gen_objects if obj not in gt_objects]
            
            if len(hallucinated_objects) > 0:
                num_hallucinated_captions += 1
            
            total_hallucinated_objects += len(hallucinated_objects)
            total_generated_objects += len(gen_objects)
        
        # Compute metrics
        chair_s = num_hallucinated_captions / max(total_captions, 1) * 100
        chair_i = total_hallucinated_objects / max(total_generated_objects, 1) * 100
        
        return {
            'CHAIRs': chair_s,  # Sentence-level: % of captions with hallucinations
            'CHAIRi': chair_i,  # Instance-level: % of hallucinated objects
            'num_hallucinated_captions': num_hallucinated_captions,
            'total_captions': total_captions,
            'total_hallucinated_objects': total_hallucinated_objects,
            'total_generated_objects': total_generated_objects
        }


class POPEEvaluator:
    """
    POPE (Polling-based Object Probing Evaluation) Evaluator
    POPE 评估器
    
    Evaluates model's ability to correctly identify presence/absence
    of objects in images using yes/no questions.
    """
    def __init__(self, pope_data_path: str):
        """
        Args:
            pope_data_path: Path to POPE evaluation data
        """
        self.pope_data_path = pope_data_path
        self.questions = []
        
        if pope_data_path and os.path.exists(pope_data_path):
            self._load_pope_data()
        else:
            print("Warning: POPE data file not found, using dummy data")
            self._create_dummy_pope_data()
    
    def _load_pope_data(self):
        """Load POPE evaluation data"""
        print(f"Loading POPE data from {self.pope_data_path}")
        with open(self.pope_data_path, 'r') as f:
            self.questions = json.load(f)
        print(f"Loaded {len(self.questions)} POPE questions")
    
    def _create_dummy_pope_data(self):
        """Create dummy POPE data for testing"""
        objects = ['person', 'car', 'dog', 'cat', 'tree']
        for i in range(100):
            obj = np.random.choice(objects)
            answer = np.random.choice(['yes', 'no'])
            self.questions.append({
                'image_id': i,
                'question': f'Is there a {obj} in the image?',
                'answer': answer
            })
    
    def evaluate(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute POPE metrics
        
        Args:
            predictions: List of dicts with keys:
                - 'question_id': int
                - 'answer': str ('yes' or 'no')
        
        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for i, pred in enumerate(predictions):
            if i >= len(self.questions):
                break
            
            gt_answer = self.questions[i]['answer'].lower()
            pred_answer = pred['answer'].lower()
            
            # Convert to binary (yes=1, no=0)
            if gt_answer == 'yes' and pred_answer == 'yes':
                true_positives += 1
            elif gt_answer == 'no' and pred_answer == 'no':
                true_negatives += 1
            elif gt_answer == 'no' and pred_answer == 'yes':
                false_positives += 1
            elif gt_answer == 'yes' and pred_answer == 'no':
                false_negatives += 1
        
        # Compute metrics
        accuracy = (true_positives + true_negatives) / len(predictions) * 100
        precision = true_positives / max(true_positives + false_positives, 1) * 100
        recall = true_positives / max(true_positives + false_negatives, 1) * 100
        f1 = 2 * (precision * recall) / max(precision + recall, 1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }


def visualize_alignment_scores(
    alignment_scores_per_layer: List[float],
    output_path: str,
    cross_layer_point: int = 16
):
    """
    Visualize vision-language alignment scores across layers
    
    Args:
        alignment_scores_per_layer: Alignment scores for each layer
        output_path: Path to save visualization
        cross_layer_point: Layer where knowledge flow switches direction
    """
    plt.figure(figsize=(12, 6))
    
    layers = list(range(len(alignment_scores_per_layer)))
    scores = alignment_scores_per_layer
    
    # Plot alignment scores
    plt.plot(layers, scores, marker='o', linewidth=2, markersize=6)
    
    # Mark the cross-layer point
    plt.axvline(x=cross_layer_point, color='r', linestyle='--', 
                label=f'Cross-layer point (layer {cross_layer_point})')
    
    # Add regions
    plt.axvspan(0, cross_layer_point, alpha=0.1, color='blue', 
                label='Vision → Language')
    plt.axvspan(cross_layer_point, len(layers), alpha=0.1, color='green',
                label='Language → Vision')
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Alignment Score', fontsize=12)
    plt.title('Vision-Language Alignment Scores Across Layers', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Alignment score visualization saved to {output_path}")
    plt.close()


def run_evaluation(
    model,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str
):
    """
    Run full evaluation suite
    
    Args:
        model: Model with adapter
        config: Configuration dictionary
        device: Computing device
        output_dir: Directory to save results
    """
    print("\n" + "="*50)
    print("Starting Evaluation")
    print("="*50)
    
    results = {}
    
    # Generate dummy predictions for testing
    # In real implementation, this would run model inference
    print("\nGenerating predictions (using dummy data for testing)...")
    dummy_predictions_chair = [
        {'image_id': i, 'caption': f'A person with a car and a dog'}
        for i in range(100)
    ]
    
    dummy_predictions_pope = [
        {'question_id': i, 'answer': np.random.choice(['yes', 'no'])}
        for i in range(100)
    ]
    
    # CHAIR Evaluation
    if config['evaluation']['compute_chair']:
        print("\n" + "-"*50)
        print("Evaluating CHAIR (Caption Hallucination)")
        print("-"*50)
        
        chair_evaluator = CHAIREvaluator(
            coco_instances_path=config['evaluation']['chair']['coco_instances_path']
        )
        chair_results = chair_evaluator.evaluate(dummy_predictions_chair)
        results['chair'] = chair_results
        
        print("\nCHAIR Results:")
        print(f"  CHAIRs (Sentence-level): {chair_results['CHAIRs']:.2f}%")
        print(f"  CHAIRi (Instance-level): {chair_results['CHAIRi']:.2f}%")
        print(f"  Hallucinated captions: {chair_results['num_hallucinated_captions']}/{chair_results['total_captions']}")
        print(f"  Hallucinated objects: {chair_results['total_hallucinated_objects']}/{chair_results['total_generated_objects']}")
    
    # POPE Evaluation
    if config['evaluation']['compute_pope']:
        print("\n" + "-"*50)
        print("Evaluating POPE (Object Probing)")
        print("-"*50)
        
        pope_evaluator = POPEEvaluator(
            pope_data_path=config['evaluation']['pope']['pope_data_path']
        )
        pope_results = pope_evaluator.evaluate(dummy_predictions_pope)
        results['pope'] = pope_results
        
        print("\nPOPE Results:")
        print(f"  Accuracy: {pope_results['accuracy']:.2f}%")
        print(f"  Precision: {pope_results['precision']:.2f}%")
        print(f"  Recall: {pope_results['recall']:.2f}%")
        print(f"  F1 Score: {pope_results['f1']:.2f}")
    
    # Alignment Scores Visualization
    if config['evaluation']['compute_alignment_scores']:
        print("\n" + "-"*50)
        print("Computing Alignment Scores")
        print("-"*50)
        
        # Generate dummy alignment scores (in practice, extract from model)
        num_layers = (config['model']['adapter']['cross_layer_point'] + 
                     (config['model']['language']['num_layers'] - 
                      config['model']['adapter']['cross_layer_point']))
        dummy_alignment_scores = np.random.uniform(0.6, 0.9, num_layers).tolist()
        
        results['alignment_scores'] = dummy_alignment_scores
        
        print(f"\nAverage alignment score: {np.mean(dummy_alignment_scores):.4f}")
        print(f"Min alignment score: {np.min(dummy_alignment_scores):.4f}")
        print(f"Max alignment score: {np.max(dummy_alignment_scores):.4f}")
        
        # Visualize
        viz_path = os.path.join(output_dir, 'alignment_scores.png')
        visualize_alignment_scores(
            dummy_alignment_scores,
            viz_path,
            config['model']['adapter']['cross_layer_point']
        )
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*50)
    print("Evaluation Completed")
    print("="*50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hallucination Metrics")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/llava_adapter_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to adapter checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config['training']['output_dir'], 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Set device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Note: In a full implementation, we would:
    # 1. Load the LLaVA model
    # 2. Load the adapter checkpoint
    # 3. Run inference on evaluation data
    # 4. Compute metrics on actual predictions
    
    # For now, run evaluation with dummy data
    print("\nNote: Running evaluation with dummy data for demonstration")
    print("In production, replace with actual model inference")
    
    results = run_evaluation(
        model=None,  # Would be actual model
        config=config,
        device=device,
        output_dir=output_dir
    )
    
    print("\n" + "="*50)
    print("Evaluation Summary")
    print("="*50)
    
    if 'chair' in results:
        print(f"\nCHAIR Metrics:")
        print(f"  CHAIRs: {results['chair']['CHAIRs']:.2f}%")
        print(f"  CHAIRi: {results['chair']['CHAIRi']:.2f}%")
    
    if 'pope' in results:
        print(f"\nPOPE Metrics:")
        print(f"  Accuracy: {results['pope']['accuracy']:.2f}%")
        print(f"  F1: {results['pope']['f1']:.2f}")
    
    if 'alignment_scores' in results:
        print(f"\nAlignment Scores:")
        print(f"  Average: {np.mean(results['alignment_scores']):.4f}")


if __name__ == '__main__':
    main()
