#!/usr/bin/env python3
"""
DocRED Baseline Evaluation Framework
Comprehensive evaluation against standard DocRED baselines with precision, recall, F1 metrics
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import logging

class DocREDEvaluator:
    """
    Evaluates knowledge graph construction performance against DocRED standards
    """
    
    def __init__(self, rel_info_path: str = None):
        """
        Initialize evaluator with relation information
        
        Args:
            rel_info_path: Path to DocRED relation info file
        """
        self.logger = logging.getLogger(__name__)
        self.rel_info = {}
        if rel_info_path:
            with open(rel_info_path, 'r') as f:
                self.rel_info = json.load(f)
    
    def load_predictions(self, pred_file: str) -> List[Dict]:
        """Load predictions from file"""
        with open(pred_file, 'r') as f:
            return json.load(f)
    
    def load_ground_truth(self, gt_file: str) -> List[Dict]:
        """Load ground truth from DocRED format file"""
        with open(gt_file, 'r') as f:
            return json.load(f)
    
    def extract_relations(self, data: List[Dict], use_evidence: bool = True) -> Dict:
        """
        Extract relations from DocRED format data
        
        Args:
            data: DocRED format data
            use_evidence: Whether to require evidence for relations
            
        Returns:
            Dictionary mapping doc_id to relations
        """
        relations = defaultdict(set)
        
        for doc in data:
            doc_id = doc.get('title', str(len(relations)))
            
            if 'labels' in doc:  # Ground truth format
                for label in doc['labels']:
                    if use_evidence and not label.get('evidence', []):
                        continue
                    
                    h_idx = label['h']
                    t_idx = label['t']
                    r = label['r']
                    
                    relations[doc_id].add((h_idx, t_idx, r))
                    
            elif 'relations' in doc:  # Prediction format
                for rel in doc['relations']:
                    h_idx = rel['h']
                    t_idx = rel['t']
                    r = rel['r']
                    
                    if use_evidence and not rel.get('evidence', []):
                        continue
                        
                    relations[doc_id].add((h_idx, t_idx, r))
        
        return relations
    
    def compute_f1_scores(self, predictions: Dict, ground_truth: Dict) -> Dict[str, float]:
        """
        Compute precision, recall, and F1 scores
        
        Args:
            predictions: Predicted relations by document
            ground_truth: Ground truth relations by document
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_pred_relations = set()
        all_gt_relations = set()
        
        # Collect all relations across documents
        for doc_id in ground_truth:
            if doc_id in predictions:
                pred_rels = predictions[doc_id]
                gt_rels = ground_truth[doc_id]
                
                # Add document context to relations
                for h, t, r in pred_rels:
                    all_pred_relations.add((doc_id, h, t, r))
                    
                for h, t, r in gt_rels:
                    all_gt_relations.add((doc_id, h, t, r))
        
        # Calculate metrics
        tp = len(all_pred_relations & all_gt_relations)
        fp = len(all_pred_relations - all_gt_relations)
        fn = len(all_gt_relations - all_pred_relations)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_predicted': len(all_pred_relations),
            'total_ground_truth': len(all_gt_relations)
        }
    
    def compute_relation_wise_scores(self, predictions: Dict, ground_truth: Dict) -> Dict[str, Dict]:
        """
        Compute scores for each relation type separately
        
        Args:
            predictions: Predicted relations by document
            ground_truth: Ground truth relations by document
            
        Returns:
            Dictionary with per-relation evaluation metrics
        """
        relation_scores = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Collect predictions and ground truth by relation type
        for doc_id in ground_truth:
            if doc_id in predictions:
                pred_rels = predictions[doc_id]
                gt_rels = ground_truth[doc_id]
                
                # Group by relation type
                pred_by_rel = defaultdict(set)
                gt_by_rel = defaultdict(set)
                
                for h, t, r in pred_rels:
                    pred_by_rel[r].add((doc_id, h, t))
                    
                for h, t, r in gt_rels:
                    gt_by_rel[r].add((doc_id, h, t))
                
                # Calculate per-relation metrics
                all_relations = set(pred_by_rel.keys()) | set(gt_by_rel.keys())
                for rel in all_relations:
                    pred_set = pred_by_rel[rel]
                    gt_set = gt_by_rel[rel]
                    
                    tp = len(pred_set & gt_set)
                    fp = len(pred_set - gt_set)
                    fn = len(gt_set - pred_set)
                    
                    relation_scores[rel]['tp'] += tp
                    relation_scores[rel]['fp'] += fp
                    relation_scores[rel]['fn'] += fn
        
        # Compute final scores
        final_scores = {}
        for rel, counts in relation_scores.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            final_scores[rel] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        
        return final_scores
    
    def evaluate_evidence_quality(self, predictions: Dict, ground_truth: Dict) -> Dict[str, float]:
        """
        Evaluate quality of evidence provided for relations
        
        Args:
            predictions: Predicted relations with evidence
            ground_truth: Ground truth relations with evidence
            
        Returns:
            Evidence evaluation metrics
        """
        evidence_scores = {
            'evidence_precision': 0.0,
            'evidence_recall': 0.0,
            'evidence_f1': 0.0,
            'avg_evidence_length': 0.0
        }
        
        total_evidence_matches = 0
        total_pred_evidence = 0
        total_gt_evidence = 0
        evidence_lengths = []
        
        for doc_id in ground_truth:
            if doc_id not in predictions:
                continue
                
            # This would need to be implemented based on your specific evidence format
            # For now, returning placeholder metrics
            pass
        
        return evidence_scores
    
    def generate_evaluation_report(self, predictions_file: str, ground_truth_file: str, 
                                 output_file: str = None) -> Dict:
        """
        Generate comprehensive evaluation report
        
        Args:
            predictions_file: Path to predictions file
            ground_truth_file: Path to ground truth file
            output_file: Optional output file for report
            
        Returns:
            Complete evaluation results
        """
        # Load data
        predictions_data = self.load_predictions(predictions_file)
        ground_truth_data = self.load_ground_truth(ground_truth_file)
        
        # Extract relations
        predictions = self.extract_relations(predictions_data)
        ground_truth = self.extract_relations(ground_truth_data)
        
        # Compute overall metrics
        overall_scores = self.compute_f1_scores(predictions, ground_truth)
        
        # Compute per-relation metrics
        relation_scores = self.compute_relation_wise_scores(predictions, ground_truth)
        
        # Compute evidence metrics
        evidence_scores = self.evaluate_evidence_quality(predictions, ground_truth)
        
        # Create comprehensive report
        report = {
            'overall_metrics': overall_scores,
            'relation_wise_metrics': relation_scores,
            'evidence_metrics': evidence_scores,
            'summary': {
                'total_documents': len(ground_truth),
                'avg_relations_per_doc': np.mean([len(rels) for rels in ground_truth.values()]),
                'total_relation_types': len(relation_scores),
                'best_performing_relations': sorted(
                    [(rel, scores['f1']) for rel, scores in relation_scores.items()],
                    key=lambda x: x[1], reverse=True
                )[:5],
                'worst_performing_relations': sorted(
                    [(rel, scores['f1']) for rel, scores in relation_scores.items()],
                    key=lambda x: x[1]
                )[:5]
            }
        }
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation report saved to {output_file}")
        
        return report

def main():
    """Example usage of the evaluation framework"""
    evaluator = DocREDEvaluator()
    
    # Example evaluation
    predictions_file = "path/to/predictions.json"
    ground_truth_file = "path/to/ground_truth.json"
    
    report = evaluator.generate_evaluation_report(
        predictions_file, 
        ground_truth_file, 
        "evaluation_report.json"
    )
    
    print(f"Overall F1 Score: {report['overall_metrics']['f1']:.4f}")
    print(f"Precision: {report['overall_metrics']['precision']:.4f}")
    print(f"Recall: {report['overall_metrics']['recall']:.4f}")

if __name__ == "__main__":
    main()
