#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for Knowledge Graph Healing
Implements graph coherence, healing efficiency, and visualization tools
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report, confusion_matrix
import networkx as nx

class GraphCoherenceMetrics:
    """
    Metrics for evaluating knowledge graph structural coherence
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_coherence_score(self, kg_relations: Dict, entity_types: Dict = None) -> float:
        """
        Calculate overall graph coherence score
        
        Args:
            kg_relations: Knowledge graph relations by document
            entity_types: Entity type information (optional)
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        scores = []
        
        # Structural coherence
        structural_score = self._calculate_structural_coherence(kg_relations)
        scores.append(('structural', structural_score, 0.3))
        
        # Semantic coherence
        semantic_score = self._calculate_semantic_coherence(kg_relations, entity_types)
        scores.append(('semantic', semantic_score, 0.3))
        
        # Consistency coherence
        consistency_score = self._calculate_consistency_coherence(kg_relations)
        scores.append(('consistency', consistency_score, 0.2))
        
        # Density coherence
        density_score = self._calculate_density_coherence(kg_relations)
        scores.append(('density', density_score, 0.2))
        
        # Weighted average
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        return max(0.0, min(1.0, weighted_score))
    
    def _calculate_structural_coherence(self, kg_relations: Dict) -> float:
        """Calculate structural coherence based on graph topology"""
        if not kg_relations:
            return 0.0
        
        total_score = 0.0
        total_docs = 0
        
        for doc_id, relations in kg_relations.items():
            if not relations:
                continue
                
            # Build graph for this document
            G = nx.DiGraph()
            for h, t, r in relations:
                G.add_edge(h, t, relation=r)
            
            if len(G.nodes()) == 0:
                continue
            
            doc_score = 0.0
            
            # Connected components (prefer more connected graphs)
            if G.number_of_nodes() > 1:
                weakly_connected = nx.number_weakly_connected_components(G)
                connectivity_score = 1.0 - (weakly_connected - 1) / (G.number_of_nodes() - 1)
                doc_score += connectivity_score * 0.4
            
            # Average clustering coefficient
            undirected_G = G.to_undirected()
            if len(undirected_G.nodes()) > 2:
                clustering_score = nx.average_clustering(undirected_G)
                doc_score += clustering_score * 0.3
            
            # Path length distribution (prefer reasonable path lengths)
            try:
                if nx.is_weakly_connected(G):
                    avg_path_length = nx.average_shortest_path_length(G.to_undirected())
                    # Normalize path length score (optimal around 2-4)
                    path_score = max(0, 1.0 - abs(avg_path_length - 3.0) / 5.0)
                    doc_score += path_score * 0.3
            except:
                pass
            
            total_score += doc_score
            total_docs += 1
        
        return total_score / total_docs if total_docs > 0 else 0.0
    
    def _calculate_semantic_coherence(self, kg_relations: Dict, entity_types: Dict) -> float:
        """Calculate semantic coherence based on relation-entity type compatibility"""
        if not kg_relations or not entity_types:
            return 0.5  # Neutral score when no type information
        
        # Define expected entity type pairs for common relations
        # This would be populated based on your relation schema
        relation_type_expectations = {
            'P17': [('PERSON', 'GPE'), ('ORG', 'GPE')],  # country
            'P19': [('PERSON', 'GPE')],  # place of birth
            'P20': [('PERSON', 'DATE')],  # date of death
            # Add more relation-type mappings as needed
        }
        
        total_score = 0.0
        total_relations = 0
        
        for doc_id, relations in kg_relations.items():
            for h, t, r in relations:
                if str(r) in relation_type_expectations:
                    head_type = entity_types.get(h, 'UNKNOWN')
                    tail_type = entity_types.get(t, 'UNKNOWN')
                    
                    expected_pairs = relation_type_expectations[str(r)]
                    if (head_type, tail_type) in expected_pairs:
                        total_score += 1.0
                    else:
                        total_score += 0.3  # Partial credit for unknown types
                else:
                    total_score += 0.5  # Neutral score for unknown relations
                
                total_relations += 1
        
        return total_score / total_relations if total_relations > 0 else 0.5
    
    def _calculate_consistency_coherence(self, kg_relations: Dict) -> float:
        """Calculate consistency coherence (absence of contradictions)"""
        if not kg_relations:
            return 1.0
        
        # Define mutually exclusive relations
        exclusive_relations = [
            {'P20', 'P570'},  # date of death vs death date (duplicates)
            {'P27', 'P17'},   # country of citizenship vs country (sometimes exclusive)
        ]
        
        total_score = 0.0
        total_checks = 0
        
        for doc_id, relations in kg_relations.items():
            # Group relations by entity pairs
            entity_pair_relations = defaultdict(set)
            
            for h, t, r in relations:
                entity_pair_relations[(h, t)].add(str(r))
                entity_pair_relations[(t, h)].add(str(r))  # Consider reverse too
            
            # Check for contradictions
            for (h, t), relation_set in entity_pair_relations.items():
                for exclusive_set in exclusive_relations:
                    intersection = relation_set & exclusive_set
                    if len(intersection) > 1:
                        total_score += 0.0  # Contradiction found
                    else:
                        total_score += 1.0  # No contradiction
                    total_checks += 1
        
        return total_score / total_checks if total_checks > 0 else 1.0
    
    def _calculate_density_coherence(self, kg_relations: Dict) -> float:
        """Calculate density coherence (appropriate relation density)"""
        if not kg_relations:
            return 0.0
        
        densities = []
        
        for doc_id, relations in kg_relations.items():
            if not relations:
                continue
            
            # Get unique entities
            entities = set()
            for h, t, r in relations:
                entities.add(h)
                entities.add(t)
            
            n_entities = len(entities)
            n_relations = len(relations)
            
            if n_entities < 2:
                continue
            
            # Calculate density (relations per possible entity pair)
            max_possible_relations = n_entities * (n_entities - 1)
            density = n_relations / max_possible_relations if max_possible_relations > 0 else 0
            
            # Optimal density is around 0.1-0.3 for most real-world KGs
            if 0.1 <= density <= 0.3:
                density_score = 1.0
            elif density < 0.1:
                density_score = density / 0.1
            else:  # density > 0.3
                density_score = max(0.1, 1.0 - (density - 0.3) / 0.7)
            
            densities.append(density_score)
        
        return np.mean(densities) if densities else 0.0

class HealingEfficiencyMetrics:
    """
    Metrics for evaluating healing process efficiency
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_healing_efficiency(self, healing_log: List[Dict]) -> Dict[str, float]:
        """
        Calculate comprehensive healing efficiency metrics
        
        Args:
            healing_log: List of healing actions and their results
            
        Returns:
            Dictionary of efficiency metrics
        """
        if not healing_log:
            return {'overall_efficiency': 0.0}
        
        metrics = {}
        
        # Time efficiency
        metrics['time_efficiency'] = self._calculate_time_efficiency(healing_log)
        
        # Action efficiency
        metrics['action_efficiency'] = self._calculate_action_efficiency(healing_log)
        
        # Improvement efficiency
        metrics['improvement_efficiency'] = self._calculate_improvement_efficiency(healing_log)
        
        # Resource efficiency
        metrics['resource_efficiency'] = self._calculate_resource_efficiency(healing_log)
        
        # Overall efficiency (weighted average)
        weights = {'time': 0.25, 'action': 0.25, 'improvement': 0.35, 'resource': 0.15}
        metrics['overall_efficiency'] = sum(
            metrics[f'{key}_efficiency'] * weight 
            for key, weight in weights.items()
        )
        
        return metrics
    
    def _calculate_time_efficiency(self, healing_log: List[Dict]) -> float:
        """Calculate time-based efficiency"""
        if not healing_log:
            return 0.0
        
        total_time = sum(action.get('execution_time', 0) for action in healing_log)
        total_improvement = sum(action.get('improvement_score', 0) for action in healing_log)
        
        if total_time == 0:
            return 1.0 if total_improvement > 0 else 0.0
        
        # Higher improvement per unit time is better
        time_efficiency = total_improvement / total_time
        return min(1.0, time_efficiency)
    
    def _calculate_action_efficiency(self, healing_log: List[Dict]) -> float:
        """Calculate action-based efficiency"""
        if not healing_log:
            return 0.0
        
        successful_actions = sum(1 for action in healing_log if action.get('improvement_score', 0) > 0)
        total_actions = len(healing_log)
        
        return successful_actions / total_actions
    
    def _calculate_improvement_efficiency(self, healing_log: List[Dict]) -> float:
        """Calculate improvement rate efficiency"""
        if not healing_log:
            return 0.0
        
        improvements = [action.get('improvement_score', 0) for action in healing_log]
        
        # Calculate improvement rate (how quickly we approach optimal performance)
        improvement_rate = 0.0
        cumulative_improvement = 0.0
        
        for i, improvement in enumerate(improvements):
            cumulative_improvement += improvement
            # Diminishing returns: later improvements are worth less
            weight = 1.0 / (i + 1)
            improvement_rate += improvement * weight
        
        # Normalize by total possible improvement
        max_possible_improvement = len(improvements) * 1.0  # Assuming max improvement of 1.0 per action
        return min(1.0, improvement_rate / max_possible_improvement) if max_possible_improvement > 0 else 0.0
    
    def _calculate_resource_efficiency(self, healing_log: List[Dict]) -> float:
        """Calculate resource usage efficiency"""
        if not healing_log:
            return 1.0
        
        # Resource usage could include memory, CPU, network calls, etc.
        # For now, we'll use action count as a proxy for resource usage
        total_resources = len(healing_log)
        total_improvement = sum(action.get('improvement_score', 0) for action in healing_log)
        
        if total_resources == 0:
            return 1.0
        
        # Higher improvement per resource unit is better
        resource_efficiency = total_improvement / total_resources
        return min(1.0, resource_efficiency)

class ComprehensiveEvaluator:
    """
    Main comprehensive evaluation suite
    """
    
    def __init__(self, output_dir: str = "evaluation_results"):
        """
        Initialize comprehensive evaluator
        
        Args:
            output_dir: Directory to save evaluation results and visualizations
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize component evaluators
        self.coherence_metrics = GraphCoherenceMetrics()
        self.efficiency_metrics = HealingEfficiencyMetrics()
        
        # Evaluation history
        self.evaluation_history = []
    
    def evaluate_system(self, predictions: Dict, ground_truth: Dict, 
                       healing_log: List[Dict] = None, system_name: str = "System") -> Dict:
        """
        Comprehensive system evaluation
        
        Args:
            predictions: System predictions
            ground_truth: Ground truth data
            healing_log: Optional healing process log
            system_name: Name for this system evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        start_time = time.time()
        
        # Basic performance metrics
        basic_metrics = self._calculate_basic_metrics(predictions, ground_truth)
        
        # Graph coherence metrics
        coherence_score = self.coherence_metrics.calculate_coherence_score(predictions)
        
        # Healing efficiency metrics (if available)
        efficiency_metrics = {}
        if healing_log:
            efficiency_metrics = self.efficiency_metrics.calculate_healing_efficiency(healing_log)
        
        # Advanced analysis
        error_analysis = self._perform_error_analysis(predictions, ground_truth)
        relation_analysis = self._analyze_relation_performance(predictions, ground_truth)
        
        # Compile comprehensive results
        evaluation_results = {
            'system_name': system_name,
            'timestamp': time.time(),
            'evaluation_time': time.time() - start_time,
            'basic_metrics': basic_metrics,
            'coherence_score': coherence_score,
            'efficiency_metrics': efficiency_metrics,
            'error_analysis': error_analysis,
            'relation_analysis': relation_analysis,
            'overall_score': self._calculate_overall_score(
                basic_metrics, coherence_score, efficiency_metrics
            )
        }
        
        # Store evaluation
        self.evaluation_history.append(evaluation_results)
        
        # Generate visualizations
        self._generate_evaluation_visualizations(evaluation_results)
        
        # Save detailed results
        self._save_evaluation_results(evaluation_results)
        
        return evaluation_results
    
    def _calculate_basic_metrics(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Calculate basic precision, recall, F1 metrics"""
        all_pred_relations = set()
        all_gt_relations = set()
        
        for doc_id in ground_truth:
            if doc_id in predictions:
                pred_rels = predictions[doc_id]
                gt_rels = ground_truth[doc_id]
                
                for h, t, r in pred_rels:
                    all_pred_relations.add((doc_id, h, t, r))
                    
                for h, t, r in gt_rels:
                    all_gt_relations.add((doc_id, h, t, r))
        
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
            'fn': fn
        }
    
    def _perform_error_analysis(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Perform detailed error analysis"""
        error_stats = {
            'missing_relations': 0,
            'spurious_relations': 0,
            'incorrect_relations': 0,
            'error_patterns': defaultdict(int)
        }
        
        for doc_id in ground_truth:
            if doc_id not in predictions:
                error_stats['missing_relations'] += len(ground_truth[doc_id])
                continue
            
            pred_set = set(predictions[doc_id])
            gt_set = set(ground_truth[doc_id])
            
            # Missing relations
            missing = gt_set - pred_set
            error_stats['missing_relations'] += len(missing)
            
            # Spurious relations
            spurious = pred_set - gt_set
            error_stats['spurious_relations'] += len(spurious)
            
            # Analyze error patterns
            for h, t, r in missing:
                error_stats['error_patterns'][f'missing_relation_{r}'] += 1
                
            for h, t, r in spurious:
                error_stats['error_patterns'][f'spurious_relation_{r}'] += 1
        
        return error_stats
    
    def _analyze_relation_performance(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Analyze performance by relation type"""
        relation_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for doc_id in ground_truth:
            if doc_id not in predictions:
                for h, t, r in ground_truth[doc_id]:
                    relation_stats[r]['fn'] += 1
                continue
            
            pred_relations = defaultdict(set)
            gt_relations = defaultdict(set)
            
            for h, t, r in predictions[doc_id]:
                pred_relations[r].add((doc_id, h, t))
                
            for h, t, r in ground_truth[doc_id]:
                gt_relations[r].add((doc_id, h, t))
            
            all_relations = set(pred_relations.keys()) | set(gt_relations.keys())
            
            for rel in all_relations:
                pred_set = pred_relations.get(rel, set())
                gt_set = gt_relations.get(rel, set())
                
                tp = len(pred_set & gt_set)
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                
                relation_stats[rel]['tp'] += tp
                relation_stats[rel]['fp'] += fp
                relation_stats[rel]['fn'] += fn
        
        # Calculate performance metrics for each relation
        relation_performance = {}
        for rel, counts in relation_stats.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            relation_performance[rel] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        
        return relation_performance
    
    def _calculate_overall_score(self, basic_metrics: Dict, coherence_score: float, 
                               efficiency_metrics: Dict) -> float:
        """Calculate weighted overall score"""
        f1_score = basic_metrics.get('f1', 0.0)
        coherence = coherence_score
        efficiency = efficiency_metrics.get('overall_efficiency', 0.5)
        
        # Weighted combination
        overall_score = (
            f1_score * 0.5 +           # Performance weight
            coherence * 0.3 +          # Coherence weight
            efficiency * 0.2           # Efficiency weight
        )
        
        return overall_score
    
    def _generate_evaluation_visualizations(self, results: Dict):
        """Generate comprehensive evaluation visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # Performance metrics radar chart
        self._create_radar_chart(results)
        
        # Relation performance heatmap
        self._create_relation_heatmap(results)
        
        # Efficiency trends
        if results['efficiency_metrics']:
            self._create_efficiency_plot(results)
        
        # Error analysis pie chart
        self._create_error_analysis_plot(results)
    
    def _create_radar_chart(self, results: Dict):
        """Create radar chart for overall performance"""
        categories = ['Precision', 'Recall', 'F1', 'Coherence', 'Efficiency']
        values = [
            results['basic_metrics']['precision'],
            results['basic_metrics']['recall'],
            results['basic_metrics']['f1'],
            results['coherence_score'],
            results['efficiency_metrics'].get('overall_efficiency', 0.5)
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=results['system_name'])
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f"Performance Overview: {results['system_name']}", y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"radar_chart_{results['system_name']}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_relation_heatmap(self, results: Dict):
        """Create heatmap of relation performance"""
        relation_analysis = results['relation_analysis']
        
        if not relation_analysis:
            return
        
        # Prepare data for heatmap
        relations = list(relation_analysis.keys())
        metrics = ['precision', 'recall', 'f1']
        
        heatmap_data = []
        for metric in metrics:
            heatmap_data.append([relation_analysis[rel][metric] for rel in relations])
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, 
                   xticklabels=[f'R{rel}' for rel in relations],
                   yticklabels=metrics,
                   annot=True, fmt='.3f', cmap='RdYlBu_r')
        plt.title(f'Relation Performance Heatmap: {results["system_name"]}')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"relation_heatmap_{results['system_name']}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_efficiency_plot(self, results: Dict):
        """Create efficiency metrics visualization"""
        efficiency_metrics = results['efficiency_metrics']
        
        metrics = ['time_efficiency', 'action_efficiency', 'improvement_efficiency', 'resource_efficiency']
        values = [efficiency_metrics.get(metric, 0) for metric in metrics]
        labels = ['Time', 'Action', 'Improvement', 'Resource']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
        plt.ylim(0, 1)
        plt.ylabel('Efficiency Score')
        plt.title(f'Healing Efficiency Breakdown: {results["system_name"]}')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"efficiency_plot_{results['system_name']}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_error_analysis_plot(self, results: Dict):
        """Create error analysis visualization"""
        error_analysis = results['error_analysis']
        
        error_types = ['missing_relations', 'spurious_relations']
        error_counts = [error_analysis.get(error_type, 0) for error_type in error_types]
        
        if sum(error_counts) == 0:
            return
        
        plt.figure(figsize=(8, 8))
        colors = ['#FF6B6B', '#4ECDC4']
        plt.pie(error_counts, labels=['Missing Relations', 'Spurious Relations'], 
               colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title(f'Error Distribution: {results["system_name"]}')
        plt.axis('equal')
        plt.savefig(self.output_dir / f"error_analysis_{results['system_name']}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_evaluation_results(self, results: Dict):
        """Save detailed evaluation results to file"""
        output_file = self.output_dir / f"evaluation_{results['system_name']}_{int(results['timestamp'])}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation results saved to {output_file}")
    
    def compare_systems(self, system_results: List[Dict]) -> Dict:
        """Compare multiple system evaluations"""
        if not system_results:
            return {}
        
        comparison = {
            'systems': [result['system_name'] for result in system_results],
            'metrics_comparison': {},
            'rankings': {}
        }
        
        # Compare basic metrics
        metrics = ['precision', 'recall', 'f1']
        for metric in metrics:
            comparison['metrics_comparison'][metric] = {
                result['system_name']: result['basic_metrics'][metric]
                for result in system_results
            }
        
        # Add coherence and overall scores
        comparison['metrics_comparison']['coherence'] = {
            result['system_name']: result['coherence_score']
            for result in system_results
        }
        
        comparison['metrics_comparison']['overall'] = {
            result['system_name']: result['overall_score']
            for result in system_results
        }
        
        # Generate rankings
        for metric in ['f1', 'coherence', 'overall']:
            if metric == 'f1':
                metric_values = [(result['system_name'], result['basic_metrics']['f1']) 
                               for result in system_results]
            elif metric == 'coherence':
                metric_values = [(result['system_name'], result['coherence_score']) 
                               for result in system_results]
            else:  # overall
                metric_values = [(result['system_name'], result['overall_score']) 
                               for result in system_results]
            
            ranked = sorted(metric_values, key=lambda x: x[1], reverse=True)
            comparison['rankings'][metric] = [name for name, _ in ranked]
        
        # Save comparison
        comparison_file = self.output_dir / f"system_comparison_{int(time.time())}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Generate comparison visualization
        self._create_comparison_plot(system_results)
        
        return comparison
    
    def _create_comparison_plot(self, system_results: List[Dict]):
        """Create system comparison visualization"""
        system_names = [result['system_name'] for result in system_results]
        f1_scores = [result['basic_metrics']['f1'] for result in system_results]
        coherence_scores = [result['coherence_score'] for result in system_results]
        overall_scores = [result['overall_score'] for result in system_results]
        
        x = np.arange(len(system_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8)
        bars2 = ax.bar(x, coherence_scores, width, label='Coherence Score', alpha=0.8)
        bars3 = ax.bar(x + width, overall_scores, width, label='Overall Score', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('System Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(system_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"system_comparison_{int(time.time())}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Example usage of comprehensive evaluator"""
    evaluator = ComprehensiveEvaluator("evaluation_results")
    
    # Mock data for demonstration
    predictions = {
        'doc1': [(0, 1, 'P1'), (1, 2, 'P2')],
        'doc2': [(0, 2, 'P1'), (1, 3, 'P3')]
    }
    
    ground_truth = {
        'doc1': [(0, 1, 'P1'), (2, 3, 'P4')],
        'doc2': [(0, 2, 'P1'), (1, 3, 'P3')]
    }
    
    healing_log = [
        {'improvement_score': 0.1, 'execution_time': 0.001},
        {'improvement_score': 0.05, 'execution_time': 0.002}
    ]
    
    # Evaluate system
    results = evaluator.evaluate_system(predictions, ground_truth, healing_log, "BiologicalHealer")
    
    print("Evaluation completed!")
    print(f"Overall Score: {results['overall_score']:.3f}")
    print(f"F1 Score: {results['basic_metrics']['f1']:.3f}")
    print(f"Coherence Score: {results['coherence_score']:.3f}")

if __name__ == "__main__":
    main()
