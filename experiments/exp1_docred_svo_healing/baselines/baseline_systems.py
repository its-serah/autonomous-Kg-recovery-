#!/usr/bin/env python3
"""
Baseline System Implementations for Knowledge Graph Healing
Rule-based cleaning, confidence filtering, and other simple alternatives for comparison
"""

import json
import re
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Set
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import statistics

@dataclass
class RelationTriple:
    """Represents a relation triple with metadata"""
    head: int
    tail: int
    relation: str
    confidence: float = 1.0
    evidence: List[int] = None
    source: str = "extracted"
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

class BaselineSystem(ABC):
    """Abstract base class for baseline systems"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def clean_knowledge_graph(self, kg_data: Dict, documents: Dict = None) -> Dict:
        """
        Clean the knowledge graph using the baseline approach
        
        Args:
            kg_data: Raw knowledge graph data
            documents: Optional document content
            
        Returns:
            Cleaned knowledge graph data
        """
        pass
    
    @abstractmethod
    def get_system_description(self) -> str:
        """Get description of the baseline system"""
        pass

class VanillaDocREDBaseline(BaselineSystem):
    """
    Vanilla DocRED baseline - no cleaning/healing applied
    """
    
    def __init__(self):
        super().__init__("VanillaDocRED")
    
    def clean_knowledge_graph(self, kg_data: Dict, documents: Dict = None) -> Dict:
        """Return data as-is without any cleaning"""
        return kg_data
    
    def get_system_description(self) -> str:
        return "Vanilla DocRED baseline with no cleaning or healing applied"

class RuleBasedCleaner(BaselineSystem):
    """
    Rule-based cleaning system using heuristics and regex patterns
    """
    
    def __init__(self, rules_config: Dict = None):
        super().__init__("RuleBasedCleaner")
        self.rules_config = rules_config or self._default_rules_config()
        self.entity_patterns = self._initialize_entity_patterns()
        self.relation_rules = self._initialize_relation_rules()
        
    def _default_rules_config(self) -> Dict:
        """Default configuration for rule-based cleaning"""
        return {
            'entity_cleaning': {
                'normalize_case': True,
                'remove_extra_spaces': True,
                'fix_common_typos': True,
                'standardize_abbreviations': True
            },
            'relation_filtering': {
                'remove_low_confidence': True,
                'confidence_threshold': 0.3,
                'apply_domain_rules': True,
                'check_entity_compatibility': True
            },
            'structural_cleaning': {
                'remove_duplicates': True,
                'resolve_contradictions': True,
                'enforce_cardinality_constraints': True
            }
        }
    
    def _initialize_entity_patterns(self) -> Dict:
        """Initialize entity cleaning patterns"""
        return {
            'typo_fixes': {
                r'\bUnited States\b': 'United States',
                r'\bNew York\b': 'New York',
                r'\bUnited Kingdom\b': 'United Kingdom',
                # Add more common typo fixes
            },
            'case_normalization': {
                'proper_nouns': r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$',
                'abbreviations': r'^[A-Z]+$',
                'mixed_case': r'^[A-Za-z\s]+$'
            },
            'abbreviation_expansion': {
                'US': 'United States',
                'UK': 'United Kingdom', 
                'NYC': 'New York City',
                'EU': 'European Union',
                # Add more abbreviations
            }
        }
    
    def _initialize_relation_rules(self) -> Dict:
        """Initialize relation validation rules"""
        return {
            'entity_type_compatibility': {
                'P17': [('PERSON', 'GPE'), ('ORG', 'GPE')],  # country
                'P19': [('PERSON', 'GPE')],  # place of birth  
                'P20': [('PERSON', 'DATE')],  # date of death
                'P27': [('PERSON', 'GPE')],  # country of citizenship
                # Add more compatibility rules
            },
            'mutual_exclusions': [
                {'P20', 'P570'},  # date of death relations
                {'P27', 'P17'},   # citizenship vs country (context dependent)
            ],
            'cardinality_constraints': {
                'P19': 1,  # place of birth (single value)
                'P20': 1,  # date of death (single value)
                # Add more cardinality constraints
            }
        }
    
    def clean_knowledge_graph(self, kg_data: Dict, documents: Dict = None) -> Dict:
        """
        Apply rule-based cleaning to the knowledge graph
        
        Args:
            kg_data: Raw knowledge graph data by document
            documents: Optional document content for context
            
        Returns:
            Cleaned knowledge graph data
        """
        cleaned_kg = {}
        
        for doc_id, relations in kg_data.items():
            # Convert to RelationTriple objects for easier manipulation
            relation_triples = []
            for h, t, r in relations:
                triple = RelationTriple(
                    head=h, tail=t, relation=str(r), 
                    confidence=self._estimate_confidence(h, t, r, documents.get(doc_id, {}) if documents else {})
                )
                relation_triples.append(triple)
            
            # Apply cleaning rules
            cleaned_triples = self._apply_cleaning_rules(relation_triples, doc_id, documents)
            
            # Convert back to tuple format
            cleaned_relations = [(t.head, t.tail, t.relation) for t in cleaned_triples]
            cleaned_kg[doc_id] = cleaned_relations
            
        return cleaned_kg
    
    def _apply_cleaning_rules(self, relation_triples: List[RelationTriple], 
                            doc_id: str, documents: Dict) -> List[RelationTriple]:
        """Apply all cleaning rules to relation triples"""
        
        # Step 1: Entity cleaning
        if self.rules_config['entity_cleaning']['normalize_case']:
            relation_triples = self._normalize_entity_case(relation_triples)
        
        if self.rules_config['entity_cleaning']['fix_common_typos']:
            relation_triples = self._fix_entity_typos(relation_triples)
        
        # Step 2: Relation filtering
        if self.rules_config['relation_filtering']['remove_low_confidence']:
            threshold = self.rules_config['relation_filtering']['confidence_threshold']
            relation_triples = [t for t in relation_triples if t.confidence >= threshold]
        
        if self.rules_config['relation_filtering']['apply_domain_rules']:
            relation_triples = self._apply_domain_rules(relation_triples)
        
        # Step 3: Structural cleaning
        if self.rules_config['structural_cleaning']['remove_duplicates']:
            relation_triples = self._remove_duplicates(relation_triples)
        
        if self.rules_config['structural_cleaning']['resolve_contradictions']:
            relation_triples = self._resolve_contradictions(relation_triples)
        
        return relation_triples
    
    def _normalize_entity_case(self, relation_triples: List[RelationTriple]) -> List[RelationTriple]:
        """Normalize entity case (placeholder - would need entity text access)"""
        # In a real implementation, this would access entity text and normalize
        return relation_triples
    
    def _fix_entity_typos(self, relation_triples: List[RelationTriple]) -> List[RelationTriple]:
        """Fix common entity typos (placeholder)"""
        # In a real implementation, this would access entity text and fix typos
        return relation_triples
    
    def _apply_domain_rules(self, relation_triples: List[RelationTriple]) -> List[RelationTriple]:
        """Apply domain-specific validation rules"""
        valid_triples = []
        
        for triple in relation_triples:
            # Apply entity type compatibility rules (simplified)
            if self._validate_relation_compatibility(triple):
                valid_triples.append(triple)
            else:
                triple.confidence *= 0.5  # Reduce confidence for incompatible relations
                if triple.confidence >= 0.1:  # Still keep if confidence not too low
                    valid_triples.append(triple)
        
        return valid_triples
    
    def _validate_relation_compatibility(self, triple: RelationTriple) -> bool:
        """Validate relation-entity type compatibility"""
        # Simplified validation - in practice would use actual entity types
        relation_rules = self.relation_rules['entity_type_compatibility']
        return triple.relation not in relation_rules or True  # Placeholder
    
    def _remove_duplicates(self, relation_triples: List[RelationTriple]) -> List[RelationTriple]:
        """Remove duplicate relation triples"""
        seen_triples = set()
        unique_triples = []
        
        for triple in relation_triples:
            triple_key = (triple.head, triple.tail, triple.relation)
            if triple_key not in seen_triples:
                seen_triples.add(triple_key)
                unique_triples.append(triple)
        
        return unique_triples
    
    def _resolve_contradictions(self, relation_triples: List[RelationTriple]) -> List[RelationTriple]:
        """Resolve contradictory relations"""
        # Group by entity pairs
        entity_pair_relations = defaultdict(list)
        for triple in relation_triples:
            entity_pair_relations[(triple.head, triple.tail)].append(triple)
        
        resolved_triples = []
        
        for (h, t), triples in entity_pair_relations.items():
            # Check for mutual exclusions
            relation_set = {triple.relation for triple in triples}
            
            has_contradiction = False
            for exclusive_set in self.relation_rules['mutual_exclusions']:
                if len(relation_set & exclusive_set) > 1:
                    has_contradiction = True
                    # Keep the relation with highest confidence
                    best_triple = max(triples, key=lambda x: x.confidence)
                    filtered_triples = [t for t in triples if t.relation == best_triple.relation]
                    resolved_triples.extend(filtered_triples)
                    break
            
            if not has_contradiction:
                resolved_triples.extend(triples)
        
        return resolved_triples
    
    def _estimate_confidence(self, head: int, tail: int, relation: str, document: Dict) -> float:
        """Estimate confidence score for a relation triple"""
        # Simplified confidence estimation
        base_confidence = 0.7
        
        # Adjust based on relation frequency (more common = higher confidence)
        relation_bonus = min(0.2, hash(relation) % 100 / 1000)  # Placeholder
        
        # Adjust based on entity distance (closer entities = higher confidence)
        entity_distance_penalty = min(0.1, abs(head - tail) * 0.01)
        
        confidence = base_confidence + relation_bonus - entity_distance_penalty
        return max(0.1, min(1.0, confidence))
    
    def get_system_description(self) -> str:
        return "Rule-based cleaning system using heuristics and domain knowledge"

class ConfidenceFilteringBaseline(BaselineSystem):
    """
    Simple confidence-based filtering baseline
    """
    
    def __init__(self, confidence_threshold: float = 0.5, 
                 confidence_estimation_method: str = 'frequency'):
        super().__init__("ConfidenceFiltering")
        self.confidence_threshold = confidence_threshold
        self.confidence_method = confidence_estimation_method
        
    def clean_knowledge_graph(self, kg_data: Dict, documents: Dict = None) -> Dict:
        """
        Filter relations based on confidence scores
        
        Args:
            kg_data: Raw knowledge graph data
            documents: Optional document content
            
        Returns:
            Filtered knowledge graph data
        """
        # First, estimate confidence scores for all relations
        relation_confidences = self._estimate_all_confidences(kg_data)
        
        # Filter based on threshold
        filtered_kg = {}
        
        for doc_id, relations in kg_data.items():
            filtered_relations = []
            
            for h, t, r in relations:
                confidence = relation_confidences.get((doc_id, h, t, r), 0.5)
                
                if confidence >= self.confidence_threshold:
                    filtered_relations.append((h, t, r))
            
            filtered_kg[doc_id] = filtered_relations
        
        return filtered_kg
    
    def _estimate_all_confidences(self, kg_data: Dict) -> Dict[Tuple, float]:
        """Estimate confidence scores for all relations"""
        confidences = {}
        
        if self.confidence_method == 'frequency':
            confidences = self._frequency_based_confidence(kg_data)
        elif self.confidence_method == 'statistical':
            confidences = self._statistical_confidence(kg_data)
        else:
            # Default to uniform confidence
            for doc_id, relations in kg_data.items():
                for h, t, r in relations:
                    confidences[(doc_id, h, t, r)] = 0.5
        
        return confidences
    
    def _frequency_based_confidence(self, kg_data: Dict) -> Dict[Tuple, float]:
        """Calculate confidence based on relation frequency"""
        relation_counts = Counter()
        total_relations = 0
        
        # Count relation frequencies
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                relation_counts[r] += 1
                total_relations += 1
        
        # Calculate confidence scores
        confidences = {}
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                frequency = relation_counts[r]
                # More frequent relations get higher confidence
                confidence = min(0.95, 0.3 + (frequency / total_relations) * 10)
                confidences[(doc_id, h, t, r)] = confidence
        
        return confidences
    
    def _statistical_confidence(self, kg_data: Dict) -> Dict[Tuple, float]:
        """Calculate confidence using statistical measures"""
        confidences = {}
        
        # Calculate entity pair statistics
        entity_pair_counts = defaultdict(int)
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                entity_pair_counts[(h, t)] += 1
        
        # Calculate confidence based on entity pair frequency and relation diversity
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                pair_count = entity_pair_counts[(h, t)]
                
                # Higher confidence for entity pairs that appear multiple times
                pair_bonus = min(0.3, pair_count * 0.1)
                
                # Base confidence
                base_confidence = 0.5
                
                confidence = base_confidence + pair_bonus
                confidences[(doc_id, h, t, r)] = min(0.95, confidence)
        
        return confidences
    
    def get_system_description(self) -> str:
        return f"Confidence filtering baseline (threshold={self.confidence_threshold}, method={self.confidence_method})"

class HybridBaseline(BaselineSystem):
    """
    Hybrid baseline combining rule-based cleaning and confidence filtering
    """
    
    def __init__(self, confidence_threshold: float = 0.4):
        super().__init__("HybridBaseline")
        self.rule_cleaner = RuleBasedCleaner()
        self.confidence_filter = ConfidenceFilteringBaseline(confidence_threshold)
        
    def clean_knowledge_graph(self, kg_data: Dict, documents: Dict = None) -> Dict:
        """
        Apply hybrid cleaning: first rules, then confidence filtering
        """
        # Step 1: Apply rule-based cleaning
        rule_cleaned = self.rule_cleaner.clean_knowledge_graph(kg_data, documents)
        
        # Step 2: Apply confidence filtering
        final_cleaned = self.confidence_filter.clean_knowledge_graph(rule_cleaned, documents)
        
        return final_cleaned
    
    def get_system_description(self) -> str:
        return "Hybrid baseline combining rule-based cleaning and confidence filtering"

class StatisticalOutlierFilter(BaselineSystem):
    """
    Statistical outlier detection and removal baseline
    """
    
    def __init__(self, outlier_threshold: float = 2.0):
        super().__init__("StatisticalOutlierFilter")
        self.outlier_threshold = outlier_threshold
        
    def clean_knowledge_graph(self, kg_data: Dict, documents: Dict = None) -> Dict:
        """
        Remove statistical outliers from the knowledge graph
        """
        # Calculate relation statistics
        relation_stats = self._calculate_relation_statistics(kg_data)
        
        # Identify and remove outliers
        filtered_kg = {}
        
        for doc_id, relations in kg_data.items():
            filtered_relations = []
            
            for h, t, r in relations:
                if not self._is_statistical_outlier(h, t, r, relation_stats):
                    filtered_relations.append((h, t, r))
            
            filtered_kg[doc_id] = filtered_relations
        
        return filtered_kg
    
    def _calculate_relation_statistics(self, kg_data: Dict) -> Dict:
        """Calculate statistical measures for relations"""
        stats = {
            'relation_frequencies': Counter(),
            'entity_degrees': defaultdict(int),
            'relation_entity_stats': defaultdict(list)
        }
        
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                stats['relation_frequencies'][r] += 1
                stats['entity_degrees'][h] += 1
                stats['entity_degrees'][t] += 1
                stats['relation_entity_stats'][r].append((h, t))
        
        # Calculate means and standard deviations
        degree_values = list(stats['entity_degrees'].values())
        stats['mean_degree'] = statistics.mean(degree_values) if degree_values else 0
        stats['std_degree'] = statistics.stdev(degree_values) if len(degree_values) > 1 else 1
        
        freq_values = list(stats['relation_frequencies'].values())
        stats['mean_frequency'] = statistics.mean(freq_values) if freq_values else 0
        stats['std_frequency'] = statistics.stdev(freq_values) if len(freq_values) > 1 else 1
        
        return stats
    
    def _is_statistical_outlier(self, head: int, tail: int, relation: str, stats: Dict) -> bool:
        """Determine if a relation triple is a statistical outlier"""
        
        # Check relation frequency outlier
        rel_freq = stats['relation_frequencies'][relation]
        freq_z_score = abs(rel_freq - stats['mean_frequency']) / stats['std_frequency']
        
        if freq_z_score > self.outlier_threshold:
            return True
        
        # Check entity degree outliers
        head_degree = stats['entity_degrees'][head]
        tail_degree = stats['entity_degrees'][tail]
        
        head_z_score = abs(head_degree - stats['mean_degree']) / stats['std_degree']
        tail_z_score = abs(tail_degree - stats['mean_degree']) / stats['std_degree']
        
        if head_z_score > self.outlier_threshold or tail_z_score > self.outlier_threshold:
            return True
        
        return False
    
    def get_system_description(self) -> str:
        return f"Statistical outlier filter (threshold={self.outlier_threshold} std deviations)"

class BaselineSystemEvaluator:
    """
    Evaluator for comparing different baseline systems
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.systems = self._initialize_baseline_systems()
    
    def _initialize_baseline_systems(self) -> Dict[str, BaselineSystem]:
        """Initialize all baseline systems for comparison"""
        return {
            'vanilla': VanillaDocREDBaseline(),
            'rule_based': RuleBasedCleaner(),
            'confidence_filter': ConfidenceFilteringBaseline(confidence_threshold=0.5),
            'confidence_filter_strict': ConfidenceFilteringBaseline(confidence_threshold=0.7),
            'hybrid': HybridBaseline(),
            'outlier_filter': StatisticalOutlierFilter(outlier_threshold=2.0),
            'outlier_filter_strict': StatisticalOutlierFilter(outlier_threshold=1.5)
        }
    
    def evaluate_all_systems(self, kg_data: Dict, ground_truth: Dict, 
                           documents: Dict = None) -> Dict[str, Dict]:
        """
        Evaluate all baseline systems on the given data
        
        Args:
            kg_data: Raw knowledge graph data
            ground_truth: Ground truth for evaluation
            documents: Optional document content
            
        Returns:
            Evaluation results for all systems
        """
        results = {}
        
        for system_name, system in self.systems.items():
            self.logger.info(f"Evaluating {system_name}...")
            
            # Clean the knowledge graph
            cleaned_kg = system.clean_knowledge_graph(kg_data, documents)
            
            # Evaluate against ground truth
            evaluation_metrics = self._evaluate_system(cleaned_kg, ground_truth)
            
            results[system_name] = {
                'system_description': system.get_system_description(),
                'metrics': evaluation_metrics,
                'cleaned_relations_count': sum(len(relations) for relations in cleaned_kg.values()),
                'original_relations_count': sum(len(relations) for relations in kg_data.values())
            }
        
        return results
    
    def _evaluate_system(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Evaluate a single system against ground truth"""
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
    
    def generate_comparison_report(self, evaluation_results: Dict[str, Dict]) -> Dict:
        """Generate a comprehensive comparison report"""
        
        # Rankings by different metrics
        rankings = {}
        for metric in ['precision', 'recall', 'f1']:
            metric_values = [(name, results['metrics'][metric]) 
                           for name, results in evaluation_results.items()]
            ranked = sorted(metric_values, key=lambda x: x[1], reverse=True)
            rankings[metric] = ranked
        
        # Best performing systems
        best_systems = {
            'precision': rankings['precision'][0][0] if rankings['precision'] else None,
            'recall': rankings['recall'][0][0] if rankings['recall'] else None,
            'f1': rankings['f1'][0][0] if rankings['f1'] else None
        }
        
        # Performance summary
        performance_summary = {}
        for system_name, results in evaluation_results.items():
            performance_summary[system_name] = {
                'f1_score': results['metrics']['f1'],
                'relations_removed': results['original_relations_count'] - results['cleaned_relations_count'],
                'removal_rate': (results['original_relations_count'] - results['cleaned_relations_count']) / 
                               results['original_relations_count'] if results['original_relations_count'] > 0 else 0
            }
        
        return {
            'rankings': rankings,
            'best_systems': best_systems,
            'performance_summary': performance_summary,
            'system_count': len(evaluation_results)
        }

def main():
    """Example usage of baseline systems"""
    # Mock data for demonstration
    kg_data = {
        'doc1': [(0, 1, 'P1'), (1, 2, 'P2'), (0, 2, 'P3')],
        'doc2': [(0, 2, 'P1'), (1, 3, 'P3'), (2, 3, 'P4')]
    }
    
    ground_truth = {
        'doc1': [(0, 1, 'P1'), (1, 2, 'P2')],
        'doc2': [(0, 2, 'P1'), (1, 3, 'P3')]
    }
    
    # Initialize evaluator
    evaluator = BaselineSystemEvaluator()
    
    # Evaluate all systems
    results = evaluator.evaluate_all_systems(kg_data, ground_truth)
    
    # Generate comparison report
    comparison = evaluator.generate_comparison_report(results)
    
    # Print results
    print("Baseline System Comparison Results:")
    print(f"Best F1 Score: {comparison['best_systems']['f1']}")
    
    for system_name, metrics in comparison['performance_summary'].items():
        print(f"{system_name}: F1={metrics['f1_score']:.3f}, "
              f"Removed {metrics['relations_removed']} relations")

if __name__ == "__main__":
    main()
