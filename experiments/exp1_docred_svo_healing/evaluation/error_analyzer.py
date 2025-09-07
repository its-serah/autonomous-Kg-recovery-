#!/usr/bin/env python3
"""
Error Analysis Framework for Knowledge Graph Construction
Categorizes and analyzes different types of failures in knowledge graph construction
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Set
import re
import logging
from enum import Enum

class ErrorType(Enum):
    """Types of errors in knowledge graph construction"""
    ENTITY_DISAMBIGUATION = "entity_disambiguation"
    RELATION_CLASSIFICATION = "relation_classification"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    COREFERENCE_RESOLUTION = "coreference_resolution"
    NOISE_HANDLING = "noise_handling"
    STRUCTURAL_INCONSISTENCY = "structural_inconsistency"

class ErrorSeverity(Enum):
    """Severity levels for errors"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class KGErrorAnalyzer:
    """
    Analyzes errors in knowledge graph construction and categorizes failure modes
    """
    
    def __init__(self, entity_vocab: Dict = None, relation_vocab: Dict = None):
        """
        Initialize error analyzer
        
        Args:
            entity_vocab: Entity vocabulary for disambiguation analysis
            relation_vocab: Relation vocabulary for classification analysis
        """
        self.logger = logging.getLogger(__name__)
        self.entity_vocab = entity_vocab or {}
        self.relation_vocab = relation_vocab or {}
        self.error_patterns = self._initialize_error_patterns()
    
    def _initialize_error_patterns(self) -> Dict:
        """Initialize patterns for error detection"""
        return {
            'entity_patterns': {
                'case_mismatch': r'[A-Z][a-z]+ vs [a-z][A-Z]',
                'partial_name': r'partial entity name',
                'abbreviation': r'[A-Z]{2,} vs full name',
                'typo': r'character substitution/deletion'
            },
            'relation_patterns': {
                'semantic_confusion': ['located_in vs part_of', 'member_of vs employee_of'],
                'inverse_relations': ['contains vs contained_in', 'parent vs child'],
                'temporal_confusion': ['was vs is', 'former vs current']
            },
            'evidence_patterns': {
                'insufficient_context': 'evidence too short',
                'irrelevant_evidence': 'evidence unrelated to relation',
                'contradictory_evidence': 'evidence contradicts relation'
            }
        }
    
    def analyze_entity_errors(self, predictions: Dict, ground_truth: Dict, 
                            documents: Dict) -> Dict[str, Any]:
        """
        Analyze entity-related errors
        
        Args:
            predictions: Predicted relations
            ground_truth: Ground truth relations
            documents: Document content for analysis
            
        Returns:
            Entity error analysis results
        """
        entity_errors = {
            'disambiguation_errors': [],
            'coreference_errors': [],
            'normalization_errors': [],
            'statistics': defaultdict(int)
        }
        
        for doc_id in ground_truth:
            if doc_id not in predictions:
                continue
                
            pred_rels = predictions[doc_id]
            gt_rels = ground_truth[doc_id]
            doc_content = documents.get(doc_id, {})
            
            # Find entity mismatches
            pred_entities = set()
            gt_entities = set()
            
            for h, t, r in pred_rels:
                pred_entities.add(h)
                pred_entities.add(t)
                
            for h, t, r in gt_rels:
                gt_entities.add(h)
                gt_entities.add(t)
            
            # Analyze disambiguation errors
            missing_entities = gt_entities - pred_entities
            spurious_entities = pred_entities - gt_entities
            
            for entity in missing_entities:
                error_info = self._analyze_entity_error(
                    entity, doc_content, ErrorType.ENTITY_DISAMBIGUATION
                )
                entity_errors['disambiguation_errors'].append(error_info)
                entity_errors['statistics']['missing_entities'] += 1
            
            for entity in spurious_entities:
                error_info = self._analyze_entity_error(
                    entity, doc_content, ErrorType.ENTITY_DISAMBIGUATION
                )
                entity_errors['disambiguation_errors'].append(error_info)
                entity_errors['statistics']['spurious_entities'] += 1
        
        return entity_errors
    
    def analyze_relation_errors(self, predictions: Dict, ground_truth: Dict) -> Dict[str, Any]:
        """
        Analyze relation classification errors
        
        Args:
            predictions: Predicted relations
            ground_truth: Ground truth relations
            
        Returns:
            Relation error analysis results
        """
        relation_errors = {
            'classification_errors': [],
            'confusion_matrix': defaultdict(lambda: defaultdict(int)),
            'semantic_errors': [],
            'statistics': defaultdict(int)
        }
        
        for doc_id in ground_truth:
            if doc_id not in predictions:
                continue
                
            pred_rels = predictions[doc_id]
            gt_rels = ground_truth[doc_id]
            
            # Create entity pair mappings
            pred_pairs = defaultdict(set)
            gt_pairs = defaultdict(set)
            
            for h, t, r in pred_rels:
                pred_pairs[(h, t)].add(r)
                
            for h, t, r in gt_rels:
                gt_pairs[(h, t)].add(r)
            
            # Analyze relation classification errors
            all_pairs = set(pred_pairs.keys()) | set(gt_pairs.keys())
            
            for pair in all_pairs:
                pred_relations = pred_pairs.get(pair, set())
                gt_relations = gt_pairs.get(pair, set())
                
                if pred_relations != gt_relations:
                    error_info = {
                        'document_id': doc_id,
                        'entity_pair': pair,
                        'predicted_relations': list(pred_relations),
                        'ground_truth_relations': list(gt_relations),
                        'error_type': self._classify_relation_error(pred_relations, gt_relations),
                        'severity': self._assess_error_severity(pred_relations, gt_relations)
                    }
                    relation_errors['classification_errors'].append(error_info)
                    
                    # Update confusion matrix
                    for pred_r in pred_relations:
                        for gt_r in gt_relations:
                            relation_errors['confusion_matrix'][gt_r][pred_r] += 1
        
        return relation_errors
    
    def analyze_evidence_errors(self, predictions: Dict, ground_truth: Dict, 
                              documents: Dict) -> Dict[str, Any]:
        """
        Analyze evidence extraction errors
        
        Args:
            predictions: Predicted relations with evidence
            ground_truth: Ground truth relations with evidence
            documents: Document content
            
        Returns:
            Evidence error analysis results
        """
        evidence_errors = {
            'insufficient_evidence': [],
            'irrelevant_evidence': [],
            'contradictory_evidence': [],
            'missing_evidence': [],
            'statistics': defaultdict(int)
        }
        
        for doc_id in ground_truth:
            if doc_id not in predictions:
                continue
                
            # This would need to be implemented based on your evidence format
            # Placeholder for evidence analysis logic
            pass
        
        return evidence_errors
    
    def analyze_structural_errors(self, kg_structure: Dict) -> Dict[str, Any]:
        """
        Analyze structural inconsistencies in the knowledge graph
        
        Args:
            kg_structure: Knowledge graph structure
            
        Returns:
            Structural error analysis results
        """
        structural_errors = {
            'cycles': [],
            'inconsistent_relations': [],
            'orphaned_entities': [],
            'duplicate_relations': [],
            'statistics': defaultdict(int)
        }
        
        # Detect cycles
        cycles = self._detect_cycles(kg_structure)
        structural_errors['cycles'] = cycles
        structural_errors['statistics']['cycle_count'] = len(cycles)
        
        # Detect inconsistent relations
        inconsistencies = self._detect_inconsistent_relations(kg_structure)
        structural_errors['inconsistent_relations'] = inconsistencies
        structural_errors['statistics']['inconsistency_count'] = len(inconsistencies)
        
        # Detect orphaned entities
        orphans = self._detect_orphaned_entities(kg_structure)
        structural_errors['orphaned_entities'] = orphans
        structural_errors['statistics']['orphan_count'] = len(orphans)
        
        return structural_errors
    
    def _analyze_entity_error(self, entity: Any, doc_content: Dict, 
                            error_type: ErrorType) -> Dict[str, Any]:
        """Analyze specific entity error"""
        return {
            'entity': entity,
            'error_type': error_type.value,
            'context': doc_content.get('text', '')[:200],  # First 200 chars
            'suggested_correction': self._suggest_entity_correction(entity, doc_content),
            'confidence': 0.5  # Placeholder confidence score
        }
    
    def _classify_relation_error(self, predicted: Set, ground_truth: Set) -> str:
        """Classify the type of relation error"""
        if not predicted and ground_truth:
            return "missing_relation"
        elif predicted and not ground_truth:
            return "spurious_relation"
        elif predicted and ground_truth:
            return "incorrect_relation"
        else:
            return "unknown_error"
    
    def _assess_error_severity(self, predicted: Set, ground_truth: Set) -> str:
        """Assess the severity of a relation error"""
        if not predicted and ground_truth:
            return ErrorSeverity.HIGH.value
        elif predicted and not ground_truth:
            return ErrorSeverity.MEDIUM.value
        else:
            return ErrorSeverity.LOW.value
    
    def _suggest_entity_correction(self, entity: Any, doc_content: Dict) -> str:
        """Suggest correction for entity error"""
        # Placeholder for entity correction logic
        return f"Suggested correction for {entity}"
    
    def _detect_cycles(self, kg_structure: Dict) -> List[List]:
        """Detect cycles in knowledge graph structure"""
        # Placeholder for cycle detection algorithm
        return []
    
    def _detect_inconsistent_relations(self, kg_structure: Dict) -> List[Dict]:
        """Detect inconsistent relations in knowledge graph"""
        # Placeholder for inconsistency detection
        return []
    
    def _detect_orphaned_entities(self, kg_structure: Dict) -> List[Any]:
        """Detect orphaned entities in knowledge graph"""
        # Placeholder for orphan detection
        return []
    
    def generate_error_report(self, predictions: Dict, ground_truth: Dict, 
                            documents: Dict, kg_structure: Dict = None) -> Dict[str, Any]:
        """
        Generate comprehensive error analysis report
        
        Args:
            predictions: Predicted relations
            ground_truth: Ground truth relations
            documents: Document content
            kg_structure: Optional knowledge graph structure
            
        Returns:
            Comprehensive error analysis report
        """
        report = {
            'entity_errors': self.analyze_entity_errors(predictions, ground_truth, documents),
            'relation_errors': self.analyze_relation_errors(predictions, ground_truth),
            'evidence_errors': self.analyze_evidence_errors(predictions, ground_truth, documents),
            'summary': {
                'total_errors': 0,
                'error_distribution': defaultdict(int),
                'severity_distribution': defaultdict(int),
                'most_common_errors': [],
                'improvement_suggestions': []
            }
        }
        
        if kg_structure:
            report['structural_errors'] = self.analyze_structural_errors(kg_structure)
        
        # Calculate summary statistics
        total_errors = 0
        for error_category in ['entity_errors', 'relation_errors', 'evidence_errors']:
            if error_category in report:
                category_stats = report[error_category].get('statistics', {})
                for error_type, count in category_stats.items():
                    total_errors += count
                    report['summary']['error_distribution'][error_type] += count
        
        report['summary']['total_errors'] = total_errors
        
        # Generate improvement suggestions
        report['summary']['improvement_suggestions'] = self._generate_improvement_suggestions(report)
        
        return report
    
    def _generate_improvement_suggestions(self, error_report: Dict) -> List[str]:
        """Generate suggestions for improving the system based on error analysis"""
        suggestions = []
        
        # Analyze error patterns and suggest improvements
        entity_stats = error_report.get('entity_errors', {}).get('statistics', {})
        if entity_stats.get('missing_entities', 0) > 10:
            suggestions.append("Improve entity recognition by enhancing NER model training")
        
        relation_stats = error_report.get('relation_errors', {}).get('statistics', {})
        if len(error_report.get('relation_errors', {}).get('confusion_matrix', {})) > 0:
            suggestions.append("Address relation classification confusion with better feature engineering")
        
        if not suggestions:
            suggestions.append("Overall system performance is satisfactory")
        
        return suggestions

def main():
    """Example usage of the error analysis framework"""
    analyzer = KGErrorAnalyzer()
    
    # Example usage would require actual data
    predictions = {}
    ground_truth = {}
    documents = {}
    
    report = analyzer.generate_error_report(predictions, ground_truth, documents)
    print(f"Total errors found: {report['summary']['total_errors']}")

if __name__ == "__main__":
    main()
