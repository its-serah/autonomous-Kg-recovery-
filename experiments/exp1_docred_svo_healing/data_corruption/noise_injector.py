#!/usr/bin/env python3
"""
Noise Injection and Corruption Simulation for DocRED Data
Simulates real-world conditions by injecting controlled corruption into clean DocRED data
"""

import json
import random
import numpy as np
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import logging
from enum import Enum
from copy import deepcopy

class CorruptionType(Enum):
    """Types of corruption to inject"""
    ENTITY_TYPOS = "entity_typos"
    ENTITY_CASE_CHANGES = "entity_case_changes"
    ENTITY_SUBSTITUTION = "entity_substitution"
    RELATION_MISLABELING = "relation_mislabeling"
    EVIDENCE_CORRUPTION = "evidence_corruption"
    MISSING_ENTITIES = "missing_entities"
    SPURIOUS_RELATIONS = "spurious_relations"
    TEXT_OCR_ERRORS = "text_ocr_errors"
    ENCODING_ERRORS = "encoding_errors"

class NoiseInjector:
    """
    Injects controlled noise into clean DocRED data to simulate real-world conditions
    """
    
    def __init__(self, corruption_config: Dict = None):
        """
        Initialize noise injector
        
        Args:
            corruption_config: Configuration for corruption types and intensities
        """
        self.logger = logging.getLogger(__name__)
        self.corruption_config = corruption_config or self._default_corruption_config()
        self.corruption_stats = defaultdict(int)
        
        # Common typo patterns and substitutions
        self.typo_patterns = self._initialize_typo_patterns()
        self.case_patterns = self._initialize_case_patterns()
        self.ocr_substitutions = self._initialize_ocr_substitutions()
        
    def _default_corruption_config(self) -> Dict:
        """Default corruption configuration"""
        return {
            'entity_corruption_rate': 0.15,
            'relation_corruption_rate': 0.10,
            'text_corruption_rate': 0.05,
            'evidence_corruption_rate': 0.08,
            'structural_corruption_rate': 0.12,
            'corruption_types': {
                CorruptionType.ENTITY_TYPOS: 0.25,
                CorruptionType.ENTITY_CASE_CHANGES: 0.20,
                CorruptionType.ENTITY_SUBSTITUTION: 0.15,
                CorruptionType.RELATION_MISLABELING: 0.30,
                CorruptionType.EVIDENCE_CORRUPTION: 0.20,
                CorruptionType.MISSING_ENTITIES: 0.10,
                CorruptionType.SPURIOUS_RELATIONS: 0.15,
                CorruptionType.TEXT_OCR_ERRORS: 0.25,
                CorruptionType.ENCODING_ERRORS: 0.10
            }
        }
    
    def _initialize_typo_patterns(self) -> Dict:
        """Initialize common typo patterns"""
        return {
            'character_substitution': {
                'a': ['e', 'o', 's'],
                'e': ['a', 'i', 'r'],
                'i': ['e', 'o', 'u'],
                'o': ['a', 'i', 'u'],
                'u': ['i', 'o', 'y'],
                'n': ['m', 'h'],
                'm': ['n', 'h'],
                'r': ['t', 'f'],
                'l': ['i', '1'],
                's': ['z', '5']
            },
            'character_deletion': ['e', 'a', 'i', 'o', 'u', 'h'],
            'character_insertion': ['e', 'a', 'i', 'o', 'u'],
            'character_swap': True,
            'double_character': ['l', 's', 'e', 'f', 'o']
        }
    
    def _initialize_case_patterns(self) -> Dict:
        """Initialize case change patterns"""
        return {
            'all_lowercase': 0.3,
            'all_uppercase': 0.2,
            'random_case': 0.3,
            'first_letter_lowercase': 0.2
        }
    
    def _initialize_ocr_substitutions(self) -> Dict:
        """Initialize OCR-like character substitutions"""
        return {
            'O': ['0', 'Q'],
            '0': ['O', 'D'],
            'I': ['1', 'l'],
            '1': ['I', 'l'],
            'l': ['1', 'I'],
            'S': ['5', '$'],
            '5': ['S', '$'],
            'B': ['8', '6'],
            '8': ['B', '6'],
            'G': ['6', 'C'],
            '6': ['G', 'C'],
            'rn': ['m'],
            'cl': ['d'],
            'nn': ['m']
        }
    
    def inject_noise(self, docred_data: List[Dict], corruption_intensity: float = 0.15) -> List[Dict]:
        """
        Inject noise into DocRED data
        
        Args:
            docred_data: Clean DocRED dataset
            corruption_intensity: Overall corruption intensity (0.0 to 1.0)
            
        Returns:
            Corrupted DocRED data
        """
        corrupted_data = []
        self.corruption_stats = defaultdict(int)
        
        for doc in docred_data:
            if random.random() < corruption_intensity:
                corrupted_doc = self._corrupt_document(doc)
                corrupted_data.append(corrupted_doc)
            else:
                corrupted_data.append(deepcopy(doc))
                
        self._log_corruption_stats()
        return corrupted_data
    
    def _corrupt_document(self, document: Dict) -> Dict:
        """Corrupt a single document"""
        corrupted_doc = deepcopy(document)
        
        # Corrupt entities
        if 'vertexSet' in corrupted_doc:
            corrupted_doc['vertexSet'] = self._corrupt_entities(corrupted_doc['vertexSet'])
        
        # Corrupt relations
        if 'labels' in corrupted_doc:
            corrupted_doc['labels'] = self._corrupt_relations(corrupted_doc['labels'])
        
        # Corrupt text
        if 'sents' in corrupted_doc:
            corrupted_doc['sents'] = self._corrupt_text(corrupted_doc['sents'])
        
        # Add spurious relations
        corrupted_doc['labels'] = self._add_spurious_relations(
            corrupted_doc.get('labels', []), 
            corrupted_doc.get('vertexSet', [])
        )
        
        return corrupted_doc
    
    def _corrupt_entities(self, vertex_set: List[List[Dict]]) -> List[List[Dict]]:
        """Corrupt entity mentions"""
        corrupted_vertices = []
        
        for entity_mentions in vertex_set:
            corrupted_mentions = []
            
            for mention in entity_mentions:
                if random.random() < self.corruption_config['entity_corruption_rate']:
                    corrupted_mention = self._corrupt_entity_mention(mention)
                    corrupted_mentions.append(corrupted_mention)
                    self.corruption_stats['entities_corrupted'] += 1
                else:
                    corrupted_mentions.append(deepcopy(mention))
            
            corrupted_vertices.append(corrupted_mentions)
        
        # Randomly remove some entities (missing entities)
        if random.random() < self.corruption_config['corruption_types'][CorruptionType.MISSING_ENTITIES]:
            remove_count = random.randint(1, min(3, len(corrupted_vertices)))
            indices_to_remove = random.sample(range(len(corrupted_vertices)), remove_count)
            corrupted_vertices = [
                vertices for i, vertices in enumerate(corrupted_vertices) 
                if i not in indices_to_remove
            ]
            self.corruption_stats['missing_entities'] += remove_count
        
        return corrupted_vertices
    
    def _corrupt_entity_mention(self, mention: Dict) -> Dict:
        """Corrupt a single entity mention"""
        corrupted_mention = deepcopy(mention)
        corruption_type = self._select_corruption_type([
            CorruptionType.ENTITY_TYPOS,
            CorruptionType.ENTITY_CASE_CHANGES,
            CorruptionType.ENTITY_SUBSTITUTION
        ])
        
        if 'name' in corrupted_mention:
            original_name = corrupted_mention['name']
            
            if corruption_type == CorruptionType.ENTITY_TYPOS:
                corrupted_mention['name'] = self._apply_typos(original_name)
                self.corruption_stats['entity_typos'] += 1
                
            elif corruption_type == CorruptionType.ENTITY_CASE_CHANGES:
                corrupted_mention['name'] = self._apply_case_changes(original_name)
                self.corruption_stats['entity_case_changes'] += 1
                
            elif corruption_type == CorruptionType.ENTITY_SUBSTITUTION:
                corrupted_mention['name'] = self._substitute_entity(original_name)
                self.corruption_stats['entity_substitutions'] += 1
        
        return corrupted_mention
    
    def _corrupt_relations(self, relations: List[Dict]) -> List[Dict]:
        """Corrupt relation labels"""
        corrupted_relations = []
        
        for relation in relations:
            if random.random() < self.corruption_config['relation_corruption_rate']:
                corrupted_relation = deepcopy(relation)
                
                # Mislabel relation
                if 'r' in corrupted_relation and random.random() < 0.7:
                    corrupted_relation['r'] = self._mislabel_relation(corrupted_relation['r'])
                    self.corruption_stats['relation_mislabels'] += 1
                
                # Corrupt evidence
                if 'evidence' in corrupted_relation and random.random() < 0.3:
                    corrupted_relation['evidence'] = self._corrupt_evidence(corrupted_relation['evidence'])
                    self.corruption_stats['evidence_corrupted'] += 1
                
                corrupted_relations.append(corrupted_relation)
            else:
                corrupted_relations.append(deepcopy(relation))
        
        return corrupted_relations
    
    def _corrupt_text(self, sentences: List[List[str]]) -> List[List[str]]:
        """Corrupt document text"""
        corrupted_sentences = []
        
        for sentence in sentences:
            corrupted_sentence = []
            
            for token in sentence:
                if random.random() < self.corruption_config['text_corruption_rate']:
                    corrupted_token = self._corrupt_token(token)
                    corrupted_sentence.append(corrupted_token)
                    self.corruption_stats['tokens_corrupted'] += 1
                else:
                    corrupted_sentence.append(token)
            
            corrupted_sentences.append(corrupted_sentence)
        
        return corrupted_sentences
    
    def _apply_typos(self, text: str) -> str:
        """Apply typo patterns to text"""
        if not text or len(text) < 2:
            return text
        
        typo_type = random.choice(['substitution', 'deletion', 'insertion', 'swap', 'double'])
        
        if typo_type == 'substitution' and len(text) > 0:
            pos = random.randint(0, len(text) - 1)
            char = text[pos].lower()
            if char in self.typo_patterns['character_substitution']:
                replacement = random.choice(self.typo_patterns['character_substitution'][char])
                return text[:pos] + replacement + text[pos+1:]
        
        elif typo_type == 'deletion' and len(text) > 1:
            pos = random.randint(0, len(text) - 1)
            if text[pos].lower() in self.typo_patterns['character_deletion']:
                return text[:pos] + text[pos+1:]
        
        elif typo_type == 'insertion':
            pos = random.randint(0, len(text))
            char = random.choice(self.typo_patterns['character_insertion'])
            return text[:pos] + char + text[pos:]
        
        elif typo_type == 'swap' and len(text) > 1:
            pos = random.randint(0, len(text) - 2)
            return text[:pos] + text[pos+1] + text[pos] + text[pos+2:]
        
        elif typo_type == 'double':
            pos = random.randint(0, len(text) - 1)
            if text[pos].lower() in self.typo_patterns['double_character']:
                return text[:pos+1] + text[pos] + text[pos+1:]
        
        return text
    
    def _apply_case_changes(self, text: str) -> str:
        """Apply case changes to text"""
        case_type = random.choices(
            list(self.case_patterns.keys()),
            weights=list(self.case_patterns.values())
        )[0]
        
        if case_type == 'all_lowercase':
            return text.lower()
        elif case_type == 'all_uppercase':
            return text.upper()
        elif case_type == 'random_case':
            return ''.join(char.upper() if random.random() < 0.5 else char.lower() for char in text)
        elif case_type == 'first_letter_lowercase':
            return text[0].lower() + text[1:] if text else text
        
        return text
    
    def _substitute_entity(self, entity_name: str) -> str:
        """Substitute entity with similar one"""
        # Simple substitution patterns
        substitutions = {
            'United States': ['USA', 'US', 'America'],
            'New York': ['NYC', 'NY'],
            'United Kingdom': ['UK', 'Britain'],
            'European Union': ['EU'],
            'World War': ['WW'],
            'President': ['Pres'],
            'Company': ['Corp', 'Co'],
            'University': ['Univ']
        }
        
        for original, alternatives in substitutions.items():
            if original.lower() in entity_name.lower():
                return entity_name.replace(original, random.choice(alternatives))
        
        # If no substitution found, return with minor modification
        return self._apply_typos(entity_name)
    
    def _mislabel_relation(self, relation_id: int) -> int:
        """Mislabel a relation with a similar one"""
        # Common relation confusion pairs (DocRED relation IDs)
        confusion_pairs = [
            (1, 17),   # country vs country of citizenship
            (3, 27),   # located in vs location of formation
            (6, 25),   # head of government vs position held
            (7, 9),    # head of state vs position held
            (20, 22),  # child vs parent
            (30, 26),  # member of vs member of political party
        ]
        
        # Find if current relation has a confusion pair
        for pair in confusion_pairs:
            if relation_id == pair[0]:
                return pair[1] if random.random() < 0.7 else relation_id
            elif relation_id == pair[1]:
                return pair[0] if random.random() < 0.7 else relation_id
        
        # If no specific confusion, randomly change to nearby ID
        return max(0, relation_id + random.choice([-2, -1, 1, 2]))
    
    def _corrupt_evidence(self, evidence: List[int]) -> List[int]:
        """Corrupt evidence indices"""
        if not evidence:
            return evidence
        
        corrupted_evidence = deepcopy(evidence)
        
        # Remove some evidence
        if len(corrupted_evidence) > 1 and random.random() < 0.4:
            remove_count = random.randint(1, len(corrupted_evidence) // 2)
            corrupted_evidence = random.sample(corrupted_evidence, 
                                             len(corrupted_evidence) - remove_count)
        
        # Add spurious evidence
        if random.random() < 0.3:
            max_sentence_id = max(corrupted_evidence) if corrupted_evidence else 10
            spurious_count = random.randint(1, 2)
            for _ in range(spurious_count):
                spurious_id = random.randint(0, max_sentence_id + 5)
                if spurious_id not in corrupted_evidence:
                    corrupted_evidence.append(spurious_id)
        
        return sorted(corrupted_evidence)
    
    def _corrupt_token(self, token: str) -> str:
        """Corrupt individual token"""
        corruption_type = random.choice(['ocr', 'typo', 'encoding'])
        
        if corruption_type == 'ocr':
            return self._apply_ocr_errors(token)
        elif corruption_type == 'typo':
            return self._apply_typos(token)
        elif corruption_type == 'encoding':
            return self._apply_encoding_errors(token)
        
        return token
    
    def _apply_ocr_errors(self, text: str) -> str:
        """Apply OCR-like errors to text"""
        corrupted_text = text
        
        for original, substitutes in self.ocr_substitutions.items():
            if original in corrupted_text and random.random() < 0.3:
                substitute = random.choice(substitutes)
                corrupted_text = corrupted_text.replace(original, substitute, 1)
                break
        
        return corrupted_text
    
    def _apply_encoding_errors(self, text: str) -> str:
        """Apply encoding errors to text"""
        # Common encoding corruptions
        encoding_errors = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ç': 'c', 'ü': 'u', 'ä': 'a', 'ö': 'o',
            '"': '"', '"': '"', ''': "'", ''': "'",
            '—': '-', '–': '-', '…': '...'
        }
        
        corrupted_text = text
        for original, replacement in encoding_errors.items():
            if original in corrupted_text and random.random() < 0.5:
                corrupted_text = corrupted_text.replace(original, replacement)
        
        return corrupted_text
    
    def _add_spurious_relations(self, relations: List[Dict], vertex_set: List[List[Dict]]) -> List[Dict]:
        """Add spurious relations to the data"""
        if not vertex_set or len(vertex_set) < 2:
            return relations
        
        spurious_relations = deepcopy(relations)
        num_entities = len(vertex_set)
        
        # Add random spurious relations
        spurious_count = random.randint(0, max(1, len(relations) // 4))
        
        for _ in range(spurious_count):
            h = random.randint(0, num_entities - 1)
            t = random.randint(0, num_entities - 1)
            
            if h != t:
                spurious_relation = {
                    'h': h,
                    't': t,
                    'r': random.randint(0, 96),  # Assuming 97 relations in DocRED
                    'evidence': [random.randint(0, 10)]  # Random evidence
                }
                spurious_relations.append(spurious_relation)
                self.corruption_stats['spurious_relations'] += 1
        
        return spurious_relations
    
    def _select_corruption_type(self, corruption_types: List[CorruptionType]) -> CorruptionType:
        """Select corruption type based on configuration"""
        weights = [self.corruption_config['corruption_types'][ct] for ct in corruption_types]
        return random.choices(corruption_types, weights=weights)[0]
    
    def _log_corruption_stats(self):
        """Log corruption statistics"""
        self.logger.info("Corruption Statistics:")
        for corruption_type, count in self.corruption_stats.items():
            self.logger.info(f"  {corruption_type}: {count}")
    
    def generate_corruption_report(self, original_data: List[Dict], 
                                 corrupted_data: List[Dict]) -> Dict:
        """
        Generate detailed corruption report
        
        Args:
            original_data: Original clean data
            corrupted_data: Corrupted data
            
        Returns:
            Detailed corruption report
        """
        report = {
            'corruption_summary': dict(self.corruption_stats),
            'dataset_comparison': {
                'original_documents': len(original_data),
                'corrupted_documents': len(corrupted_data),
                'corruption_rate': sum(self.corruption_stats.values()) / len(original_data) if original_data else 0
            },
            'corruption_types_applied': list(self.corruption_stats.keys()),
            'severity_analysis': self._analyze_corruption_severity(original_data, corrupted_data)
        }
        
        return report
    
    def _analyze_corruption_severity(self, original_data: List[Dict], 
                                   corrupted_data: List[Dict]) -> Dict:
        """Analyze corruption severity levels"""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for orig_doc, corr_doc in zip(original_data, corrupted_data):
            severity = self._assess_document_corruption_severity(orig_doc, corr_doc)
            severity_counts[severity] += 1
        
        return severity_counts
    
    def _assess_document_corruption_severity(self, original: Dict, corrupted: Dict) -> str:
        """Assess corruption severity for a single document"""
        changes = 0
        
        # Count entity changes
        if 'vertexSet' in original and 'vertexSet' in corrupted:
            orig_entities = len(original['vertexSet'])
            corr_entities = len(corrupted['vertexSet'])
            changes += abs(orig_entities - corr_entities)
        
        # Count relation changes
        if 'labels' in original and 'labels' in corrupted:
            orig_relations = len(original['labels'])
            corr_relations = len(corrupted['labels'])
            changes += abs(orig_relations - corr_relations)
        
        # Determine severity based on number of changes
        if changes == 0:
            return 'low'
        elif changes <= 2:
            return 'medium'
        elif changes <= 5:
            return 'high'
        else:
            return 'critical'

def main():
    """Example usage of noise injector"""
    # Load sample DocRED data
    with open('sample_docred.json', 'r') as f:
        clean_data = json.load(f)
    
    # Initialize noise injector
    injector = NoiseInjector()
    
    # Inject noise
    corrupted_data = injector.inject_noise(clean_data, corruption_intensity=0.2)
    
    # Generate report
    report = injector.generate_corruption_report(clean_data, corrupted_data)
    
    print("Corruption Report:")
    print(json.dumps(report, indent=2))
    
    # Save corrupted data
    with open('corrupted_docred.json', 'w') as f:
        json.dump(corrupted_data, f, indent=2)

if __name__ == "__main__":
    main()
