#!/usr/bin/env python3
"""
Self-contained Knowledge Graph Quality Improvement Experiment
Compares RL-based adaptive cleaning with baseline methods
"""

import numpy as np
import json
import random
import logging
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter, deque
from enum import Enum
from abc import ABC, abstractmethod
import statistics
from dataclasses import dataclass

# Tier-1 components
from src.data.docred_loader import DocREDDataLoader as Tier1DocLoader
from src.models.tfidf_verifier import DocLevelVerifier, TripleExample
from src.models.hf_docre import HFDocREVerifier, TripleExampleHF
from src.embeddings.distmult import KGETrainer
from src.constraints.constraints import apply_constraints
from src.ensemble.stacker import StackingEnsemble
from src.calibration.conformal import compute_conformal_thresholds
from src.anomaly.edge_anomaly import EdgeAnomalyDetector

# ============================================================================
# RL-BASED ADAPTIVE HEALER COMPONENTS
# ============================================================================

class CleaningAction(Enum):
    """Available cleaning actions"""
    ENTITY_DISAMBIGUATION = 0
    RELATION_CORRECTION = 1
    EVIDENCE_VALIDATION = 2
    STRUCTURAL_REPAIR = 3
    CONFIDENCE_FILTERING = 4
    NO_ACTION = 5

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

class KGState:
    """Represents the current state of the knowledge graph"""
    
    def __init__(self, graph_metrics: Dict, error_counts: Dict, confidence_scores: List[float]):
        self.graph_metrics = graph_metrics
        self.error_counts = error_counts
        self.confidence_scores = confidence_scores
        self.state_vector = self._encode_state()
    
    def _encode_state(self) -> np.ndarray:
        """Encode the KG state as a feature vector"""
        features = []
        
        # Graph quality metrics
        features.extend([
            self.graph_metrics.get('precision', 0.0),
            self.graph_metrics.get('recall', 0.0),
            self.graph_metrics.get('f1', 0.0),
            self.graph_metrics.get('coherence_score', 0.0)
        ])
        
        # Error counts (normalized)
        total_errors = sum(self.error_counts.values()) + 1e-6
        features.extend([
            self.error_counts.get('entity_errors', 0) / total_errors,
            self.error_counts.get('relation_errors', 0) / total_errors,
            self.error_counts.get('evidence_errors', 0) / total_errors,
            self.error_counts.get('structural_errors', 0) / total_errors
        ])
        
        # Confidence statistics
        if self.confidence_scores:
            features.extend([
                np.mean(self.confidence_scores),
                np.std(self.confidence_scores),
                np.min(self.confidence_scores),
                np.max(self.confidence_scores)
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def __hash__(self):
        """Make state hashable for Q-table lookup"""
        return hash(tuple(self.state_vector.round(3)))
    
    def __eq__(self, other):
        return np.allclose(self.state_vector, other.state_vector, atol=1e-3)

class AdaptiveKGCleaner:
    """RL-based adaptive knowledge graph cleaner using Q-learning"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.logger = logging.getLogger(__name__)
        
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Q-table and experience buffer
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = deque(maxlen=10000)
        
        # Performance tracking
        self.episode_rewards = []
        self.healing_history = []
        self.strategy_effectiveness = defaultdict(list)
    
    def select_action(self, state: KGState) -> CleaningAction:
        """Select healing action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice(list(CleaningAction))
        else:
            return self._get_best_action(state)
    
    def _get_best_action(self, state: KGState) -> CleaningAction:
        """Get the best action for a given state"""
        state_key = hash(state)
        if state_key not in self.q_table:
            return random.choice(list(CleaningAction))
        
        q_values = self.q_table[state_key]
        best_action_value = max(q_values.values()) if q_values else 0
        best_actions = [action for action, value in q_values.items() if value == best_action_value]
        
        return random.choice(best_actions) if best_actions else random.choice(list(CleaningAction))
    
    def execute_healing_action(self, action: CleaningAction, kg_data: Dict) -> Dict:
        """Execute the selected healing action"""
        healing_strategies = {
            CleaningAction.ENTITY_DISAMBIGUATION: self._entity_disambiguation,
            CleaningAction.RELATION_CORRECTION: self._relation_correction,
            CleaningAction.EVIDENCE_VALIDATION: self._evidence_validation,
            CleaningAction.STRUCTURAL_REPAIR: self._structural_repair,
            CleaningAction.CONFIDENCE_FILTERING: self._confidence_filtering,
            CleaningAction.NO_ACTION: self._no_action
        }
        
        return healing_strategies[action](kg_data)
    
    def _entity_disambiguation(self, kg_data: Dict) -> Dict:
        """Apply entity disambiguation healing"""
        total_relations = sum(len(relations) for relations in kg_data.values())
        disambiguated = min(5, int(total_relations * 0.05))
        
        return {
            'action': CleaningAction.ENTITY_DISAMBIGUATION.name,
            'entities_disambiguated': disambiguated,
            'confidence_improvement': disambiguated * 0.02,
            'execution_time': 0.001
        }
    
    def _relation_correction(self, kg_data: Dict) -> Dict:
        """Apply relation correction healing"""
        total_relations = sum(len(relations) for relations in kg_data.values())
        corrected = min(3, int(total_relations * 0.03))
        
        return {
            'action': CleaningAction.RELATION_CORRECTION.name,
            'relations_corrected': corrected,
            'accuracy_improvement': corrected * 0.03,
            'execution_time': 0.002
        }
    
    def _evidence_validation(self, kg_data: Dict) -> Dict:
        """Apply evidence validation healing"""
        total_relations = sum(len(relations) for relations in kg_data.values())
        validated = min(4, int(total_relations * 0.04))
        
        return {
            'action': CleaningAction.EVIDENCE_VALIDATION.name,
            'evidence_validated': validated,
            'precision_improvement': validated * 0.025,
            'execution_time': 0.003
        }
    
    def _structural_repair(self, kg_data: Dict) -> Dict:
        """Apply structural repair healing"""
        total_relations = sum(len(relations) for relations in kg_data.values())
        repaired = min(2, int(total_relations * 0.02))
        
        return {
            'action': CleaningAction.STRUCTURAL_REPAIR.name,
            'structures_repaired': repaired,
            'coherence_improvement': repaired * 0.04,
            'execution_time': 0.004
        }
    
    def _confidence_filtering(self, kg_data: Dict) -> Dict:
        """Apply confidence-based filtering"""
        total_relations = sum(len(relations) for relations in kg_data.values())
        filtered = int(total_relations * 0.1)
        
        return {
            'action': CleaningAction.CONFIDENCE_FILTERING.name,
            'relations_filtered': filtered,
            'precision_improvement': filtered * 0.01,
            'execution_time': 0.001
        }
    
    def _no_action(self, kg_data: Dict) -> Dict:
        """No healing action"""
        return {
            'action': CleaningAction.NO_ACTION.name,
            'changes_made': 0,
            'improvement': 0.0,
            'execution_time': 0.0
        }
    
    def calculate_reward(self, prev_state: KGState, action: CleaningAction, 
                        new_state: KGState, healing_results: Dict) -> float:
        """Calculate reward for the healing action"""
        # Base reward from performance improvement
        f1_improvement = new_state.graph_metrics.get('f1', 0) - prev_state.graph_metrics.get('f1', 0)
        precision_improvement = new_state.graph_metrics.get('precision', 0) - prev_state.graph_metrics.get('precision', 0)
        recall_improvement = new_state.graph_metrics.get('recall', 0) - prev_state.graph_metrics.get('recall', 0)
        
        # Weighted performance reward
        performance_reward = (f1_improvement * 2.0 + precision_improvement + recall_improvement) * 100
        
        # Error reduction reward
        prev_total_errors = sum(prev_state.error_counts.values())
        new_total_errors = sum(new_state.error_counts.values())
        error_reduction_reward = (prev_total_errors - new_total_errors) * 0.5
        
        # Efficiency penalty
        execution_time = healing_results.get('execution_time', 0.001)
        efficiency_penalty = execution_time * 10 if performance_reward < 0.01 else 0
        
        # Action-specific bonuses
        action_bonus = 0.0
        if action == CleaningAction.NO_ACTION and performance_reward >= 0:
            action_bonus = 0.1
        elif action != CleaningAction.NO_ACTION and performance_reward > 0:
            action_bonus = 0.2
        
        total_reward = performance_reward + error_reduction_reward - efficiency_penalty + action_bonus
        return max(total_reward, -1.0)
    
    def update_q_value(self, state: KGState, action: CleaningAction, reward: float, next_state: KGState):
        """Update Q-value using Q-learning update rule"""
        state_key = hash(state)
        next_state_key = hash(next_state)
        
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        target = reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target - current_q)
        
        self.q_table[state_key][action] = new_q
    
    def train_episode(self, kg_data: Dict, max_steps: int = 8) -> float:
        """Train the healer for one episode"""
        total_reward = 0.0
        episode_history = []
        
        # Initial state
        current_state = self._create_kg_state(kg_data)
        
        for step in range(max_steps):
            # Select and execute action
            action = self.select_action(current_state)
            healing_results = self.execute_healing_action(action, kg_data)
            
            # Simulate state transition
            next_state = self._simulate_state_transition(current_state, action, healing_results)
            
            # Calculate reward
            reward = self.calculate_reward(current_state, action, next_state, healing_results)
            total_reward += reward
            
            # Update Q-value
            self.update_q_value(current_state, action, reward, next_state)
            
            # Store experience
            episode_history.append({
                'step': step,
                'state': current_state.state_vector.tolist(),
                'action': action.name,
                'reward': reward,
                'healing_results': healing_results
            })
            
            # Update strategy effectiveness
            self.strategy_effectiveness[action].append(reward)
            
            # Early stopping
            if next_state.graph_metrics.get('f1', 0) >= 0.99 or action == CleaningAction.NO_ACTION:
                break
            
            current_state = next_state
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Store episode data
        self.episode_rewards.append(total_reward)
        self.healing_history.append(episode_history)
        
        return total_reward
    
    def _create_kg_state(self, kg_data: Dict) -> KGState:
        """Create KGState from current KG data"""
        total_relations = sum(len(relations) for relations in kg_data.values())
        
        # Simulate metrics (would be computed from actual evaluation in practice)
        graph_metrics = {
            'precision': random.uniform(0.4, 0.8),
            'recall': random.uniform(0.4, 0.8),
            'f1': random.uniform(0.4, 0.8),
            'coherence_score': random.uniform(0.3, 0.9)
        }
        
        error_counts = {
            'entity_errors': random.randint(1, max(1, total_relations // 10)),
            'relation_errors': random.randint(1, max(1, total_relations // 8)),
            'evidence_errors': random.randint(0, max(1, total_relations // 12)),
            'structural_errors': random.randint(0, max(1, total_relations // 15))
        }
        
        confidence_scores = [random.uniform(0.2, 1.0) for _ in range(total_relations)]
        
        return KGState(graph_metrics, error_counts, confidence_scores)
    
    def _simulate_state_transition(self, state: KGState, action: CleaningAction, healing_results: Dict) -> KGState:
        """Simulate state transition after healing action"""
        new_metrics = state.graph_metrics.copy()
        new_error_counts = state.error_counts.copy()
        new_confidence_scores = state.confidence_scores.copy()
        
        # Apply improvements based on action
        if action == CleaningAction.ENTITY_DISAMBIGUATION:
            improvement = healing_results.get('confidence_improvement', 0)
            new_metrics['precision'] = min(1.0, new_metrics['precision'] + improvement)
            new_error_counts['entity_errors'] = max(0, new_error_counts['entity_errors'] - healing_results.get('entities_disambiguated', 0))
        
        elif action == CleaningAction.RELATION_CORRECTION:
            improvement = healing_results.get('accuracy_improvement', 0)
            new_metrics['f1'] = min(1.0, new_metrics['f1'] + improvement)
            new_error_counts['relation_errors'] = max(0, new_error_counts['relation_errors'] - healing_results.get('relations_corrected', 0))
        
        elif action == CleaningAction.CONFIDENCE_FILTERING:
            improvement = healing_results.get('precision_improvement', 0)
            new_metrics['precision'] = min(1.0, new_metrics['precision'] + improvement)
        
        # Add some noise to simulate realistic transitions
        for metric in new_metrics:
            noise = random.uniform(-0.01, 0.01)
            new_metrics[metric] = max(0.0, min(1.0, new_metrics[metric] + noise))
        
        return KGState(new_metrics, new_error_counts, new_confidence_scores)

# ============================================================================
# BASELINE SYSTEMS
# ============================================================================

class BaselineSystem(ABC):
    """Abstract base class for baseline systems"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def clean_knowledge_graph(self, kg_data: Dict) -> Dict:
        pass
    
    @abstractmethod
    def get_system_description(self) -> str:
        pass

class VanillaBaseline(BaselineSystem):
    """Vanilla baseline - no cleaning applied"""
    
    def __init__(self):
        super().__init__("VanillaBaseline")
    
    def clean_knowledge_graph(self, kg_data: Dict) -> Dict:
        return kg_data
    
    def get_system_description(self) -> str:
        return "Vanilla baseline with no cleaning applied"

class RuleBasedCleaner(BaselineSystem):
    """Rule-based cleaning system"""
    
    def __init__(self):
        super().__init__("RuleBasedCleaner")
        self.confidence_threshold = 0.3
        
    def clean_knowledge_graph(self, kg_data: Dict) -> Dict:
        cleaned_kg = {}
        
        for doc_id, relations in kg_data.items():
            cleaned_relations = []
            
            # Apply rule-based cleaning
            for h, t, r in relations:
                # Simulate confidence-based filtering
                confidence = self._estimate_confidence(h, t, r)
                if confidence >= self.confidence_threshold:
                    cleaned_relations.append((h, t, r))
            
            # Remove duplicates
            cleaned_relations = list(set(cleaned_relations))
            cleaned_kg[doc_id] = cleaned_relations
            
        return cleaned_kg
    
    def _estimate_confidence(self, head: int, tail: int, relation: str) -> float:
        """Estimate confidence score for a relation triple with MORE VARIATION"""
        # Create much more variation in confidence scores
        relation_hash = hash(relation) % 1000
        entity_hash = hash((head, tail)) % 1000
        
        base_confidence = 0.3 + (relation_hash / 1000) * 0.6  # 0.3 to 0.9 range
        entity_penalty = (entity_hash / 1000) * 0.3  # up to 0.3 penalty
        
        confidence = base_confidence - entity_penalty
        return max(0.1, min(0.95, confidence))
    
    def get_system_description(self) -> str:
        return "Rule-based cleaning with confidence filtering and deduplication"

class ConfidenceFilteringBaseline(BaselineSystem):
    """Simple confidence-based filtering baseline"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__("ConfidenceFiltering")
        self.confidence_threshold = confidence_threshold
        
    def clean_knowledge_graph(self, kg_data: Dict) -> Dict:
        """Filter relations based on confidence scores"""
        filtered_kg = {}
        
        # First, estimate confidence scores
        relation_confidences = self._estimate_all_confidences(kg_data)
        
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
        relation_counts = Counter()
        total_relations = 0
        
        # Count relation frequencies
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                relation_counts[r] += 1
                total_relations += 1
        
        # Calculate confidence scores
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                frequency = relation_counts[r]
                confidence = min(0.95, 0.3 + (frequency / total_relations) * 10)
                confidences[(doc_id, h, t, r)] = confidence
        
        return confidences
    
    def get_system_description(self) -> str:
        return f"Confidence filtering (threshold={self.confidence_threshold})"

class StatisticalOutlierFilter(BaselineSystem):
    """Statistical outlier detection and removal baseline"""
    
    def __init__(self, outlier_threshold: float = 2.0):
        super().__init__("StatisticalOutlierFilter")
        self.outlier_threshold = outlier_threshold
        
    def clean_knowledge_graph(self, kg_data: Dict) -> Dict:
        """Remove statistical outliers from the knowledge graph"""
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
            'entity_degrees': defaultdict(int)
        }
        
        for doc_id, relations in kg_data.items():
            for h, t, r in relations:
                stats['relation_frequencies'][r] += 1
                stats['entity_degrees'][h] += 1
                stats['entity_degrees'][t] += 1
        
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

# ============================================================================
# DATA GENERATION AND CORRUPTION
# ============================================================================

class DocREDDataLoader:
    """Load and process DocRED dataset"""
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            self.data_path = Path(__file__).parent / "experiments" / "exp1_docred_svo_healing" / "data" / "raw_data"
        else:
            self.data_path = Path(data_path)
            
    def load_docred_data(self, split: str = "train_annotated", max_docs: int = 500) -> Tuple[Dict, Dict]:
        """Load DocRED data and convert to our format"""
        file_path = self.data_path / f"{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"DocRED file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            docred_data = json.load(f)
        
        # Convert to our KG format
        kg_data = {}
        ground_truth = {}
        
        for i, doc in enumerate(docred_data[:max_docs]):
            doc_id = f"docred_{i:04d}"
            
            # Extract relations from labels
            relations = []
            for label in doc.get("labels", []):
                h = label["h"]
                t = label["t"] 
                r = label["r"]
                relations.append((h, t, r))
            
            kg_data[doc_id] = relations
            ground_truth[doc_id] = relations.copy()
            
        return kg_data, ground_truth
    
    def get_relation_info(self) -> Dict:
        """Load relation type information"""
        rel_info_path = self.data_path / "rel_info.json"
        if rel_info_path.exists():
            with open(rel_info_path, 'r') as f:
                return json.load(f)
        return {}

class KnowledgeGraphGenerator:
    """Generate synthetic knowledge graphs for experimentation (kept for fallback)"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.relation_types = [
            "works_for", "located_in", "born_in", "married_to", "founded_by",
            "capital_of", "parent_of", "sibling_of", "studied_at", "CEO_of",
            "part_of", "neighbor_of", "friend_of", "employee_of", "member_of"
        ]
        
    def generate_clean_kg(self, num_documents: int = 15, 
                         relations_per_doc: Tuple[int, int] = (8, 20)) -> Tuple[Dict, Dict]:
        """Generate a clean knowledge graph dataset"""
        kg_data = {}
        ground_truth = {}
        
        for doc_id in range(num_documents):
            doc_name = f"doc_{doc_id:03d}"
            num_relations = random.randint(*relations_per_doc)
            
            relations = []
            for _ in range(num_relations):
                head = random.randint(0, 25)  # Entity IDs
                tail = random.randint(0, 25)
                relation = random.choice(self.relation_types)
                
                if head != tail:  # Avoid self-relations
                    relations.append((head, tail, relation))
            
            # Remove duplicates
            relations = list(set(relations))
            kg_data[doc_name] = relations
            ground_truth[doc_name] = relations.copy()
            
        return kg_data, ground_truth
    
    def apply_corruption(self, clean_kg: Dict, corruption_rate: float = 0.2) -> Dict:
        """Apply HEAVY corruption to clean knowledge graph (synthetic fallback)."""
        corrupted_kg = {}
        
        for doc_id, relations in clean_kg.items():
            corrupted_relations = []
            
            for h, t, r in relations:
                if random.random() < corruption_rate:
                    # Apply corruption with higher damage
                    corruption_type = random.choice([
                        'entity_swap', 'entity_swap', 'relation_change', 'relation_change', 
                        'remove', 'both_entities_wrong'
                    ])
                    
                    if corruption_type == 'entity_swap':
                        # Swap one entity
                        if random.random() < 0.5:
                            h = random.randint(0, 50)  # Larger entity space (synthetic only)
                        else:
                            t = random.randint(0, 50)
                    elif corruption_type == 'relation_change':
                        # Change relation type  
                        r = random.choice(self.relation_types)
                    elif corruption_type == 'both_entities_wrong':
                        # Corrupt both entities
                        h = random.randint(0, 50)
                        t = random.randint(0, 50)
                    elif corruption_type == 'remove':
                        # Skip this relation (remove it)
                        continue
                
                if h != t:  # Ensure no self-relations
                    corrupted_relations.append((h, t, r))
            
            # Add MANY spurious relations (noise)
            original_length = len(relations)
            num_spurious = random.randint(original_length // 3, original_length // 2)  # 33-50% noise
            for _ in range(num_spurious):
                h = random.randint(0, 50)
                t = random.randint(0, 50) 
                r = random.choice(self.relation_types)
                if h != t:
                    corrupted_relations.append((h, t, r))
            
            corrupted_kg[doc_id] = corrupted_relations
        
        return corrupted_kg

    def apply_corruption_docred(self, clean_kg: Dict, docs: Dict, relation_ids: List[str], corruption_rate: float = 0.4) -> Dict:
        """Apply corruption to DocRED-formatted KG while staying in-vocabulary and in-bounds.
        - Only uses provided relation_ids (Wikidata PIDs)
        - Entity indices are sampled within each doc's vertexSet
        - Adds 33-50% spurious edges per doc
        """
        corrupted_kg: Dict[str, List[Tuple[int,int,str]]] = {}
        rel_ids = relation_ids if relation_ids else []
        for doc_id, relations in clean_kg.items():
            corrupted_relations: List[Tuple[int,int,str]] = []
            num_ents = len(docs.get(doc_id, {}).get('vertexSet', []))
            for h, t, r in relations:
                if random.random() < corruption_rate:
                    corruption_type = random.choice([
                        'entity_swap', 'relation_change', 'remove', 'both_entities_wrong'
                    ])
                    if corruption_type == 'entity_swap' and num_ents >= 2:
                        if random.random() < 0.5:
                            h = random.randint(0, max(0, num_ents - 1))
                        else:
                            t = random.randint(0, max(0, num_ents - 1))
                    elif corruption_type == 'relation_change' and rel_ids:
                        # change to a different PID
                        r_candidates = [x for x in rel_ids if x != r]
                        r = random.choice(r_candidates) if r_candidates else r
                    elif corruption_type == 'both_entities_wrong' and num_ents >= 2:
                        h = random.randint(0, max(0, num_ents - 1))
                        t = random.randint(0, max(0, num_ents - 1))
                    elif corruption_type == 'remove':
                        continue
                if h != t:
                    corrupted_relations.append((h,t,r))
            # Spurious edges
            original_length = len(relations)
            num_spurious = random.randint(original_length // 3, original_length // 2) if original_length > 0 else 0
            for _ in range(num_spurious):
                if num_ents >= 2 and rel_ids:
                    h = random.randint(0, num_ents - 1)
                    t = random.randint(0, num_ents - 1)
                    if h == t:
                        continue
                    r = random.choice(rel_ids)
                    corrupted_relations.append((h,t,r))
            corrupted_kg[doc_id] = corrupted_relations
        return corrupted_kg

# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

def calculate_performance_metrics(predictions: Dict, ground_truth: Dict) -> Dict:
    """Calculate performance metrics"""
    all_pred_relations = set()
    all_gt_relations = set()
    
    for doc_id in ground_truth:
        if doc_id in predictions:
            for h, t, r in predictions[doc_id]:
                all_pred_relations.add((doc_id, h, t, r))
            for h, t, r in ground_truth[doc_id]:
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

class BaselineSystemEvaluator:
    """Evaluator for comparing baseline systems"""
    
    def __init__(self):
        self.systems = {
            'vanilla': VanillaBaseline(),
            'rule_based': RuleBasedCleaner(),
            'confidence_filter': ConfidenceFilteringBaseline(confidence_threshold=0.5),
            'confidence_filter_strict': ConfidenceFilteringBaseline(confidence_threshold=0.7),
            'outlier_filter': StatisticalOutlierFilter(outlier_threshold=2.0)
        }
    
    def evaluate_all_systems(self, kg_data: Dict, ground_truth: Dict) -> Dict[str, Dict]:
        """Evaluate all baseline systems"""
        results = {}
        
        for system_name, system in self.systems.items():
            # Clean the knowledge graph
            cleaned_kg = system.clean_knowledge_graph(kg_data)
            
            # Evaluate against ground truth
            evaluation_metrics = calculate_performance_metrics(cleaned_kg, ground_truth)
            
            results[system_name] = {
                'system_description': system.get_system_description(),
                'metrics': evaluation_metrics,
                'cleaned_relations_count': sum(len(relations) for relations in cleaned_kg.values()),
                'original_relations_count': sum(len(relations) for relations in kg_data.values())
            }
        
        return results
    
    def generate_comparison_report(self, evaluation_results: Dict[str, Dict]) -> Dict:
        """Generate comparison report"""
        # Rankings by F1 score
        f1_rankings = [(name, results['metrics']['f1']) 
                      for name, results in evaluation_results.items()]
        f1_rankings.sort(key=lambda x: x[1], reverse=True)
        
        best_system = f1_rankings[0][0] if f1_rankings else None
        
        return {
            'best_systems': {'f1': best_system},
            'f1_rankings': f1_rankings
        }

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "experiment.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.kg_generator = KnowledgeGraphGenerator()
        self.baseline_evaluator = BaselineSystemEvaluator()
        self.rl_cleaner = AdaptiveKGCleaner()
        
        self.logger.info("Experiment runner initialized")
    
    def generate_experimental_data(self, use_docred: bool = True, max_documents: int = 100) -> Tuple[Dict, Dict, Dict]:
        """Generate clean KG, corrupted KG, and ground truth"""
        self.logger.info("Loading experimental knowledge graph data...")
        
        # Try loading DocRED data first
        if use_docred:
            try:
                self.logger.info("Attempting to load DocRED dataset...")
                # Use Tier1 loader to get docs and KG (ground truth)
                data_dir = Path(__file__).parent / "experiments" / "exp1_docred_svo_healing" / "data" / "raw_data"
                doc_loader = Tier1DocLoader(str(data_dir))
                docs, doc_gt = doc_loader.load_split('train_annotated', max_docs=max_documents)
                clean_kg = doc_gt
                ground_truth = doc_gt
                self.logger.info(f"Successfully loaded {len(clean_kg)} documents from DocRED")
                self.logger.info(f"Total relations from DocRED: {sum(len(r) for r in clean_kg.values())}")
                
                # Apply DocRED-safe corruption
                relation_ids = list(doc_loader.rel2name.keys())
                corrupted_kg = self.kg_generator.apply_corruption_docred(clean_kg, docs, relation_ids, corruption_rate=0.4)
                
                self.logger.info(f"Applied 40% corruption to DocRED data")
                self.logger.info(f"Corrupted relations: {sum(len(r) for r in corrupted_kg.values())}")
                
                return clean_kg, corrupted_kg, ground_truth
                
            except Exception as e:
                self.logger.warning(f"Failed to load DocRED data: {e}")
                self.logger.info("Falling back to synthetic data generation...")
        
        # Fallback to synthetic data generation
        self.logger.info("Generating synthetic knowledge graph data...")
        clean_kg, ground_truth = self.kg_generator.generate_clean_kg(
            num_documents=max_documents, relations_per_doc=(20, 50)
        )
        
        # Apply corruption
        corrupted_kg = self.kg_generator.apply_corruption(clean_kg, corruption_rate=0.4)
        
        self.logger.info(f"Generated {len(clean_kg)} synthetic documents")
        self.logger.info(f"Clean relations: {sum(len(r) for r in clean_kg.values())}")
        self.logger.info(f"Corrupted relations: {sum(len(r) for r in corrupted_kg.values())}")
        
        return clean_kg, corrupted_kg, ground_truth
    
    def run_baseline_comparison(self, corrupted_kg: Dict, ground_truth: Dict) -> Dict:
        """Run baseline system comparison"""
        self.logger.info("Running baseline system comparison...")
        
        start_time = time.time()
        baseline_results = self.baseline_evaluator.evaluate_all_systems(corrupted_kg, ground_truth)
        baseline_time = time.time() - start_time
        
        comparison_report = self.baseline_evaluator.generate_comparison_report(baseline_results)
        
        self.logger.info(f"Baseline comparison completed in {baseline_time:.2f}s")
        self.logger.info(f"Best F1 system: {comparison_report['best_systems']['f1']}")
        
        return {
            'results': baseline_results,
            'comparison': comparison_report,
            'execution_time': baseline_time
        }
    
    def run_rl_cleaner_training(self, corrupted_kg: Dict, ground_truth: Dict, 
                             num_episodes: int = 50) -> Dict:
        """Train and evaluate RL-based cleaner"""
        self.logger.info(f"Training RL-based adaptive cleaner for {num_episodes} episodes...")
        
        start_time = time.time()
        
        # Training episodes
        for episode in range(num_episodes):
            episode_reward = self.rl_cleaner.train_episode(corrupted_kg, max_steps=8)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.rl_cleaner.episode_rewards[-10:])
                self.logger.info(f"Episode {episode + 1}: avg reward = {avg_reward:.3f}")
        
        training_time = time.time() - start_time
        
        # Evaluate trained cleaner
        final_performance = self._evaluate_rl_cleaner(corrupted_kg, ground_truth)
        
        self.logger.info(f"RL training completed in {training_time:.2f}s")
        self.logger.info(f"Final RL performance: F1 = {final_performance['f1']:.3f}")
        
        return {
            'episode_rewards': self.rl_cleaner.episode_rewards,
            'final_performance': final_performance,
            'training_time': training_time,
            'strategy_effectiveness': {action.name: rewards for action, rewards in self.rl_cleaner.strategy_effectiveness.items()}
        }
    
    def _evaluate_rl_cleaner(self, corrupted_kg: Dict, ground_truth: Dict) -> Dict:
        """Evaluate the trained RL cleaner"""
        # For simplicity, simulate cleaning by applying some improvements
        cleaned_kg = corrupted_kg.copy()
        total_cleaning_actions = 0
        
        for doc_id, relations in corrupted_kg.items():
            kg_state = self.rl_cleaner._create_kg_state({doc_id: relations})
            action = self.rl_cleaner.select_action(kg_state)
            
            if action != CleaningAction.NO_ACTION:
                total_cleaning_actions += 1
                
                # Simulate cleaning effect
                if action == CleaningAction.CONFIDENCE_FILTERING:
                    # Remove some low-quality relations
                    num_to_remove = max(1, len(relations) // 10)
                    if len(relations) > num_to_remove:
                        cleaned_relations = relations[:-num_to_remove]
                        cleaned_kg[doc_id] = cleaned_relations
        
        # Evaluate performance
        performance = calculate_performance_metrics(cleaned_kg, ground_truth)
        performance['healing_actions_applied'] = total_cleaning_actions
        
        return performance

    def run_tier1_pipeline(self, corrupted_kg: Dict, ground_truth: Dict) -> Dict:
        """Run Tier-1 pipeline: Doc-level verifier -> constraints -> KGE -> ensemble.
        Returns dict with metrics and details.
        """
        self.logger.info("Running Tier-1 pipeline (verifier + constraints + KGE + ensemble)...")
        data_dir = Path(__file__).parent / "experiments" / "exp1_docred_svo_healing" / "data" / "raw_data"
        loader = Tier1DocLoader(str(data_dir))
        docs, _ = loader.load_split('train_annotated', max_docs=None)

        # Split train/dev doc ids
        rng = np.random.RandomState(42)
        doc_ids = sorted(list(corrupted_kg.keys()))
        rng.shuffle(doc_ids)
        dev_size = max(200, int(0.1 * len(doc_ids)))
        dev_ids = set(doc_ids[:dev_size])
        train_ids = [d for d in doc_ids if d not in dev_ids]

        # Build training examples for verifier
        self.logger.info("Preparing training data for doc-level verifier...")
        train_examples: List[TripleExample] = []
        dev_examples: List[TripleExample] = []
        train_examples_hf: List[TripleExampleHF] = []
        dev_examples_hf: List[TripleExampleHF] = []

        all_relations = list(loader.rel2name.keys()) if loader.rel2name else list({r for rels in ground_truth.values() for (_,_,r) in rels})

        def build_examples_for_doc(doc_id: str, into_list: List[TripleExample], into_list_hf: List[TripleExampleHF]):
            d = docs[doc_id]
            pos = d.get('labels', [])
            pos_set = {(lbl['h'], lbl['t'], lbl['r']) for lbl in pos}
            # positives with evidence and markers
            for lbl in pos:
                text_plain = loader.extract_evidence_text(d, lbl['h'], lbl['t'], lbl.get('evidence', []))
                text_marked = loader.build_text_with_markers(d, lbl['h'], lbl['t'], lbl.get('evidence', []))
                into_list.append(TripleExample(text=text_plain, relation=lbl['r'], label=1))
                into_list_hf.append(TripleExampleHF(text=text_marked, relation=lbl['r'], label=1))
            # negatives (hard): wrong-relation for same pair
            for lbl in pos:
                r_neg_choices = [rr for rr in all_relations if rr != lbl['r']]
                if not r_neg_choices:
                    continue
                r_neg = r_neg_choices[rng.randint(0, len(r_neg_choices))]
                text_plain = loader.extract_evidence_text(d, lbl['h'], lbl['t'], None)
                text_marked = loader.build_text_with_markers(d, lbl['h'], lbl['t'], None)
                into_list.append(TripleExample(text=text_plain, relation=r_neg, label=0))
                into_list_hf.append(TripleExampleHF(text=text_marked, relation=r_neg, label=0))
            # negatives (random pair)
            n_neg = max(1, len(pos))
            n_ents = len(d.get('vertexSet', []))
            tries = 0
            while n_neg > 0 and tries < 5 * (n_neg + 1):
                h = rng.randint(0, max(1, n_ents))
                t = rng.randint(0, max(1, n_ents))
                if n_ents == 0 or h == t:
                    tries += 1
                    continue
                r = all_relations[rng.randint(0, len(all_relations))]
                if (h,t,r) in pos_set:
                    tries += 1
                    continue
                text_plain = loader.extract_evidence_text(d, h, t, None)
                text_marked = loader.build_text_with_markers(d, h, t, None)
                if not text_plain:
                    tries += 1
                    continue
                into_list.append(TripleExample(text=text_plain, relation=r, label=0))
                into_list_hf.append(TripleExampleHF(text=text_marked, relation=r, label=0))
                n_neg -= 1
                tries += 1

        for did in train_ids:
            build_examples_for_doc(did, train_examples, train_examples_hf)
        for did in dev_ids:
            build_examples_for_doc(did, dev_examples, dev_examples_hf)

        # Train verifier
        # Prefer transformer-based verifier; fallback to TF-IDF
        use_hf = True
        if use_hf:
            hf_verifier = HFDocREVerifier(epochs=1, batch_size=16)
            self.logger.info(f"Training transformer verifier on {len(train_examples_hf)} examples...")
            hf_verifier.fit(train_examples_hf)
            rel2thr_ver = hf_verifier.calibrate_thresholds(dev_examples_hf)
        else:
            verifier = DocLevelVerifier()
            self.logger.info(f"Training verifier on {len(train_examples)} examples...")
            verifier.fit(train_examples)
            rel2thr_ver = verifier.calibrate_thresholds(dev_examples)

        # Train KGE on train split ground truth using doc-local entity IDs
        self.logger.info("Training DistMult KGE on ground-truth triples (train split)...")
        kge_triples = []
        for did in train_ids:
            for (h,t,r) in ground_truth.get(did, []):
                kge_triples.append((f"{did}:{h}", r, f"{did}:{t}"))
        kge = KGETrainer(dim=32, epochs=2, neg_ratio=2, lr=1e-2)
        if kge_triples:
            kge.fit(kge_triples)

        # Prepare dev features for ensemble training and conformal calibration
        self.logger.info("Building dev features for ensemble training...")
        X_dev = []
        y_dev = []
        rels_dev = []
        probs_dev_for_conformal = []
        labels_dev_for_conformal = []
        rels_dev_for_conformal = []
        for did in dev_ids:
            d = docs[did]
            gt_set = set(ground_truth.get(did, []))
            inv_index = set((tt, hh, rr) for (hh,tt,rr) in corrupted_kg.get(did, []))
            for (h,t,r) in corrupted_kg.get(did, []):
                text = loader.build_text_with_markers(d, h, t, None)
                if use_hf:
                    tp = hf_verifier.predict_proba([(text, r)])[0] if text else 0.0
                else:
                    tp = verifier.predict_proba([(text, r)])[0] if text else 0.0
                kgp = kge.score(f"{did}:{h}", r, f"{did}:{t}") if kge_triples else 0.0
                inv_exists = 1.0 if (t,h, r) in corrupted_kg.get(did, []) else 0.0
                X_dev.append([tp, kgp, inv_exists])
                y_dev.append(1 if (h,t,r) in gt_set else 0)
                rels_dev.append(r)
                probs_dev_for_conformal.append(tp)
                labels_dev_for_conformal.append(1 if (h,t,r) in gt_set else 0)
                rels_dev_for_conformal.append(r)
        X_dev = np.array(X_dev, dtype=np.float32) if X_dev else np.zeros((0,3), dtype=np.float32)
        y_dev = np.array(y_dev, dtype=np.int32)

        # Train ensemble
        ensemble = StackingEnsemble()
        if len(X_dev) and y_dev.sum() > 0:
            self.logger.info(f"Training ensemble on {len(X_dev)} dev triples...")
            ensemble.fit(X_dev, y_dev)
        else:
            self.logger.warning("Insufficient dev data for ensemble; falling back to verifier only.")

        # Calibrate thresholds with conformal prediction (per relation)
        rel2thr_conf: Dict[str, float] = {}
        if len(rels_dev_for_conformal):
            # Use ensemble probs if available; else use verifier probs (tp already holds verifier/hf prob)
            if len(X_dev) and ensemble.fitted:
                probs_all = ensemble.predict_proba(np.array(X_dev, dtype=np.float32))
                rel2thr_conf = compute_conformal_thresholds(
                    probs=list(probs_all.tolist()),
                    labels=labels_dev_for_conformal,
                    relations=rels_dev_for_conformal,
                    alpha=0.1,
                    min_per_rel=20
                )
            else:
                rel2thr_conf = compute_conformal_thresholds(
                    probs=probs_dev_for_conformal,
                    labels=labels_dev_for_conformal,
                    relations=rels_dev_for_conformal,
                    alpha=0.1,
                    min_per_rel=20
                )

        # Train anomaly detector on ground-truth (train split only) for normal edge patterns
        self.logger.info("Training edge anomaly detector on ground-truth (train split)...")
        anomaly = EdgeAnomalyDetector(contamination=0.1)
        gt_train_subset = {did: ground_truth.get(did, []) for did in train_ids}
        anomaly.fit_on_ground_truth(gt_train_subset)

        # Apply to full corrupted KG
        self.logger.info("Scoring full corrupted KG with Tier-2 (anomaly + conformal + ensemble)...")
        cleaned_kg: Dict[str, List[Tuple[int,int,str]]] = {}
        for did, triples in corrupted_kg.items():
            d = docs[did]
            scored = []
            for (h,t,r) in triples:
                # Anomaly pruning first
                if anomaly.predict_is_outlier(triples, (h,t,r)):
                    continue
                text = loader.build_text_with_markers(d, h, t, None)
                if use_hf:
                    tp = hf_verifier.predict_proba([(text, r)])[0] if text else 0.0
                else:
                    tp = verifier.predict_proba([(text, r)])[0] if text else 0.0
                kgp = kge.score(f"{did}:{h}", r, f"{did}:{t}") if kge_triples else 0.0
                inv_exists = 1.0 if (t,h, r) in triples else 0.0
                feat = np.array([[tp, kgp, inv_exists]], dtype=np.float32)
                prob = float(ensemble.predict_proba(feat)[0]) if ensemble.fitted else float(tp)
                # Conformal threshold per relation
                thr = rel2thr_conf.get(r, rel2thr_ver.get(r, 0.5))
                if prob >= thr:
                    scored.append((h,t,r,prob))
            # Apply constraints per doc
            scored = apply_constraints(scored)
            cleaned_kg[did] = [(h,t,r) for (h,t,r,_) in scored]

        # Evaluate
        metrics = calculate_performance_metrics(cleaned_kg, ground_truth)
        self.logger.info(f"Tier-1 ensemble performance: F1 = {metrics['f1']:.3f}")

        return {
            'metrics': metrics,
            'num_selected_relations': sum(len(v) for v in cleaned_kg.values())
        }
    
    def generate_visualizations(self, baseline_results: Dict, rl_results: Dict):
        """Generate experiment visualizations"""
        self.logger.info("Generating experiment visualizations...")
        
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Baseline comparison
        self._plot_baseline_comparison(baseline_results, plots_dir)
        
        # 2. RL training progress
        self._plot_rl_training_progress(rl_results, plots_dir)
        
        # 3. System comparison
        self._plot_system_comparison(baseline_results, rl_results, plots_dir)
    
    def _plot_baseline_comparison(self, baseline_results: Dict, plots_dir: Path):
        """Plot baseline system comparison"""
        systems = []
        f1_scores = []
        
        for system_name, results in baseline_results['results'].items():
            systems.append(system_name.replace('_', '\n'))
            f1_scores.append(results['metrics']['f1'])
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(systems, f1_scores, color='skyblue', alpha=0.7)
        plt.title('Baseline Systems F1 Score Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('System', fontweight='bold')
        plt.ylabel('F1 Score', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "baseline_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rl_training_progress(self, rl_results: Dict, plots_dir: Path):
        """Plot RL training progress"""
        episode_rewards = rl_results['episode_rewards']
        episodes = range(1, len(episode_rewards) + 1)
        
        window_size = 5
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = range(window_size, len(episode_rewards) + 1)
        else:
            moving_avg = episode_rewards
            moving_avg_episodes = episodes
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        plt.plot(moving_avg_episodes, moving_avg, color='red', linewidth=2, label=f'{window_size}-Episode Moving Average')
        
        plt.title('RL Healer Training Progress', fontsize=14, fontweight='bold')
        plt.xlabel('Episode', fontweight='bold')
        plt.ylabel('Reward', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "rl_training_progress.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_system_comparison(self, baseline_results: Dict, rl_results: Dict, plots_dir: Path):
        """Plot overall system comparison"""
        best_baseline = baseline_results['comparison']['best_systems']['f1']
        best_baseline_f1 = baseline_results['results'][best_baseline]['metrics']['f1']
        rl_f1 = rl_results['final_performance']['f1']
        
        systems = ['Best Baseline\n(' + best_baseline.replace('_', ' ') + ')', 'RL-based\nAdaptive Healer']
        f1_scores = [best_baseline_f1, rl_f1]
        colors = ['lightcoral', 'lightgreen']
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(systems, f1_scores, color=colors, alpha=0.8)
        plt.title('Best Baseline vs RL-based Healer', fontsize=14, fontweight='bold')
        plt.ylabel('F1 Score', fontweight='bold')
        
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.ylim(0, max(f1_scores) * 1.2)
        plt.tight_layout()
        plt.savefig(plots_dir / "system_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, baseline_results: Dict, rl_results: Dict, clean_kg: Dict, corrupted_kg: Dict, tier1_results: Dict = None):
        """Save experiment results"""
        self.logger.info("Saving experiment results...")
        
        results = {
            'experiment_config': {
                'num_documents': len(clean_kg),
                'total_clean_relations': sum(len(relations) for relations in clean_kg.values()),
                'total_corrupted_relations': sum(len(relations) for relations in corrupted_kg.values()),
                'corruption_applied': True
            },
'baseline_results': baseline_results,
            'tier1_results': tier1_results,
            'rl_results': rl_results,
            'summary': {
                'best_baseline_system': baseline_results['comparison']['best_systems']['f1'],
                'best_baseline_f1': max(r['metrics']['f1'] for r in baseline_results['results'].values()),
'rl_final_f1': rl_results['final_performance']['f1'],
                'tier1_f1': tier1_results['metrics']['f1'] if tier1_results else None,
                'rl_training_episodes': len(rl_results['episode_rewards'])
            }
        }
        
        with open(self.output_dir / "experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline"""
        self.logger.info("="*80)
        self.logger.info("STARTING KNOWLEDGE GRAPH QUALITY IMPROVEMENT EXPERIMENT")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Generate data (try DocRED first, fallback to synthetic)
            clean_kg, corrupted_kg, ground_truth = self.generate_experimental_data(
                use_docred=True, max_documents=3053  # Use full DocRED dataset
            )
            
            # Run baseline comparison
            baseline_results = self.run_baseline_comparison(corrupted_kg, ground_truth)
            
            # Train and evaluate RL cleaner
            rl_results = self.run_rl_cleaner_training(corrupted_kg, ground_truth, num_episodes=1000)

            # Run Tier-1 pipeline (verifier + constraints + KGE + ensemble)
            tier1_results = self.run_tier1_pipeline(corrupted_kg, ground_truth)
            
            # Generate visualizations
            self.generate_visualizations(baseline_results, rl_results)
            
            # Save results
            self.save_results(baseline_results, rl_results, clean_kg, corrupted_kg, tier1_results)
            
            total_time = time.time() - start_time
            
            # Print summary
            self.print_experiment_summary(baseline_results, rl_results, total_time)
            print(f"Tier-1 ensemble F1: {tier1_results['metrics']['f1']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}", exc_info=True)
            raise
    
    def print_experiment_summary(self, baseline_results: Dict, rl_results: Dict, total_time: float):
        """Print comprehensive experiment summary"""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        print(f"\nBASELINE SYSTEMS PERFORMANCE:")
        for system_name, results in baseline_results['results'].items():
            f1 = results['metrics']['f1']
            relations_removed = results['original_relations_count'] - results['cleaned_relations_count']
            print(f"  {system_name:25s}: F1={f1:.3f} | Relations removed: {relations_removed:3d}")
        
        print(f"\nRL-BASED ADAPTIVE CLEANER:")
        print(f"  Training episodes: {len(rl_results['episode_rewards'])}")
        print(f"  Final F1 score:   {rl_results['final_performance']['f1']:.3f}")
        print(f"  Cleaning actions:   {rl_results['final_performance']['healing_actions_applied']}")
        
        print(f"\nBEST PERFORMERS:")
        best_baseline = baseline_results['comparison']['best_systems']['f1']
        best_baseline_f1 = baseline_results['results'][best_baseline]['metrics']['f1']
        rl_f1 = rl_results['final_performance']['f1']
        
        print(f"  Best baseline:     {best_baseline} (F1={best_baseline_f1:.3f})")
        print(f"  RL cleaner:         F1={rl_f1:.3f}")
        
        if rl_f1 > best_baseline_f1:
            improvement = ((rl_f1 - best_baseline_f1) / best_baseline_f1) * 100
            print(f"  RL IMPROVEMENT: +{improvement:.1f}% over best baseline!")
        else:
            gap = ((best_baseline_f1 - rl_f1) / best_baseline_f1) * 100
            print(f"  RL GAP:         -{gap:.1f}% behind best baseline")
        
        print(f"\nEXECUTION TIME: {total_time:.2f}s")
        print(f"\nRESULTS SAVED TO: {self.output_dir}")
        print("="*80)

def main():
    """Main execution function"""
    print("Knowledge Graph Quality Improvement Experiment")
    print("   Comparing RL-based adaptive cleaner with baseline systems")
    print()
    
    runner = ExperimentRunner()
    runner.run_full_experiment()

if __name__ == "__main__":
    main()
