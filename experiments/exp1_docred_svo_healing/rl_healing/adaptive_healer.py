#!/usr/bin/env python3
"""
Reinforcement Learning-based Adaptive Knowledge Graph Healing
Q-learning approach for adaptive healing strategy selection
"""

import numpy as np
import json
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Any, Optional
import logging
from enum import Enum
import pickle

class HealingAction(Enum):
    """Available healing actions"""
    ENTITY_DISAMBIGUATION = 0
    RELATION_CORRECTION = 1
    EVIDENCE_VALIDATION = 2
    STRUCTURAL_REPAIR = 3
    CONFIDENCE_FILTERING = 4
    NO_ACTION = 5

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

class AdaptiveKGHealer:
    """
    RL-based adaptive knowledge graph healer using Q-learning
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        """
        Initialize the adaptive healer
        
        Args:
            learning_rate: Q-learning learning rate
            discount_factor: Future reward discount factor
            epsilon: Exploration probability
            epsilon_decay: Epsilon decay rate
        """
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
        
        # Healing strategies
        self.healing_strategies = {
            HealingAction.ENTITY_DISAMBIGUATION: self._entity_disambiguation_strategy,
            HealingAction.RELATION_CORRECTION: self._relation_correction_strategy,
            HealingAction.EVIDENCE_VALIDATION: self._evidence_validation_strategy,
            HealingAction.STRUCTURAL_REPAIR: self._structural_repair_strategy,
            HealingAction.CONFIDENCE_FILTERING: self._confidence_filtering_strategy,
            HealingAction.NO_ACTION: self._no_action_strategy
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.healing_history = []
        self.strategy_effectiveness = defaultdict(list)
    
    def select_action(self, state: KGState) -> HealingAction:
        """
        Select healing action using epsilon-greedy policy
        
        Args:
            state: Current KG state
            
        Returns:
            Selected healing action
        """
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.choice(list(HealingAction))
        else:
            # Exploitation: best known action
            return self._get_best_action(state)
    
    def _get_best_action(self, state: KGState) -> HealingAction:
        """Get the best action for a given state"""
        state_key = hash(state)
        if state_key not in self.q_table:
            return random.choice(list(HealingAction))
        
        q_values = self.q_table[state_key]
        best_action_value = max(q_values.values()) if q_values else 0
        best_actions = [action for action, value in q_values.items() if value == best_action_value]
        
        return random.choice(best_actions) if best_actions else random.choice(list(HealingAction))
    
    def execute_healing_action(self, action: HealingAction, kg_data: Dict, 
                             error_analysis: Dict) -> Dict:
        """
        Execute the selected healing action
        
        Args:
            action: Healing action to execute
            kg_data: Current knowledge graph data
            error_analysis: Error analysis results
            
        Returns:
            Healing results and updated KG data
        """
        strategy_function = self.healing_strategies[action]
        return strategy_function(kg_data, error_analysis)
    
    def _entity_disambiguation_strategy(self, kg_data: Dict, error_analysis: Dict) -> Dict:
        """Apply entity disambiguation healing"""
        healing_results = {
            'action': HealingAction.ENTITY_DISAMBIGUATION.name,
            'entities_disambiguated': 0,
            'confidence_improvement': 0.0,
            'execution_time': 0.001  # Placeholder
        }
        
        entity_errors = error_analysis.get('entity_errors', {})
        disambiguation_errors = entity_errors.get('disambiguation_errors', [])
        
        # Apply disambiguation logic
        for error in disambiguation_errors[:5]:  # Limit to 5 errors per action
            # Placeholder disambiguation logic
            healing_results['entities_disambiguated'] += 1
        
        healing_results['confidence_improvement'] = healing_results['entities_disambiguated'] * 0.02
        
        return healing_results
    
    def _relation_correction_strategy(self, kg_data: Dict, error_analysis: Dict) -> Dict:
        """Apply relation correction healing"""
        healing_results = {
            'action': HealingAction.RELATION_CORRECTION.name,
            'relations_corrected': 0,
            'accuracy_improvement': 0.0,
            'execution_time': 0.002
        }
        
        relation_errors = error_analysis.get('relation_errors', {})
        classification_errors = relation_errors.get('classification_errors', [])
        
        # Apply relation correction logic
        for error in classification_errors[:3]:  # Limit to 3 errors per action
            healing_results['relations_corrected'] += 1
        
        healing_results['accuracy_improvement'] = healing_results['relations_corrected'] * 0.03
        
        return healing_results
    
    def _evidence_validation_strategy(self, kg_data: Dict, error_analysis: Dict) -> Dict:
        """Apply evidence validation healing"""
        healing_results = {
            'action': HealingAction.EVIDENCE_VALIDATION.name,
            'evidence_validated': 0,
            'precision_improvement': 0.0,
            'execution_time': 0.003
        }
        
        evidence_errors = error_analysis.get('evidence_errors', {})
        insufficient_evidence = evidence_errors.get('insufficient_evidence', [])
        
        # Apply evidence validation logic
        healing_results['evidence_validated'] = min(len(insufficient_evidence), 4)
        healing_results['precision_improvement'] = healing_results['evidence_validated'] * 0.025
        
        return healing_results
    
    def _structural_repair_strategy(self, kg_data: Dict, error_analysis: Dict) -> Dict:
        """Apply structural repair healing"""
        healing_results = {
            'action': HealingAction.STRUCTURAL_REPAIR.name,
            'structures_repaired': 0,
            'coherence_improvement': 0.0,
            'execution_time': 0.004
        }
        
        structural_errors = error_analysis.get('structural_errors', {})
        inconsistencies = structural_errors.get('inconsistent_relations', [])
        
        # Apply structural repair logic
        healing_results['structures_repaired'] = min(len(inconsistencies), 2)
        healing_results['coherence_improvement'] = healing_results['structures_repaired'] * 0.04
        
        return healing_results
    
    def _confidence_filtering_strategy(self, kg_data: Dict, error_analysis: Dict) -> Dict:
        """Apply confidence-based filtering"""
        healing_results = {
            'action': HealingAction.CONFIDENCE_FILTERING.name,
            'relations_filtered': 0,
            'precision_improvement': 0.0,
            'execution_time': 0.001
        }
        
        # Apply confidence filtering logic
        total_relations = sum(len(relations) for relations in kg_data.get('relations', {}).values())
        low_confidence_threshold = 0.3
        
        # Simulate filtering low-confidence relations
        estimated_filtered = int(total_relations * 0.1)  # Assume 10% are low confidence
        healing_results['relations_filtered'] = estimated_filtered
        healing_results['precision_improvement'] = estimated_filtered * 0.01
        
        return healing_results
    
    def _no_action_strategy(self, kg_data: Dict, error_analysis: Dict) -> Dict:
        """No healing action"""
        return {
            'action': HealingAction.NO_ACTION.name,
            'changes_made': 0,
            'improvement': 0.0,
            'execution_time': 0.0
        }
    
    def calculate_reward(self, prev_state: KGState, action: HealingAction, 
                        new_state: KGState, healing_results: Dict) -> float:
        """
        Calculate reward for the healing action
        
        Args:
            prev_state: Previous KG state
            action: Action taken
            new_state: Resulting KG state
            healing_results: Results of healing action
            
        Returns:
            Reward value
        """
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
        
        # Efficiency penalty (discourage expensive actions if improvement is small)
        execution_time = healing_results.get('execution_time', 0.001)
        efficiency_penalty = execution_time * 10 if performance_reward < 0.01 else 0
        
        # Action-specific bonuses
        action_bonus = 0.0
        if action == HealingAction.NO_ACTION and performance_reward >= 0:
            action_bonus = 0.1  # Small bonus for correctly doing nothing
        elif action != HealingAction.NO_ACTION and performance_reward > 0:
            action_bonus = 0.2  # Bonus for beneficial actions
        
        total_reward = performance_reward + error_reduction_reward - efficiency_penalty + action_bonus
        
        return max(total_reward, -1.0)  # Clip minimum reward
    
    def update_q_value(self, state: KGState, action: HealingAction, reward: float, 
                      next_state: KGState):
        """
        Update Q-value using Q-learning update rule
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = hash(state)
        next_state_key = hash(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for next state
        max_next_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        # Q-learning update
        target = reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target - current_q)
        
        self.q_table[state_key][action] = new_q
    
    def train_episode(self, kg_data: Dict, error_analysis: Dict, max_steps: int = 10) -> float:
        """
        Train the healer for one episode
        
        Args:
            kg_data: Initial knowledge graph data
            error_analysis: Initial error analysis
            max_steps: Maximum steps per episode
            
        Returns:
            Total episode reward
        """
        total_reward = 0.0
        episode_history = []
        
        # Initial state
        current_state = self._create_kg_state(kg_data, error_analysis)
        
        for step in range(max_steps):
            # Select and execute action
            action = self.select_action(current_state)
            healing_results = self.execute_healing_action(action, kg_data, error_analysis)
            
            # Simulate state transition (in practice, you would re-evaluate the KG)
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
            
            # Early stopping if perfect performance or no improvement possible
            if next_state.graph_metrics.get('f1', 0) >= 0.99 or action == HealingAction.NO_ACTION:
                break
            
            current_state = next_state
        
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Store episode data
        self.episode_rewards.append(total_reward)
        self.healing_history.append(episode_history)
        
        return total_reward
    
    def _create_kg_state(self, kg_data: Dict, error_analysis: Dict) -> KGState:
        """Create KGState from current KG data and error analysis"""
        # Extract metrics (placeholder values)
        graph_metrics = {
            'precision': random.uniform(0.6, 0.8),
            'recall': random.uniform(0.5, 0.7),
            'f1': random.uniform(0.55, 0.75),
            'coherence_score': random.uniform(0.4, 0.6)
        }
        
        # Extract error counts
        error_counts = {
            'entity_errors': len(error_analysis.get('entity_errors', {}).get('disambiguation_errors', [])),
            'relation_errors': len(error_analysis.get('relation_errors', {}).get('classification_errors', [])),
            'evidence_errors': len(error_analysis.get('evidence_errors', {}).get('insufficient_evidence', [])),
            'structural_errors': len(error_analysis.get('structural_errors', {}).get('inconsistent_relations', []))
        }
        
        # Generate confidence scores (placeholder)
        confidence_scores = [random.uniform(0.3, 0.9) for _ in range(20)]
        
        return KGState(graph_metrics, error_counts, confidence_scores)
    
    def _simulate_state_transition(self, state: KGState, action: HealingAction, 
                                 healing_results: Dict) -> KGState:
        """Simulate the state transition after applying healing action"""
        new_metrics = state.graph_metrics.copy()
        new_error_counts = state.error_counts.copy()
        new_confidence_scores = state.confidence_scores.copy()
        
        # Apply action effects
        if action == HealingAction.ENTITY_DISAMBIGUATION:
            entities_fixed = healing_results.get('entities_disambiguated', 0)
            new_error_counts['entity_errors'] = max(0, new_error_counts['entity_errors'] - entities_fixed)
            new_metrics['precision'] += entities_fixed * 0.01
            
        elif action == HealingAction.RELATION_CORRECTION:
            relations_fixed = healing_results.get('relations_corrected', 0)
            new_error_counts['relation_errors'] = max(0, new_error_counts['relation_errors'] - relations_fixed)
            new_metrics['f1'] += relations_fixed * 0.02
            
        elif action == HealingAction.EVIDENCE_VALIDATION:
            evidence_fixed = healing_results.get('evidence_validated', 0)
            new_error_counts['evidence_errors'] = max(0, new_error_counts['evidence_errors'] - evidence_fixed)
            new_metrics['recall'] += evidence_fixed * 0.015
            
        elif action == HealingAction.STRUCTURAL_REPAIR:
            structures_fixed = healing_results.get('structures_repaired', 0)
            new_error_counts['structural_errors'] = max(0, new_error_counts['structural_errors'] - structures_fixed)
            new_metrics['coherence_score'] += structures_fixed * 0.03
            
        elif action == HealingAction.CONFIDENCE_FILTERING:
            relations_filtered = healing_results.get('relations_filtered', 0)
            new_metrics['precision'] += relations_filtered * 0.005
            # Improve confidence scores by removing low-confidence ones
            new_confidence_scores = [score for score in new_confidence_scores if score > 0.4]
        
        # Clip metrics to valid ranges
        for metric in ['precision', 'recall', 'f1']:
            new_metrics[metric] = min(1.0, max(0.0, new_metrics[metric]))
        new_metrics['coherence_score'] = min(1.0, max(0.0, new_metrics['coherence_score']))
        
        return KGState(new_metrics, new_error_counts, new_confidence_scores)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'strategy_effectiveness': dict(self.strategy_effectiveness)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.epsilon = model_data['epsilon']
        self.episode_rewards = model_data['episode_rewards']
        self.strategy_effectiveness = defaultdict(list, model_data['strategy_effectiveness'])
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_summary(self) -> Dict:
        """Get summary of training progress"""
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'best_episode_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'current_epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'most_effective_strategy': max(
                self.strategy_effectiveness.keys(),
                key=lambda k: np.mean(self.strategy_effectiveness[k]) if self.strategy_effectiveness[k] else 0
            ).name if self.strategy_effectiveness else 'None',
            'strategy_performance': {
                action.name: np.mean(rewards) if rewards else 0
                for action, rewards in self.strategy_effectiveness.items()
            }
        }

def main():
    """Example usage of adaptive KG healer"""
    healer = AdaptiveKGHealer()
    
    # Example training loop
    for episode in range(100):
        # Mock data - in practice, you would load real KG data and error analysis
        kg_data = {'relations': {'doc1': [(0, 1, 'P1'), (1, 2, 'P2')]}}
        error_analysis = {'entity_errors': {'disambiguation_errors': [1, 2, 3]}}
        
        reward = healer.train_episode(kg_data, error_analysis)
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Reward = {reward:.3f}, Epsilon = {healer.epsilon:.3f}")
    
    # Print training summary
    summary = healer.get_training_summary()
    print("Training Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
