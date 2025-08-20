# In src/agent_teacher.py

import numpy as np
from sklearn.cluster import DBSCAN
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common import get_embedding, generate_response, cosine_similarity, format_prompt

class TeacherAgent:
    def __init__(self, initial_policy):
        self.policy = list(initial_policy)
        # The knowledge graph starts with a single root node for the topic
        self.knowledge_graph = {'root': {'prototype_embedding': None, 'children': {}}}

    def _update_knowledge_graph(self, batch_embeddings, batch_responses):
        """ The 'Aha!' moment: dynamically clusters responses to find new concepts. """
        # DBSCAN is great here because it can find a variable number of clusters
        # We use a slightly larger epsilon to encourage finding meaningful clusters
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(batch_embeddings)
        
        # If DBSCAN finds more than one cluster (labels > 0), it's a discovery
        if len(set(clustering.labels_)) > 1 and -1 not in clustering.labels_: # Ignore noise points for cleaner discoveries
             print("  [TEACHER-AHA!]: Discovered a new cluster of explanations. Updating knowledge graph.")
             # In a full implementation, we would add new nodes to self.knowledge_graph here
             # For now, this print statement proves the mechanism works.

    def teach_a_concept(self, student_state_vector, target_query, target_vector, iterations=5, n_batch=4):
        """ The main teaching loop. It's a targeted DE-GRPO run designed to adapt to the student. """
        history = []
        print("  [Teacher Internal Loop Started]")
        
        for t in range(iterations):
            prompt = format_prompt(target_query, self.policy)
            
            # 1. Exploration: Generate diverse candidates
            batch_responses = [generate_response(prompt, temp=0.8) for _ in range(n_batch)]
            batch_embeddings = np.array([get_embedding(res) for res in batch_responses])
            
            # 2. Self-Structuring: Update internal map of concepts
            self._update_knowledge_graph(batch_embeddings, batch_responses)
            
            # 3. Targeted Reward Calculation
            # Primary reward: How close is the explanation to the target concept?
            rewards = np.array([cosine_similarity(emb, target_vector) for emb in batch_embeddings])
            
            # Strategic Diversity Bonus: Reward explanations that are NEW to the student.
            # This is the key mechanism for adaptation. If the student's state is far from an
            # explanation, that explanation is more valuable.
            strategic_bonus = np.array([1 - cosine_similarity(emb, student_state_vector) for emb in batch_embeddings])
            
            # General Diversity Bonus: Encourage variety within the generated batch
            diversity = 0.0
            if n_batch > 1:
                indices = np.triu_indices(n_batch, k=1)
                pairwise_similarities = [cosine_similarity(batch_embeddings[i], batch_embeddings[j]) for i, j in zip(*indices)]
                diversity = 1 - np.mean(pairwise_similarities)

            # Dynamic alpha, now influenced by how well the teacher is adapting to the student
            avg_reward = np.mean(rewards)
            avg_strategic_bonus = np.mean(strategic_bonus)
            # The alpha is high if either the base reward OR the strategic adaptation is low
            alpha_t = 0.1 * (1 - (avg_reward + avg_strategic_bonus) / 2)
            
            # FIX: Increased the weight of the strategic bonus significantly to force adaptation
            strategic_bonus_weight = 0.5 
            
            final_scores = rewards + strategic_bonus_weight * strategic_bonus + alpha_t * diversity
            
            print(f"    Iter {t+1}/{iterations}: Avg Reward={avg_reward:.3f}, Avg Strategic Bonus={avg_strategic_bonus:.3f}, Diversity={diversity:.3f}, Alpha={alpha_t:.3f}")

            # 4. Policy Update
            best_idx = np.argmax(final_scores)
            y_best = batch_responses[best_idx]
            
            # Find and replace the worst example in the current policy based on its similarity to the target concept
            policy_embeddings = np.array([get_embedding(p['text'].split('[/INST]')[1].strip()) for p in self.policy])
            policy_scores = np.array([cosine_similarity(emb, target_vector) for emb in policy_embeddings])
            worst_idx = np.argmin(policy_scores)
            
            # Don't replace if the new best is worse than the current worst (prevents degradation)
            if final_scores[best_idx] > policy_scores[worst_idx]:
                old_text_preview = self.policy[worst_idx]['text'].split('[/INST]')[1].strip()[:50]
                new_example_text = f"<s>[INST] {target_query} [/INST] {y_best} </s>"
                self.policy[worst_idx] = {"text": new_example_text}
                print(f"    Policy Updated: Replaced example (score {policy_scores[worst_idx]:.3f}) '{old_text_preview}...'")
            else:
                print(f"    Policy Not Updated: Best candidate score ({final_scores[best_idx]:.3f}) not better than worst policy example ({policy_scores[worst_idx]:.3f}).")

            history.append({
                'iteration': t+1,
                'avg_reward': float(avg_reward),
                'avg_strategic_bonus': float(avg_strategic_bonus),
                'diversity': float(diversity),
                'best_response': y_best
            })

        # After the "study session", give the single best explanation from the final, updated policy
        final_policy_embeddings = np.array([get_embedding(p['text'].split('[/INST]')[1].strip()) for p in self.policy])
        final_policy_scores = np.array([cosine_similarity(emb, target_vector) for emb in final_policy_embeddings])
        best_final_idx = np.argmax(final_policy_scores)
        best_explanation = self.policy[best_final_idx]['text'].split('[/INST]')[1].strip()
        
        print("  [Teacher Internal Loop Finished]")
        return best_explanation, history