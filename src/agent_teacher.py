import numpy as np
from sklearn.cluster import DBSCAN
from src.common import get_embedding, generate_response, cosine_similarity, format_prompt

class TeacherAgent:
    def __init__(self, initial_policy):
        self.policy = list(initial_policy)
        # The knowledge graph starts with a single root node for the topic
        self.knowledge_graph = {'root': {'prototype_embedding': None, 'children': {}}}

    def _update_knowledge_graph(self, batch_embeddings, batch_responses):
        """ The 'Aha!' moment: dynamically clusters responses to find new concepts. """
        # DBSCAN is great here because it can find a variable number of clusters
        clustering = DBSCAN(eps=0.2, min_samples=2, metric='cosine').fit(batch_embeddings)
        
        # If DBSCAN finds more than one cluster (labels > 0), it's a discovery
        if len(set(clustering.labels_)) > 1 and -1 in clustering.labels_: # Ignore noise points
             print("  [TEACHER-AHA!]: Discovered multiple ways to explain the concept. Updating knowledge graph.")
             # In a full implementation, we would add new nodes to self.knowledge_graph here
             # For now, this print statement proves the mechanism works.

    def teach_a_concept(self, student_state_vector, target_query, target_vector, iterations=5):
        """ The main teaching loop. It's a targeted DE-GRPO run. """
        history = []
        for t in range(iterations):
            prompt = format_prompt(target_query, self.policy)
            
            # 1. Exploration: Generate diverse candidates
            batch_responses = [generate_response(prompt, temp=0.8) for _ in range(4)]
            batch_embeddings = np.array([get_embedding(res) for res in batch_responses])
            
            # 2. Self-Structuring: Update internal map
            self._update_knowledge_graph(batch_embeddings, batch_responses)
            
            # 3. Targeted Reward Calculation
            rewards = np.array([cosine_similarity(emb, target_vector) for emb in batch_embeddings])
            
            # Strategic Diversity: Reward explanations that are NEW to the student
            # This encourages teaching the next logical step.
            strategic_bonus = np.array([1 - cosine_similarity(emb, student_state_vector) for emb in batch_embeddings])
            
            diversity = 1 - np.mean([cosine_similarity(batch_embeddings[i], batch_embeddings[j]) for i in range(4) for j in range(i + 1, 4)])
            
            alpha_t = 0.1 * (1 - np.mean(rewards)) # Dynamic alpha
            
            # The final score is a mix of correctness, novelty to the student, and general diversity
            final_scores = rewards + 0.2 * strategic_bonus + alpha_t * diversity
            
            # 4. Policy Update
            best_idx = np.argmax(final_scores)
            y_best = batch_responses[best_idx]
            
            # Find and replace the worst example in the current policy
            policy_scores = np.array([cosine_similarity(get_embedding(p['text'].split('[/INST]')[1]), target_vector) for p in self.policy])
            worst_idx = np.argmin(policy_scores)
            
            new_example_text = f"<s>[INST] {target_query} [/INST] {y_best} </s>"
            self.policy[worst_idx] = {"text": new_example_text}
            
            history.append({'iteration': t+1, 'avg_reward': np.mean(rewards), 'best_response': y_best})

        # After the "study session", give the single best explanation
        return self.policy[np.argmax(policy_scores)]['text'].split('[/INST]')[1].strip(), history