# in src/contender2_degrpo.py
import numpy as np
from tqdm import tqdm
from src.common import get_embedding, generate_response, cosine_similarity, format_prompt

def run_degrpo(initial_policy, target_query, target_vector, iterations=10, n_batch=4, temperature=0.8, alpha_base=0.1, **kwargs):
    print("\n" + "="*60)
    print("--- Running Main Agent: DE-GRPO (Dynamic Entropy) ---")
    print("Iteratively improving the policy with a dynamic, state-aware entropy controller...")
    print("="*60)
    
    policy = list(initial_policy)
    history = []

    for t in tqdm(range(iterations), desc="DE-GRPO (Dynamic)"):
        prompt = format_prompt(target_query, policy)
        batch_responses = [generate_response(prompt, temp=temperature) for _ in range(n_batch)]
        batch_embeddings = np.array([get_embedding(res) for res in batch_responses])
        rewards = np.array([cosine_similarity(emb, target_vector) for emb in batch_embeddings])
        
        # Calculate diversity - handle the case of a single response in a batch
        if n_batch > 1:
            # Create all unique pairs of indices
            indices = np.triu_indices(n_batch, k=1)
            # Calculate cosine similarity for all pairs
            pairwise_similarities = [cosine_similarity(batch_embeddings[i], batch_embeddings[j]) for i, j in zip(*indices)]
            diversity = 1 - np.mean(pairwise_similarities)
        else:
            diversity = 0.0

        alpha_t = alpha_base * (1 - np.mean(rewards))
        final_scores = rewards + alpha_t * diversity
        
        # --- FIX: Cast all numpy types to standard Python floats before logging ---
        history.append({
            "iteration": t + 1,
            "prompt_used": prompt,
            "avg_reward": float(np.mean(rewards)),
            "diversity": float(diversity),
            "alpha_t": float(alpha_t),
            "candidates": [{"text": r, "reward": float(rw), "final_score": float(fs)} 
                        for r, rw, fs in zip(batch_responses, rewards, final_scores)]
        })
        # --- END FIX ---

        best_idx = np.argmax(final_scores)
        y_best = batch_responses[best_idx]
        policy_scores = np.array([cosine_similarity(get_embedding(p['text'].split('[/INST]')[1]), target_vector) for p in policy])
        worst_idx = np.argmin(policy_scores)
        new_example_text = f"<s>[INST] {target_query} [/INST] {y_best} </s>"
        policy[worst_idx] = {"text": new_example_text}

    final_response = generate_response(format_prompt(target_query, policy), temp=0.0)
    return final_response, history