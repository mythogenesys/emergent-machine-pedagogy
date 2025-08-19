# In src/contender_grpo_clip.py
import numpy as np
from tqdm import tqdm
from src.common import get_embedding, generate_response, cosine_similarity, format_prompt, CLIP_MODEL

def run_grpo_clip(initial_policy, target_query, target_vector, iterations=10, n_batch=4, temperature=0.8, **kwargs):
    print("\n" + "="*60)
    print("--- Running Novel Agent: GRPO w/ CLIP Covariance ---")
    print("Using visual-semantic diversity from CLIP as the exploration signal...")
    print("="*60)
    
    policy = list(initial_policy)
    history = []

    # --- NEW: Define a truncation length for CLIP ---
    # The CLIP model has a max sequence length of 77 tokens.
    # We truncate based on an approximate character count to be safe.
    # 300 chars is roughly 75 tokens.
    CLIP_MAX_CHARS = 300

    for t in tqdm(range(iterations), desc="GRPO (CLIP-Cov)"):
        prompt = format_prompt(target_query, policy)
        batch_responses = [generate_response(prompt, temp=temperature) for _ in range(n_batch)]
        
        text_embeddings = np.array([get_embedding(res) for res in batch_responses])
        rewards = np.array([cosine_similarity(emb, target_vector) for emb in text_embeddings])
        
        # --- FIX: Truncate responses before encoding with CLIP ---
        truncated_responses = [res[:CLIP_MAX_CHARS] for res in batch_responses]
        clip_embeddings = CLIP_MODEL.encode(truncated_responses)
        # --- END FIX ---
        
        diversity = np.trace(np.cov(clip_embeddings, rowvar=False)) if n_batch > 1 else 0.0
        
        # We'll use the degrpo alpha_base from the config if it's passed in kwargs
        alpha_base = kwargs.get('alpha_base', 0.1)
        alpha_t = alpha_base * (1 - np.mean(rewards))
        final_scores = rewards + alpha_t * diversity
        
        history.append({
            "iteration": t + 1, "prompt_used": prompt, "avg_reward": float(np.mean(rewards)), "clip_diversity": float(diversity),
            "candidates": [{"text": r, "reward": float(rw), "final_score": float(fs)} for r, rw, fs in zip(batch_responses, rewards, final_scores)]
        })

        best_idx = np.argmax(final_scores)
        y_best = batch_responses[best_idx]
        policy_scores = np.array([cosine_similarity(get_embedding(p['text'].split('[/INST]')[1]), target_vector) for p in policy])
        worst_idx = np.argmin(policy_scores)
        new_example_text = f"<s>[INST] {target_query} [/INST] {y_best} </s>"
        policy[worst_idx] = {"text": new_example_text}

    final_response = generate_response(format_prompt(target_query, policy), temp=0.0)
    return final_response, history