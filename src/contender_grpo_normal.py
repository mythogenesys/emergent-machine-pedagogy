import numpy as np
from tqdm import tqdm
from src.common import get_embedding, generate_response, cosine_similarity, format_prompt

# This is the most basic GRPO. It explores but has no diversity metric.
# We expect it to perform poorly, proving that a diversity signal is necessary.
def run_grpo_normal(initial_policy, target_query, target_vector, iterations=10, n_batch=4, temperature=0.8):    
    print("\n" + "="*60)
    print("--- Running Baseline Agent: Normal GRPO (Naive Explorer) ---")
    print("Testing policy optimization with exploration but NO diversity bonus...")
    print("="*60)
    
    policy = list(initial_policy)
    history = [] # NEW: Initialize history log

    for t in tqdm(range(iterations), desc="GRPO (Normal)"):
        prompt = format_prompt(target_query, policy)
        batch_responses = [generate_response(prompt, temp=temperature) for _ in range(n_batch)]
        rewards = np.array([cosine_similarity(get_embedding(res), target_vector) for res in batch_responses])
        final_scores = rewards
        
        # NEW: Log detailed data for this iteration
        history.append({
            "iteration": t + 1,
            "prompt_used": prompt,
            "candidates": [
                {"text": resp, "reward": r} for resp, r in zip(batch_responses, rewards)
            ],
            "avg_reward": np.mean(rewards)
        })

        best_idx = np.argmax(final_scores)
        y_best = batch_responses[best_idx]
        
        policy_scores = np.array([cosine_similarity(get_embedding(p['text'].split('[/INST]')[1]), target_vector) for p in policy])
        worst_idx = np.argmin(policy_scores)
        
        new_example_text = f"<s>[INST] {target_query} [/INST] {y_best} </s>"
        policy[worst_idx] = {"text": new_example_text}

    final_response = generate_response(format_prompt(target_query, policy), temp=0.0)
    return final_response, history # NEW: Return the full history