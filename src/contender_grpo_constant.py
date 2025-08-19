import numpy as np
from tqdm import tqdm
from src.common import get_embedding, generate_response, cosine_similarity, format_prompt

def run_grpo_constant(initial_policy, target_query, target_vector, iterations=10, n_batch=4, temperature=0.8, constant_alpha=0.1):
    print("\n" + "="*60)
    print("--- Running Ablation Agent: GRPO w/ Constant Alpha ---")
    print(f"Testing policy optimization with a fixed diversity weight (alpha={constant_alpha})...")
    print("="*60)
    
    policy = list(initial_policy)
    history = []

    for t in tqdm(range(iterations), desc="GRPO (Constant Alpha)"):
        prompt = format_prompt(target_query, policy)
        batch_responses = [generate_response(prompt, temp=temperature) for _ in range(n_batch)]
        batch_embeddings = np.array([get_embedding(res) for res in batch_responses])
        rewards = np.array([cosine_similarity(emb, target_vector) for emb in batch_embeddings])
        diversity = 1 - np.mean([cosine_similarity(batch_embeddings[i], batch_embeddings[j]) for i in range(n_batch) for j in range(i + 1, n_batch)]) if n_batch > 1 else 0.0
        final_scores = rewards + constant_alpha * diversity
        
        history.append({
            "iteration": t + 1, "prompt_used": prompt, "avg_reward": np.mean(rewards), "diversity": diversity,
            "candidates": [{"text": r, "reward": rw, "final_score": fs} for r, rw, fs in zip(batch_responses, rewards, final_scores)]
        })
        best_idx = np.argmax(final_scores)
        y_best = batch_responses[best_idx]
        policy_scores = np.array([cosine_similarity(get_embedding(p['text'].split('[/INST]')[1]), target_vector) for p in policy])
        worst_idx = np.argmin(policy_scores)
        new_example_text = f"<s>[INST] {target_query} [/INST] {y_best} </s>"
        policy[worst_idx] = {"text": new_example_text}

    final_response = generate_response(format_prompt(target_query, policy), temp=0.0)
    return final_response, history