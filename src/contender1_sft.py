import random
from tqdm import tqdm
from src.common import get_embedding, generate_response, cosine_similarity, format_prompt

def run_sft(expert_dataset, target_query, target_vector, k_samples=2, m_tournaments=10):
    print("\n" + "="*60)
    print("--- Running Contender 1: In-Context SFT (The Expert Mimic) ---")
    print("Finding the best static prompt via tournament selection...")
    print("="*60)

    best_policy = None
    best_score = -1.0

    # Use tqdm for a progress bar over the tournament rounds
    for i in tqdm(range(m_tournaments), desc="SFT Tournament"):
        policy_candidate = random.sample(expert_dataset, k_samples)
        prompt = format_prompt(target_query, policy_candidate)
        
        # We use temp=0.0 for a deterministic response from the policy
        response = generate_response(prompt, temp=0.0)
        
        response_embedding = get_embedding(response)
        score = cosine_similarity(response_embedding, target_vector)
        
        if score > best_score:
            best_score = score
            best_policy = policy_candidate

    print("\n--- SFT Tournament Complete ---")
    print(f"Final Best Score: {best_score:.4f}")

    final_response = generate_response(format_prompt(target_query, best_policy), temp=0.0)
    return final_response, best_policy