import os
import json
import time
import argparse
import yaml
from src.common import get_embedding
from src.contender1_sft import run_sft
from src.contender_grpo_normal import run_grpo_normal
from src.contender_grpo_constant import run_grpo_constant
from src.contender_grpo_clip import run_grpo_clip
from src.contender2_degrpo import run_degrpo

def load_config(config_path, mode):
    """Loads config from YAML and applies dry-run overrides if specified."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"--- Running in '{mode}' mode ---")
    if mode == "dry-run":
        # Override base settings with dry-run settings
        print("Applying dry-run configuration overrides.")
        config['sft'].update(config['dry_run']['sft'])
        config['grpo_base'].update(config['dry_run']['grpo_base'])
        # Add other overrides as needed
        
    return config

def run_experiment_for_task(task_name: str, task_query: str, target_concept: str, config: dict):
    print(f"\n{'='*20} RUNNING EXPERIMENT SUITE FOR TASK: {task_name.upper()} {'='*20}")

    # --- Setup ---
    expert_data_path = f'src/data/{task_name}_expert_data.jsonl'
    with open(expert_data_path, 'r') as f:
        expert_dataset = [json.loads(line) for line in f]
    target_vector = get_embedding(target_concept)
    
    results = {}
    verbose_logs = {}

    # --- Get configs for each agent ---
    sft_config = config['sft']
    grpo_base_config = config['grpo_base']
    grpo_const_config = {**grpo_base_config, **config['grpo_constant']}
    degrpo_config = {**grpo_base_config, **config['degrpo']}

    # --- 1. SFT ---
    start_time = time.time()
    sft_response, sft_policy = run_sft(expert_dataset, task_query, target_vector, **sft_config)
    results['sft'] = {'response': sft_response, 'time': time.time() - start_time}
    verbose_logs['sft'] = {'final_policy': sft_policy}
    print(f"\n--- SFT Final Response ---\n{sft_response}\n--------------------\n")

    # --- 2. Normal GRPO (Naive) ---
    start_time = time.time()
    grpo_normal_response, grpo_normal_history = run_grpo_normal(sft_policy, task_query, target_vector, **grpo_base_config)
    results['grpo_normal'] = {'response': grpo_normal_response, 'time': time.time() - start_time}
    verbose_logs['grpo_normal'] = grpo_normal_history
    print(f"\n--- Normal GRPO Final Response ---\n{grpo_normal_response}\n--------------------\n")

    # --- 3. GRPO w/ Constant Alpha ---
    start_time = time.time()
    grpo_const_response, grpo_const_history = run_grpo_constant(sft_policy, task_query, target_vector, **grpo_const_config)
    results['grpo_constant'] = {'response': grpo_const_response, 'time': time.time() - start_time}
    verbose_logs['grpo_constant'] = grpo_const_history
    print(f"\n--- Constant Alpha GRPO Final Response ---\n{grpo_const_response}\n--------------------\n")
    
    # --- 4. GRPO w/ CLIP Covariance ---
    start_time = time.time()
    grpo_clip_response, grpo_clip_history = run_grpo_clip(sft_policy, task_query, target_vector, **degrpo_config) # Uses same params as DEGRPO
    results['grpo_clip'] = {'response': grpo_clip_response, 'time': time.time() - start_time}
    verbose_logs['grpo_clip'] = grpo_clip_history
    print(f"\n--- CLIP-Cov GRPO Final Response ---\n{grpo_clip_response}\n--------------------\n")
    
    # --- 5. GRPO w/ Dynamic Entropy (DE-GRPO) ---
    start_time = time.time()
    degrpo_response, degrpo_history = run_degrpo(sft_policy, task_query, target_vector, **degrpo_config)
    results['degrpo'] = {'response': degrpo_response, 'time': time.time() - start_time}
    verbose_logs['degrpo'] = degrpo_history
    print(f"\n--- Dynamic Entropy GRPO Final Response ---\n{degrpo_response}\n--------------------\n")

    # --- Save Results ---
    output_dir = config.get("output_dir_base", "results")
    os.makedirs(output_dir, exist_ok=True)
    run_id = int(time.time())
    
    results_filename = os.path.join(output_dir, f"{task_name}_results_{run_id}.json")
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=4)
        
    verbose_filename = os.path.join(output_dir, f"{task_name}_verbose_log_{run_id}.json")
    with open(verbose_filename, 'w') as f:
        json.dump(verbose_logs, f, indent=4)
        
    print(f"\n{'='*20} EXPERIMENT FOR TASK '{task_name.upper()}' COMPLETE {'='*20}")
    print(f"Simple results saved to: {results_filename}")
    print(f"Verbose logs saved to: {verbose_filename}")
    
    print("\n--- Final Summary ---")
    print(f"{'Agent':<25} | {'Time (s)':<10} | {'Response (first 50 chars)':<50}")
    print("-" * 90)
    for agent, res in results.items():
        resp_preview = res['response'].replace('\n', ' ')[:50] + "..."
        print(f"{agent:<25} | {res['time']:<10.2f} | {resp_preview:<50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full ICPO experiment suite for a given task.")
    parser.add_argument('--task', type=str, required=True, help='The name of the task to run (e.g., "entropy").')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file.')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'dry-run'], help='Run mode: full or dry-run.')
    args = parser.parse_args()
    
    config = load_config(args.config, args.mode)

    tasks = {
        "entropy": {"query": "I don't get entropy. The 'messy room' analogy is confusing. Explain it to me in a new way.", "target": "A clear, novel, and insightful explanation of entropy for a beginner, focusing on the dispersal of energy or information."},
        "d-day": {"query": "Explain the significance of the Normandy Landings (D-Day) to a middle school student, focusing on the human element.", "target": "A compelling narrative about the courage, sacrifice, and strategic importance of D-Day, making it relatable to a young student."},
        "eulers_identity": {"query": "Give an intuitive explanation for why e^(i*pi) + 1 = 0, without just showing the derivation.", "target": "An intuitive, geometrical, or metaphorical explanation of Euler's Identity that builds understanding without relying on complex math."}
    }

    if args.task in tasks:
        task_info = tasks[args.task]
        run_experiment_for_task(
            task_name=args.task, 
            task_query=task_info["query"], 
            target_concept=task_info["target"],
            config=config
        )
    else:
        print(f"Error: Task '{args.task}' not found. Available tasks are: {list(tasks.keys())}")