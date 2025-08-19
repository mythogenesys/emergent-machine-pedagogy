import os
import json
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.common import get_embedding, cosine_similarity, generate_response # We need the generator for efficacy

def calculate_novelty(response_embedding, expert_embeddings):
    """Calculates novelty as 1 - max similarity to expert data."""
    if not expert_embeddings:
        return 1.0
    similarities = [cosine_similarity(response_embedding, exp_emb) for exp_emb in expert_embeddings]
    return 1 - max(similarities)

def calculate_efficacy(response_text, quiz_prompt, correct_answers):
    """Uses the base LLM as a 'student' to test the explanation's effectiveness."""
    student_prompt = f"Here is an explanation of a concept:\n\n---\n{response_text}\n---\n\n{quiz_prompt}"
    
    # Use the same reliable generator, but with low temp for factual quiz answers
    student_answer_raw = generate_response(student_prompt, temp=0.01, max_tokens=50)
    
    # Basic parsing of "True" or "False" from the student's answer
    student_answers = [ans.strip() for ans in student_answer_raw.split('\n')]
    
    score = 0
    for i, correct_ans in enumerate(correct_answers):
        if i < len(student_answers) and correct_ans.lower() in student_answers[i].lower():
            score += 1
            
    return score / len(correct_answers)

def main():
    print("--- Starting Post-Experiment Analysis ---")
    
    # Load Quiz Data
    with open('src/data/quiz_data.json', 'r') as f:
        quiz_data = json.load(f)

    # Load all expert datasets to calculate novelty
    expert_data_embeddings = {}
    for task_name in quiz_data.keys():
        path = f'src/data/{task_name}_expert_data.jsonl'
        with open(path, 'r') as f:
            expert_answers = [json.loads(line)['text'].split('[/INST]')[1].strip() for line in f]
            expert_data_embeddings[task_name] = [get_embedding(ans) for ans in expert_answers]

    # Find all result files in the 'results' directory
    result_files = glob.glob('results/*.json')
    if not result_files:
        print("No result files found in the 'results/' directory. Please run the experiments first.")
        return

    all_results = []
    
    # Use tqdm to show progress over the many result files
    for file_path in tqdm(result_files, desc="Analyzing Result Files"):
        # Extract task name and run id from filename, e.g., "entropy_results_167...json"
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        task_name = parts[0]
        run_id = parts[2].split('.')[0]
        
        with open(file_path, 'r') as f:
            data = json.load(f)

        for agent, metrics in data.items():
            response_text = metrics['response']
            response_embedding = get_embedding(response_text)
            
            # --- Calculate All Metrics ---
            novelty = calculate_novelty(response_embedding, expert_data_embeddings.get(task_name, []))
            efficacy = calculate_efficacy(
                response_text,
                quiz_data[task_name]['quiz_prompt'],
                quiz_data[task_name]['answers']
            )
            
            all_results.append({
                'task': task_name,
                'run_id': run_id,
                'agent': agent,
                'time': metrics['time'],
                'novelty': novelty,
                'efficacy': efficacy
            })

    # Convert to a pandas DataFrame for easy analysis
    df = pd.DataFrame(all_results)
    
    # --- Generate Summary Statistics (Mean and Std Dev) ---
    summary = df.groupby(['task', 'agent']).agg(
        mean_time=('time', 'mean'),
        std_time=('time', 'std'),
        mean_novelty=('novelty', 'mean'),
        std_novelty=('novelty', 'std'),
        mean_efficacy=('efficacy', 'mean'),
        std_efficacy=('efficacy', 'std'),
        runs=('run_id', 'count')
    ).round(3)

    # Save the final aggregated data to a CSV file
    summary_filename = 'final_summary_results.csv'
    summary.to_csv(summary_filename)

    print("\n--- Analysis Complete ---")
    print("\nAggregated summary statistics:")
    print(summary)
    print(f"\n✅ Final summary saved to '{summary_filename}'. This is the data for your paper's tables!")

if __name__ == "__main__":
    main()