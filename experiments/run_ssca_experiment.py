import os
import json
import time
import argparse
import pandas as pd
from tqdm import tqdm
from src.common import get_embedding, cosine_similarity
from src.agent_teacher import TeacherAgent
from src.agent_student import StudentAgent

def run_ssca_session(task_name, task_query, target_concept, num_sessions=10):
    print(f"\n{'='*20} STARTING SSCA SESSION FOR TASK: {task_name.upper()} {'='*20}")

    # --- Setup ---
    # We provide the Teacher with an initial "textbook" (SFT policy) to start
    expert_data_path = f'src/data/{task_name}_expert_data.jsonl'
    with open(expert_data_path, 'r') as f:
        initial_policy = [json.loads(line) for line in f][:2] # Start with 2 expert examples

    teacher = TeacherAgent(initial_policy)
    student = StudentAgent()
    target_vector = get_embedding(target_concept)
    
    history = []

    for session in tqdm(range(num_sessions), desc=f"SSCA Teaching Sessions ({task_name})"):
        print(f"\n---------- Teaching Session {session + 1}/{num_sessions} ----------")
        
        # 1. Teacher teaches, which is an internal "study session"
        print("Teacher is preparing the lesson...")
        explanation, teacher_learning_history = teacher.teach_a_concept(
            student.state_vector, task_query, target_vector
        )
        print(f"Teacher Explains: {explanation[:120]}...")

        # 2. Student learns and critiques
        re_explanation, clarifying_question = student.learn_and_critique(explanation)
        print(f"Student Re-explains: {re_explanation[:120]}...")
        print(f"Student Asks: {clarifying_question[:120]}...")

        # 3. We measure the outcome of this session
        efficacy_score = cosine_similarity(get_embedding(re_explanation), target_vector)
        
        history.append({
            'session': session + 1,
            'efficacy_score': efficacy_score,
            'teacher_explanation': explanation,
            'student_re_explanation': re_explanation,
            'student_question': clarifying_question
        })
        print(f"Session Efficacy: {efficacy_score:.4f}")
    
    # --- Save Results ---
    output_dir = "results_ssca"
    os.makedirs(output_dir, exist_ok=True)
    run_id = int(time.time())
    output_filename = os.path.join(output_dir, f"{task_name}_ssca_{run_id}.csv")
    
    pd.DataFrame(history).to_csv(output_filename, index=False)
    print(f"\n✅ SSCA session complete. Full history saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Self-Structuring Cognitive Agent experiment.")
    parser.add_argument('--task', type=str, required=True, choices=['entropy', 'd-day', 'eulers_identity'])
    args = parser.parse_args()
    
    tasks = {
        "entropy": {"query": "I don't get entropy. The 'messy room' analogy is confusing. Explain it to me in a new way.", "target": "A clear, novel, and insightful explanation of entropy for a beginner, focusing on the dispersal of energy or information."},
        "d-day": {"query": "Explain the significance of the Normandy Landings (D-Day) to a middle school student, focusing on the human element.", "target": "A compelling narrative about the courage, sacrifice, and strategic importance of D-Day, making it relatable to a young student."},
        "eulers_identity": {"query": "Give an intuitive explanation for why e^(i*pi) + 1 = 0, without just showing the derivation.", "target": "An intuitive, geometrical, or metaphorical explanation of Euler's Identity that builds understanding without relying on complex math."}
    }

    run_ssca_session(
        task_name=args.task,
        task_query=tasks[args.task]["query"],
        target_concept=tasks[args.task]["target"]
    )