# In src/common.py
import os # <-- Add this import
import numpy as np
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# --- Global Models (Load Once at the start of any script that imports this) ---
print("--- Loading Foundational Models ---")

# --- MODIFICATION ---
# Get the model path from an environment variable for container flexibility.
# Default to the local path if the variable is not set.
model_path = os.getenv("MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
print(f"Loading Llama.cpp model from: {model_path}")

LLM = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=0, # Crucial for CPU-only execution in the container
    verbose=False
)

# Load the sentence embedding model for calculating semantic similarity
print("Loading Sentence Transformer model for rewards...")
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Load the CLIP model for the cross-modal diversity metric
print("Loading CLIP model for cross-modal analysis...")
CLIP_MODEL = SentenceTransformer("sentence-transformers/clip-ViT-B-32")


print("--- ✅ All models loaded successfully ---\n")


# --- Core Utility Functions ---

def get_embedding(text: str) -> np.ndarray:
    """Generates a sentence embedding for a given text."""
    return EMBEDDING_MODEL.encode(text)

def generate_response(prompt: str, temp: float = 0.7, max_tokens: int = 256) -> str:
    """
    Generates a response using the stable llama-cpp-python library.
    This provides reliable performance and full control over temperature.
    """
    output = LLM(
        prompt,
        max_tokens=max_tokens,
        temperature=temp,
        stop=["</s>", "[INST]"] # Prevents the model from generating extra conversational turns
    )
    # The output is a dictionary; the generated text is nested inside
    return output['choices'][0]['text'].strip()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors, handling zero vectors gracefully."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # If either vector is a zero vector, their similarity is 0.
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    # Otherwise, calculate as normal.
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def format_prompt(query: str, policy_examples: list) -> str:
    """Formats a full prompt with a query and a list of few-shot examples."""
    if not policy_examples:
        return f"<s>[INST] {query} [/INST]"
        
    context = "\n".join([ex.get('text', '') for ex in policy_examples])
    return f"{context}\n<s>[INST] {query} [/INST]"