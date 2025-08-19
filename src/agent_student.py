import numpy as np
from src.common import get_embedding, generate_response

class StudentAgent:
    def __init__(self, embed_dim=384):
        # The student's understanding starts at zero
        self.state_vector = np.zeros(embed_dim)
        self.learning_rate = 0.5

    def update_state(self, explanation_text):
        """ The student 'learns' by moving their state vector towards the explanation. """
        explanation_embedding = get_embedding(explanation_text)
        self.state_vector += self.learning_rate * (explanation_embedding - self.state_vector)

    def learn_and_critique(self, explanation_text):
        """ The student processes the explanation and provides feedback. """
        # 1. The student learns from the explanation
        self.update_state(explanation_text)

        # 2. The student tries to re-explain the concept in simple terms
        re_explanation_prompt = f"Based ONLY on the following text, re-explain the core concept in one simple sentence.\n\nText: \"{explanation_text}\""
        re_explanation = generate_response(re_explanation_prompt, temp=0.2, max_tokens=64)

        # 3. The student identifies a point of confusion
        confusion_prompt = f"After reading this explanation: \"{explanation_text}\"\n\nWhat is the single most important and insightful clarifying question you still have?"
        clarifying_question = generate_response(confusion_prompt, temp=0.7, max_tokens=64)
        
        return re_explanation, clarifying_question