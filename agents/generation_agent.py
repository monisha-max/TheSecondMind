import openai
import os
from .memory_manager import MemoryManager

class GenerationAgent:
    """
    Creates an initial hypothesis using OpenAI's ChatCompletion API.
    Includes error handling and fallback logic.
    """
    def __init__(self, memory: MemoryManager, use_llm=True):
        self.memory = memory
        self.use_llm = use_llm
        openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

    def generate_hypothesis(self, query: str) -> str:
        self.memory.log_event(f"[GenerationAgent] Generating hypothesis for query: '{query}'")
        if self.use_llm:
            prompt = (
                f"You are a research assistant. Generate an initial hypothesis or research direction for the topic: {query}. "
                "Provide a concise statement and mention key challenges."
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful and creative research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.7
                )
                hypothesis = response.choices[0].message["content"].strip()
            except Exception as e:
                self.memory.log_event(f"[GenerationAgent] LLM call failed: {str(e)}")
                hypothesis = f"Initial hypothesis for '{query}': Explore recent challenges and possible solutions."
        else:
            hypothesis = f"Initial hypothesis for '{query}': Explore recent challenges and possible solutions."
        
        self.memory.store_data("current_hypothesis", hypothesis)
        return hypothesis
