import os
import openai
from .memory_manager import MemoryManager

class EvolutionAgent:
    """
    Refines a hypothesis by incorporating reflection notes and external data.
    Uses the OpenAI ChatCompletion API.
    """
    def __init__(self, memory: MemoryManager, use_llm=True):
        self.memory = memory
        self.use_llm = use_llm
        openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

    def evolve(self, hypothesis: str, reflection: str, external_data: str) -> str:
        self.memory.log_event("[EvolutionAgent] Evolving the hypothesis.")
        if self.use_llm:
            prompt = (
                "Refine the following hypothesis based on reflection and external data.\n\n"
                f"Hypothesis:\n{hypothesis}\n\n"
                f"Reflection:\n{reflection}\n\n"
                f"External Data:\n{external_data}\n\n"
                "Return an improved version of the hypothesis, highlighting how it addresses any weaknesses."
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                refined_hypothesis = response.choices[0].message["content"].strip()
            except Exception as e:
                self.memory.log_event(f"[EvolutionAgent] LLM call failed: {str(e)}")
                refined_hypothesis = f"Refined hypothesis based on reflection: {reflection}"
        else:
            refined_hypothesis = f"Refined hypothesis based on reflection: {reflection}"
        
        self.memory.store_data("refined_hypothesis", refined_hypothesis)
        return refined_hypothesis
