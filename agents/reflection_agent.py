import openai
import os
from .memory_manager import MemoryManager

class ReflectionAgent:
    """
    Evaluates and refines the hypothesis by integrating external data.
    Uses the OpenAI ChatCompletion API with adjustable max_tokens for performance.
    """
    def __init__(self, memory: MemoryManager, use_llm=True, max_tokens=150):
        self.memory = memory
        self.use_llm = use_llm
        self.max_tokens = max_tokens
        openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

    def reflect(self, hypothesis: str, external_data: str) -> str:
        self.memory.log_event("[ReflectionAgent] Reflecting on the hypothesis.")
        if self.use_llm:
            prompt = (
                "You are a research reviewer tasked with improving a research hypothesis. "
                "Evaluate the following hypothesis and external data. Identify strengths, weaknesses, and suggest improvements.\n\n"
                f"Hypothesis:\n{hypothesis}\n\n"
                f"External Data:\n{external_data}\n\n"
                "Provide a refined analysis in 2-3 sentences."
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a critical, analytical research reviewer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.5
                )
                reflection_comment = response.choices[0].message["content"].strip()
            except Exception as e:
                self.memory.log_event(f"[ReflectionAgent] LLM call error: {str(e)}")
                reflection_comment = "Reflection: The hypothesis appears reasonable but may need further details."
        else:
            reflection_comment = (
                f"Hypothesis: {hypothesis}\n"
                f"External Data: {external_data}\n"
                "Reflection: The approach shows promise; consider exploring additional research angles."
            )
        self.memory.store_data("reflection_notes", reflection_comment)
        return reflection_comment
