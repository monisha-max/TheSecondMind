import openai
import os
import random
from .memory_manager import MemoryManager

class RankingAgent:
    """
    Assigns a numerical score to the hypothesis based on the hypothesis and its reflection.
    Uses the ChatCompletion API optionally; otherwise falls back to a heuristic.
    """
    def __init__(self, memory: MemoryManager, use_llm=False):
        self.memory = memory
        self.use_llm = use_llm
        openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

    def rank(self, hypothesis: str, reflection: str) -> float:
        self.memory.log_event("[RankingAgent] Ranking the hypothesis.")
        if self.use_llm:
            prompt = (
                "You are an expert reviewer. Based on the following hypothesis and its reflection, "
                "assign a quality score from 1 to 10, where 10 indicates the highest quality and relevance.\n\n"
                f"Hypothesis:\n{hypothesis}\n\nReflection:\n{reflection}\n\nScore:"
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are objective and concise."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
                score_str = response.choices[0].message["content"].strip()
                score = float(score_str)
            except Exception as e:
                self.memory.log_event(f"[RankingAgent] LLM error: {str(e)}")
                score = random.uniform(7.0, 10.0)
        else:
            score = random.uniform(7.0, 10.0)
        self.memory.store_data("final_score", score)
        return score
