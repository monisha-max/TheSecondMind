import openai
import os
from .memory_manager import MemoryManager

class SummarizationAgent:
    """
    Synthesizes the external research data and a domain summary into a final, cohesive output.
    Uses the ChatCompletion API with clear instructions.
    """
    def __init__(self, memory: MemoryManager, use_llm=True):
        self.memory = memory
        self.use_llm = use_llm
        openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

    def summarize(self, external_data: str, domain_summary: str) -> str:
        self.memory.log_event("[SummarizationAgent] Generating combined summary.")
        if self.use_llm:
            prompt = (
                "You are an expert research summarizer. Combine the following recent research findings and the domain summary "
                "to create a comprehensive final summary that highlights the latest developments and provides clear insights.\n\n"
                "Recent Research Findings:\n" + external_data + "\n\n" +
                "Domain Summary:\n" + domain_summary + "\n\n" +
                "Final Summary:"
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a concise and insightful research summarizer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                final_summary = response.choices[0].message["content"].strip()
            except Exception as e:
                self.memory.log_event(f"[SummarizationAgent] LLM error: {str(e)}")
                final_summary = f"Combined Summary:\nRecent Research: {external_data}\nDomain Summary: {domain_summary}"
        else:
            final_summary = f"Combined Summary:\nRecent Research: {external_data}\nDomain Summary: {domain_summary}"
        return final_summary
