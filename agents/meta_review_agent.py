import openai
import os

class MetaReviewAgent:
    """
    Analyzes the system logs and provides feedback on potential bottlenecks and improvements.
    Uses the OpenAI ChatCompletion API.
    """
    def __init__(self, use_llm=True, model="gpt-3.5-turbo", temperature=0.5):
        self.use_llm = use_llm
        self.model = model
        self.temperature = temperature
        openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

    def review(self, logs):
        if self.use_llm:
            prompt = (
                "You are a performance evaluation expert. Analyze the following system logs and provide constructive "
                "feedback, identify bottlenecks, and suggest improvements:\n\n" 
                + "\n".join(logs)
            )
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a performance evaluation expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                return response['choices'][0]['message']['content'].strip()
            except Exception as e:
                return f"Meta-review error: {str(e)}"
        else:
            return "Meta-review disabled."
