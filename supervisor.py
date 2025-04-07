import time
import concurrent.futures
import re
from agents.memory_manager import MemoryManager
from agents.generation_agent import GenerationAgent
from agents.api_selection_agent import APISelectionAgent
from agents.reflection_agent import ReflectionAgent
from agents.ranking_agent import RankingAgent
from agents.summarization_agent import SummarizationAgent
from agents.proximity_agent import ProximityAgent
from agents.evolution_agent import EvolutionAgent
from agents.meta_review_agent import MetaReviewAgent

class Supervisor:
    def __init__(self,
                 memory: MemoryManager,
                 generation_agent: GenerationAgent,
                 api_selection_agent: APISelectionAgent,
                 reflection_agent: ReflectionAgent,
                 ranking_agent: RankingAgent,
                 summarization_agent: SummarizationAgent,
                 proximity_agent: ProximityAgent,
                 evolution_agent: EvolutionAgent,
                 meta_review_agent: MetaReviewAgent,
                 max_iterations=3,
                 score_threshold=8.5):
        self.memory = memory
        self.generation_agent = generation_agent
        self.api_selection_agent = api_selection_agent
        self.reflection_agent = reflection_agent
        self.ranking_agent = ranking_agent
        self.summarization_agent = summarization_agent
        self.proximity_agent = proximity_agent
        self.evolution_agent = evolution_agent
        self.meta_review_agent = meta_review_agent

        self.max_iterations = max_iterations
        self.score_threshold = score_threshold

        self._init_proximity_memories()

    def _init_proximity_memories(self):
        self.proximity_agent.add_memory("case1", "The solar panel project was highly successful in urban environments.")
        self.proximity_agent.add_memory("case2", "A previous experiment with wind turbines in rural areas did not meet expectations.")
        self.proximity_agent.add_memory("case3", "A case study on integrating solar technology into building facades in metropolitan areas.")

    def handle_query(self, query: str):
        self.memory.log_event(f"[Supervisor] Handling query: {query}")
        hypothesis = self.generation_agent.generate_hypothesis(query)

        best_result = None
        best_score = -1
        score_history = []

        domain_summaries = {
            "AGI": "AGI aims to create machines that can perform any intellectual task a human can.",
            "Quantum Learning": "Quantum Learning applies quantum principles to improve ML algorithms.",
            "Machine Unlearning": "Focuses on effectively removing data influences from models for privacy or adaptability.",
            "Sustainable software design": "Emphasizes maintainability, energy efficiency, and minimal environmental impact.",
            "Common sense reasoning in Agentic AI": "Focuses on integrating human-like reasoning and context awareness into AI."
        }
        domain_summary = domain_summaries.get(query, "General domain summary for research.")

        for iteration in range(self.max_iterations):
            self.memory.log_event(f"[Supervisor] Iteration {iteration+1}/{self.max_iterations}")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_data = executor.submit(self.api_selection_agent.fetch_external_data, query)
                future_prox = executor.submit(self.proximity_agent.retrieve_similar_cases, hypothesis, 2)
                external_data = future_data.result()
                similar_cases = future_prox.result()

            if similar_cases:
                self.memory.log_event(f"[Supervisor] Similar past cases: {similar_cases}")
            else:
                self.memory.log_event("[Supervisor] No similar past cases found.")

            if isinstance(external_data, dict):
                summary = external_data.get("summary", "")
                advancements = external_data.get("recent_advancements", [])
            else:
                summary = external_data
                advancements = []

            # Updated sorting logic for advancements
            def parse_date(paper):
                if "parsed_date" in paper:
                    return paper["parsed_date"]
                date_str = paper.get("published", "")
                try:
                    return time.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    try:
                        return time.strptime(date_str, "%Y-%m-%d")
                    except Exception:
                        return time.gmtime(0)
            advancements = sorted(advancements, key=lambda x: parse_date(x), reverse=True)[:3]

            advancement_text = ""
            for i, item in enumerate(advancements):
                advancement_text += f"\nAdvancement {i+1}:\n"
                advancement_text += f"Title: {item.get('title')}\n"
                authors = item.get("authors", [])
                advancement_text += f"Authors: {', '.join(authors) if authors else 'N/A'}\n"
                advancement_text += f"Published: {item.get('published')}\n"
                advancement_text += f"Summary: {item.get('summary')}\n"
                link = item.get("link", "N/A")
                advancement_text += f"Source Link: {link}\n"

            full_external_knowledge = f"{summary}\n\nRecent Advancements:\n{advancement_text.strip()}"

            reflection = self.reflection_agent.reflect(hypothesis, full_external_knowledge)
            score = self.ranking_agent.rank(hypothesis, reflection)
            score_history.append(score)

            refined_hypothesis = self.evolution_agent.evolve(hypothesis, reflection, full_external_knowledge)

            current_result = {
                "hypothesis": hypothesis,
                "reflection": reflection,
                "score": score,
                "refined_hypothesis": refined_hypothesis,
                "score_history": score_history,
                "recent_advancements": advancements
            }

            # Update best_result if the current score is higher
            if score > best_score:
                best_score = score
                best_result = current_result

            if score < self.score_threshold and iteration < self.max_iterations - 1:
                self.memory.log_event("[Supervisor] Score below threshold, using refined hypothesis next iteration.")
                hypothesis = refined_hypothesis
            else:
                self.memory.log_event("[Supervisor] Score threshold met or final iteration.")
                # Continue iterating even if current iteration is above threshold,
                # because a future iteration might still produce a better score.

        # Use the best_result (highest score) for final output
        best_result["final_summary"] = self.summarization_agent.summarize(full_external_knowledge, domain_summary)
        self.memory.log_event(f"[Supervisor] Final result (best iteration): {best_result}")

        meta_review_output = self.meta_review_agent.review(self.memory.get_logs())
        best_result["meta_review"] = meta_review_output
        self.memory.log_event(f"[Supervisor] MetaReviewAgent output: {meta_review_output}")

        self._apply_meta_review_suggestions(meta_review_output)

        return best_result


    def _apply_meta_review_suggestions(self, meta_review_text: str):
        if "reflection is slow" in meta_review_text.lower() or "reflection took too long" in meta_review_text.lower():
            self.memory.log_event("[Supervisor] Meta-review suggests Reflection is slow. Lowering reflection tokens.")
            if hasattr(self.reflection_agent, "max_tokens"):
                new_tokens = max(50, self.reflection_agent.max_tokens - 50)
                self.reflection_agent.max_tokens = new_tokens
                self.memory.log_event(f"[Supervisor] ReflectionAgent max_tokens set to {new_tokens}")

        if "threshold too high" in meta_review_text.lower():
            old_threshold = self.score_threshold
            self.score_threshold = max(5.0, old_threshold - 1.0)
            self.memory.log_event(f"[Supervisor] score_threshold changed from {old_threshold} to {self.score_threshold}")

        if "increase iterations" in meta_review_text.lower() or "use more iterations" in meta_review_text.lower():
            old_iterations = self.max_iterations
            self.max_iterations += 1
            self.memory.log_event(f"[Supervisor] max_iterations changed from {old_iterations} to {self.max_iterations}")
