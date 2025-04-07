import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import streamlit as st
st.set_page_config(page_title="Multi-Agent Research Assistant", layout="wide")

import pandas as pd
import plotly.express as px
import openai
import json
from dotenv import load_dotenv
load_dotenv()
import os
os.environ['FAISS_NO_GPU_WARN'] = '1'

from agents.memory_manager import MemoryManager
from agents.generation_agent import GenerationAgent
from agents.api_selection_agent import APISelectionAgent
from agents.reflection_agent import ReflectionAgent
from agents.ranking_agent import RankingAgent
from agents.summarization_agent import SummarizationAgent
from agents.proximity_agent import ProximityAgent
from agents.evolution_agent import EvolutionAgent
from agents.meta_review_agent import MetaReviewAgent
from supervisor import Supervisor

# Helper function to evaluate agent outputs dynamically using the LLM via the Ranking Agent
def evaluate_agent_output(agent_name, output_text):
    """
    Evaluates an agent's output by asking the LLM to rate it on clarity, relevance, and feasibility.
    Returns a dictionary with keys 'clarity', 'relevance', and 'feasibility'.
    """
    prompt = (
        f"You are an objective evaluator. Please rate the following output from the {agent_name} on a scale of 1 to 10 "
        "for clarity, relevance, and feasibility. Return your answer in JSON format with keys 'clarity', 'relevance', "
        "and 'feasibility'.\n\n"
        f"Output:\n{output_text}\n\n"
        "Example response: {\"clarity\": 8, \"relevance\": 9, \"feasibility\": 7}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an objective evaluator."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.0
        )
        rating_json = response.choices[0].message["content"].strip()
        rating = json.loads(rating_json)
        return rating
    except Exception as e:
        # Fallback values in case of an error
        return {"clarity": 7, "relevance": 7, "feasibility": 7}

def main():
    st.title("Multi-Agent Research Assistant")
    
    topics = [
        "AGI",
        "Quantum Learning",
        "Machine Unlearning",
        "Sustainable software design",
        "Common sense reasoning in Agentic AI"
    ]
    
    query = st.text_input("Enter a research topic:", "")
    
    if st.button("Process Topic") and query.strip():

        with st.spinner("Running multi-agent pipeline..."):
            memory = MemoryManager()

            gen_agent = GenerationAgent(memory, use_llm=True)
            api_agent = APISelectionAgent(memory, max_retries=2)
            ref_agent = ReflectionAgent(memory, use_llm=True, max_tokens=150)
            rank_agent = RankingAgent(memory, use_llm=False)
            summ_agent = SummarizationAgent(memory, use_llm=True)
            prox_agent = ProximityAgent()
            evol_agent = EvolutionAgent(memory, use_llm=True)
            meta_agent = MetaReviewAgent(use_llm=True)

            supervisor = Supervisor(
                memory=memory,
                generation_agent=gen_agent,
                api_selection_agent=api_agent,
                reflection_agent=ref_agent,
                ranking_agent=rank_agent,
                summarization_agent=summ_agent,
                proximity_agent=prox_agent,
                evolution_agent=evol_agent,
                meta_review_agent=meta_agent,
                max_iterations=3,
                score_threshold=8.5
            )

            result = supervisor.handle_query(query)

        st.success("Processing complete!")
        
        # Organized output: two columns for agent outputs and visualizations
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Agent Outputs")
            # Table summarizing score evolution across iterations
            iteration_data = {
                "Iteration": list(range(1, len(result["score_history"]) + 1)),
                "Score": result["score_history"]
            }
            df_iterations = pd.DataFrame(iteration_data)
            st.dataframe(df_iterations, height=200)
            
            st.markdown("#### Final Hypothesis (Original)")
            st.write(result["hypothesis"])
            
            st.markdown("#### Refined Hypothesis (Evolution Agent)")
            st.write(result.get("refined_hypothesis", "No evolution output"))
            
            st.markdown("#### Reflection")
            st.write(result["reflection"])
            
            st.markdown("#### Combined Final Summary")
            st.write(result["final_summary"])
            
        with col_right:
            st.markdown("### Visualizations")
            # Line chart for score evolution
            ranking_data = pd.DataFrame({
                "Iteration": list(range(1, len(result["score_history"]) + 1)),
                "Score": result["score_history"]
            })
            fig_line = px.line(ranking_data, x="Iteration", y="Score", markers=True,
                               title="Score Evolution Over Iterations")
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Dynamic evaluation heatmap using the new helper function
            final_hypothesis_rating = evaluate_agent_output("Final Hypothesis", result["hypothesis"])
            refined_hypothesis_rating = evaluate_agent_output("Refined Hypothesis", result.get("refined_hypothesis", ""))
            reflection_rating = evaluate_agent_output("Reflection", result["reflection"])
            
            metrics = ["clarity", "relevance", "feasibility"]
            dynamic_data = {"Agent Output": ["Final Hypothesis", "Refined Hypothesis", "Reflection"]}
            for metric in metrics:
                dynamic_data[metric] = [
                    final_hypothesis_rating.get(metric, 0),
                    refined_hypothesis_rating.get(metric, 0),
                    reflection_rating.get(metric, 0)
                ]
            df_dynamic = pd.DataFrame(dynamic_data).set_index("Agent Output")
            
            st.markdown("#### Dynamic Agent Metrics Heatmap")
            fig_heatmap = px.imshow(df_dynamic, text_auto=True, aspect="auto",
                                    title="Dynamic Agent Metrics Heatmap")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("#### Recent Advancements")
            if "recent_advancements" in result and result["recent_advancements"]:
                for i, adv in enumerate(result["recent_advancements"], start=1):
                    st.markdown(f"**{i}. {adv.get('title', 'No Title')}**")
                    st.write(adv.get("summary", "No summary available."))
                    if adv.get("year"):
                        st.write(f"Year: {adv.get('year')}")
                    if adv.get("link"):
                        st.markdown(f"[Source Link]({adv.get('link')})")
                    if adv.get("references"):
                        st.write("References: " + ", ".join(adv.get("references")))
                    st.markdown("---")
            else:
                st.write("No recent advancements found.")
        
        st.markdown("---")
        st.markdown("### Meta-Review Feedback")
        st.write(result["meta_review"])

        st.markdown("---")
        st.markdown("#### Logs")
        for log in memory.get_logs():
            st.write(log)
        
        st.markdown("### Agent Interaction Diagram")
        st.markdown("""
```mermaid
graph LR
  U[User Query] --> G[Generation Agent]
  G --> A[APISelection Agent]
  A --> R[Reflection Agent]
  R --> P[Proximity Agent]
  R --> Ra[Ranking Agent]
  Ra --> E[Evolution Agent]
  E --> S[Summarization Agent]
  S --> M[MetaReview Agent]
  M --> OUT[Final Output]


""")
            
if __name__ == "__main__":
    main()