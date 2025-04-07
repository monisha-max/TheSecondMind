# TheSecondMind

# Key Features

Modular Multi-Agent Architecture:
Distinct agents for hypothesis generation, external data integration, reflection, ranking, evolution, summarisation, semantic memory (using FAISS), and meta-review.


Real-Time Data Integration:
Uses live APIs (Google Custom Search and NASA) to fetch up-to-date research information.


Dynamic Adaptation & Self-Improvement:
The Supervisor learns from meta-review feedback, automatically adjusting parameters (e.g., reflection token limits, score thresholds, and iteration counts) for improved performance over time.


Iterative Refinement Process:
The system iteratively refines hypotheses using the Evolution Agent when initial outputs do not meet quality thresholds.


Parallel Processing & Caching:
External data fetching and similarity retrieval are parallelised to reduce overall processing time.
API responses are cached to avoid redundant calls, boosting efficiency.


Semantic Memory with FAISS:
Incorporates an FAISS-powered Proximity Agent to retrieve similar past cases for enhanced contextual insight.


Comprehensive Summarization:
Synthesises recent research findings with domain-specific summaries to provide a coherent final output.


Transparent Performance Metrics:
Detailed logs and score evolution graphs provide clear insights into system performance and improvements over iterations.


User-Friendly, Interactive Interface:
Built with Streamlit, the interface allows users to select topics, view detailed outputs, and explore a visual diagram of agent interactions.


Scalable & Extensible Design:
Each component is modular, facilitating easy maintenance and future enhancements.
