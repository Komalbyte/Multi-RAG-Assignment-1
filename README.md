# Self-Reflective Multi-Agent RAG System

## What This Project Does

This is a question-answering system for research papers that goes beyond the typical RAG setup. Most RAG systems follow a simple pattern — embed a query, pull some chunks from a vector database, feed them to an LLM, and return whatever it generates. That works for basic questions, but it breaks down with anything more complex.

The main idea here is to add a few extra layers of "thinking" to the pipeline. Instead of one shot at an answer, the system plans the retrieval, generates an answer, then actually checks if the answer is any good. If its not, it tries again. The whole thing is built as separate agents that each handle one job.

## Why Not Just Use Basic RAG?

I started with a basic RAG setup initially, and it worked okayish for simple questions like "what dataset did they use?". But the moment you ask something like "explain the methodology and discuss its limitations", things fall apart. The embedding for that query ends up somewhere in between methodology and limitations in vector space, so the retrieved chunks are mediocre for both.

The other problem is there's no quality control. If the LLM hallucinates something or gives a half-baked answer, theres nothing in the pipeline to catch that. You just get whatever it generates.

### Basic RAG Flow

```
Question -> Embed -> Search -> Generate -> Done
```

### What This System Does

```
Question -> Check Complexity -> Plan Subtasks -> Retrieve (per subtask) -> 
Answer -> Critique -> Revise (if needed) -> Done
```

The second approach is obviously slower (more LLM calls), but the answer quality is noticeably better, especially for multi-part questions. For an assignment or research context where accuracy matters more than speed, thats a fair tradeoff.

## Personal Note

While building this system, I noticed that the basic retrieve-and-generate approach kept giving incomplete answers whenever the question touched multiple topics. For example, when I asked about methodology AND limitations, it would usually cover one and barely mention the other. Thats why I added the planning step — it breaks down these compound questions so each part gets its own focused retrieval. It made a clear difference in the quality of answers, even with a small model like flan-t5-base.

## System Architecture

```
User Question
    |
    v
+------------------+
|   Complexity      |  <-- checks if query is simple or complex
|   Detector        |
+--------+---------+
         |
         v
+------------------+
|  Planner Agent   |  <-- splits complex queries into subtasks
+--------+---------+
         |
         v (for each subtask)
+------------------+
| Retrieval Agent  |  <-- FAISS search, returns top-k chunks
+--------+---------+
         |
         v
+------------------+
|  Answer Agent    |  <-- generates answer from context (flan-t5-base)
+--------+---------+
         |
         v
+------------------+
|  Critic Agent    |  <-- heuristic + LLM evaluation, scores 1-10
+--------+---------+
         |
    score >= 7? --> output
         |
    score < 7
         v
+------------------+
| Revision Agent   |  <-- refines answer, max 2 rounds
+--------+---------+
         |
         v
    Final Answer

+------------------+
|  Memory Module   |  <-- stores everything for follow-up queries
+------------------+
```

## Query Complexity Detection

I added this feature because I noticed not every query needs the full planner treatment. If someone asks "what dataset did they use?", thats straightforward — just retrieve and answer. But "compare the methodology with related work and discuss limitations" clearly needs to be broken down.

The complexity detector checks for specific patterns:

- Words like **"compare"**, **"difference"**, **"advantages and disadvantages"**, **"limitations"**  mark a query as complex
- The word **"and"** connecting two different topics (not just "advantages and disadvantages" which is one concept) also triggers complexity
- Everything else is treated as simple

When a query is complex, the planner agent kicks in and splits it into subtasks. When its simple, we go straight to retrieval. This saves time on easy questions and gives better handling for hard ones.

## Components

| File | What it does |
|------|-------------|
| `pdf_loader.py` | Extracts text from PDFs using pypdf |
| `chunking.py` | Splits text into overlapping chunks (600 chars, 100 overlap) |
| `embeddings.py` | Creates 384-dim vectors with all-MiniLM-L6-v2, builds FAISS index |
| `retrieval.py` | Searches the index for relevant chunks |
| `planner_agent.py` | Detects complexity and splits compound queries into subtasks |
| `answer_agent.py` | Generates answers using flan-t5-base |
| `critic_agent.py` | Evaluates answers using heuristics + LLM feedback |
| `revision_agent.py` | Improves answers based on critic feedback (max 2 rounds) |
| `memory.py` | Stores session history for follow-up questions |
| `main.py` | CLI pipeline that orchestrates everything |
| `app.py` | Streamlit web interface |

## Setup

You need Python 3.9+ and pip.

```bash
# create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# install everything
pip install -r requirements.txt
```

First run will download models (~500MB total). After that they're cached locally.

**No API keys needed.** Everything runs locally on your CPU.

### Models Used

- **Embeddings:** all-MiniLM-L6-v2 (~80MB) — produces 384-dim vectors, trained on 1B sentence pairs
- **LLM:** google/flan-t5-base (~900MB) — 250M param seq2seq model, handles Q&A well enough for this scale

## How to Run

### Command Line
```bash
# put your PDF in the data folder
cp your_paper.pdf data/research_paper.pdf

# run it
python -m src.main

# or specify a path directly
python -m src.main path/to/some_paper.pdf
```

It runs 3 demo queries first, then drops into interactive mode where you can type your own questions.

### Streamlit Web UI
```bash
streamlit run src/app.py
```
Upload a PDF through the sidebar, type questions in the main area. Each agent's output is shown in expandable sections.

### Jupyter Notebook
```bash
jupyter notebook notebook/Agentic_RAG.ipynb
```
Step-by-step walkthrough of every module with explanations.

## Example Queries

These are the three I used for testing:

1. "What is the problem statement of this paper?"
2. "Explain the methodology and limitations."
3. "What are the key contributions compared to existing work?"

Query 2 tests the planner (splits into methodology + limitations + synthesis). Query 3 tests retrieval across different sections.

## Sample Output

```
QUERY: Explain the methodology and limitations.

[1] PLANNER
  Complexity: Complex - Found 'limitations' - query likely needs multiple retrieval steps
  Split into 2 parts + synthesis step.
  Subtasks:
    - Subtask 1: Explain the methodology
    - Subtask 2: Limitations
    - Subtask 3: Combine and summarize findings.

[2] RETRIEVAL
  Searching: Explain the methodology
    Rank 1 (dist: 0.4523)
    Rank 2 (dist: 0.5891)
    ...

[3] ANSWER AGENT
  Initial answer: The methodology involves...

[4] CRITIC
  Score: 6/10
  Needs revision: True
    - length: Length seems fine
    - grounding: Good grounding (72%)
    - llm says: Could include more detail...

[5] REVISION
  Round 1: score = 7/10
  Final answer: The paper uses a methodology based on...
```

## Known Limitations

These are things I'm aware of but couldn't fully solve within the scope of this project:

1. **Small model.** flan-t5-base is only 250M parameters. It generates short answers and sometimes misses nuance. A bigger model would help a lot but needs GPU.

2. **Hallucination.** Even with grounding instructions in the prompt, the model sometimes makes stuff up. The critic catches some of this but not everything.

3. **Simple planner.** The query decomposition uses keyword matching which works for typical academic questions but would miss unusual phrasing. An LLM-based planner would be more flexible.

4. **FAISS is brute force.** IndexFlatL2 does O(n×d) search per query. Fine for a single paper (maybe a few hundred chunks), but wouldn't scale to thousands of documents without switching to approximate methods.

5. **PDF parsing.** pypdf handles normal text PDFs well but struggles with scanned documents, tables, and equations. Multi-column layouts also cause issues.

6. **Context window.** flan-t5-base only handles about 512 tokens of input. We have to truncate the retrieved context which means some information gets lost.

7. **Critic accuracy.** The heuristic checks work okay but they're approximate. The LLM-based evaluation from a small model is hit-or-miss.

## What I'd Do Differently With More Time

- Use a larger model (flan-t5-large or xl) if I had access to a GPU
- Replace keyword-based planning with actual LLM-based query analysis
- Add automatic query reformulation when retrieval scores are low
- Support multiple PDFs at once
- Store memory in a database instead of just in-memory
- Try using NLI models for better grounding evaluation
- Add OCR for scanned PDFs

## Project Structure

```
RAG_Assignment/
├── data/
│   └── research_paper.pdf
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── planner_agent.py
│   ├── answer_agent.py
│   ├── critic_agent.py
│   ├── revision_agent.py
│   ├── memory.py
│   ├── main.py
│   └── app.py
├── notebook/
│   └── Agentic_RAG.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Technical Notes

**Chunking:** 600 characters per chunk with 100 character overlap. The overlap is there because without it, you lose context at chunk boundaries. I tested both ways and retrieval was worse without overlap.

**FAISS:** Using IndexFlatL2 which does exact nearest-neighbor search. Memory usage is roughly n × 384 × 4 bytes. For a 20-page paper with ~100 chunks thats about 150KB — nothing.

**Embeddings:** 384 dimensions from all-MiniLM-L6-v2. These are dense vectors tuned for semantic similarity, so "methodology" and "approach" end up close together in vector space even though they're different words.

**Critic scoring:** The score is mostly from heuristic checks (length, word overlap with context, query coverage). The LLM feedback is supplementary — I found that flan-t5-base isn't reliable enough on its own for scoring, so the heuristics act as a safety net.
