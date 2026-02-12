"""
planner_agent.py
Breaks down complex queries into smaller subtasks before retrieval.

I added this because I noticed that when you search for something like
"explain methodology and limitations" as one query, the embedding ends up
somewhere in between and you get mediocre results for both topics. Splitting
into focused subtasks gives much better retrieval.

Also includes query complexity detection - we check if the query is simple
or complex before deciding whether to involve the planner at all.
"""

import re


# words that usually mean the query has multiple parts
COMPOUND_WORDS = [
    "and", "compare", "comparison", "difference", "differences",
    "versus", "vs", "contrast", "advantages and disadvantages",
]

# maps keywords to what section they probably want from the paper
SECTION_MAP = {
    "methodology": "Find info about the methodology or methods used.",
    "method": "Find info about the methodology or methods used.",
    "approach": "Find info about the approach taken.",
    "limitation": "Find info about the limitations.",
    "limitations": "Find info about the limitations.",
    "result": "Find info about the results.",
    "results": "Find info about the results.",
    "finding": "Find the key findings.",
    "findings": "Find the key findings.",
    "contribution": "Find the contributions of this work.",
    "contributions": "Find the contributions of this work.",
    "conclusion": "Find the conclusions.",
    "abstract": "Find the abstract or summary.",
    "introduction": "Find information from the introduction.",
    "related work": "Find info about related work.",
    "future work": "Find info about future work suggestions.",
    "evaluation": "Find info about how they evaluated their work.",
    "experiment": "Find info about the experiments.",
    "dataset": "Find info about the dataset used.",
}


def check_complexity(query):
    """
    Figures out if a query is simple or complex.
    
    Complex queries have words like "compare", "and", "difference",
    "advantages and disadvantages", "limitations" etc.
    Simple queries are straightforward single-topic questions.
    
    I added this because not every query needs the full planner treatment.
    For simple stuff like "what dataset did they use?", direct retrieval
    is faster and works just fine.
    """
    q = query.lower().strip()
    
    # check for phrases that indicate complexity
    complex_phrases = [
        "compare", "difference", "advantages and disadvantages",
        "limitations", "pros and cons", "contrast",
    ]
    
    for phrase in complex_phrases:
        if phrase in q:
            return {
                "is_complex": True,
                "reason": f"Found '{phrase}' - query likely needs multiple retrieval steps",
            }
    
    # check for "and" connecting different topics
    # but careful - "and" in "advantages and disadvantages" is already caught above
    if re.search(r'\b(and)\b', q):
        # make sure its actually connecting two different topics
        parts = re.split(r'\band\b', q)
        if len(parts) >= 2 and all(len(p.strip()) > 5 for p in parts):
            return {
                "is_complex": True,
                "reason": "Query uses 'and' to connect multiple topics",
            }
    
    return {
        "is_complex": False,
        "reason": "Simple single-topic query",
    }


def plan_query(query):
    """
    Takes a query and decides if/how to split it into subtasks.
    
    For compound queries (like "explain X and Y"), it creates separate
    subtasks plus a synthesis step. For simple queries it just passes
    the original query through as a single subtask.
    """
    q_lower = query.lower().strip()
    subtasks = []
    reasoning = ""

    # first check if we even need planning
    complexity = check_complexity(query)
    
    if not complexity["is_complex"]:
        # simple query - check if it targets a specific section
        matched = []
        seen_desc = set()
        for keyword, desc in SECTION_MAP.items():
            if keyword in q_lower and desc not in seen_desc:
                seen_desc.add(desc)
                matched.append(desc)
        
        if matched:
            for i, m in enumerate(matched):
                subtasks.append(f"Subtask {i+1}: {m}")
            reasoning = f"Simple query targeting {len(matched)} section(s)."
        else:
            subtasks = [f"Subtask 1: {query}"]
            reasoning = "Simple query, no decomposition needed."
        
        return {
            "original_query": query,
            "complexity": complexity,
            "is_compound": len(subtasks) > 1,
            "subtasks": subtasks,
            "reasoning": reasoning,
        }

    # complex query - try to split it
    is_compound = False
    for word in COMPOUND_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, q_lower):
            is_compound = True
            break

    if is_compound:
        # split on connecting words
        parts = re.split(
            r'\b(?:and|vs\.?|versus|compared?\s+to|contrast\s+with)\b',
            q_lower
        )
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) > 1:
            for i, part in enumerate(parts):
                subtasks.append(f"Subtask {i+1}: {part.capitalize()}")
            subtasks.append(f"Subtask {len(parts)+1}: Combine and summarize findings.")
            reasoning = f"Split into {len(parts)} parts + synthesis step."
        else:
            subtasks = [f"Subtask 1: {query}"]
            reasoning = "Detected compound keyword but couldn't split meaningfully."
    else:
        # complex but not compound - just use original
        subtasks = [f"Subtask 1: {query}"]
        reasoning = "Marked complex but single-topic."

    return {
        "original_query": query,
        "complexity": complexity,
        "is_compound": is_compound or len(subtasks) > 1,
        "subtasks": subtasks,
        "reasoning": reasoning,
    }


def show_plan(plan):
    """Print the plan so we can see what the planner decided."""
    print(f"\n{'='*50}")
    print("PLANNER OUTPUT")
    print(f"{'='*50}")
    print(f"Query: {plan['original_query']}")
    print(f"Complex: {plan['complexity']['is_complex']} ({plan['complexity']['reason']})")
    print(f"Compound: {plan['is_compound']}")
    print(f"Reasoning: {plan['reasoning']}")
    print(f"Subtasks:")
    for t in plan['subtasks']:
        print(f"  - {t}")


if __name__ == "__main__":
    queries = [
        "Explain the methodology and limitations.",
        "What is the main contribution?",
        "Compare the results with baseline methods and discuss future work.",
        "What dataset was used?",
        "What are the advantages and disadvantages of their approach?",
    ]

    for q in queries:
        plan = plan_query(q)
        show_plan(plan)
        print()
