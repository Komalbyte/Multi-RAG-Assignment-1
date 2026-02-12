"""
critic_agent.py
Evaluates the answer the LLM generated to see if its actually good.

This is what makes the system "self-reflective" - it doesn't just generate
an answer and call it done. It checks whether the answer is:
  - grounded in the actual retrieved context
  - complete enough to answer the question
  - not hallucinating stuff

I use a mix of heuristic checks (rule-based) and LLM evaluation.
The heuristics are honestly more reliable than asking flan-t5 to judge
its own output - small models aren't great at self-evaluation.
"""

from src.answer_agent import get_llm, LLM_NAME


def heuristic_checks(answer, context, question):
    """
    Rule-based evaluation that catches common problems.
    More reliable than asking a small LLM to score itself.
    """
    notes = {}
    score = 10  # start perfect, deduct for issues

    # is the answer suspiciously short?
    if len(answer) < 20:
        notes["length"] = "Very short answer (< 20 chars), probably incomplete"
        score -= 3
    elif len(answer) < 50:
        notes["length"] = "Kinda short, might be missing detail"
        score -= 1
    else:
        notes["length"] = "Length seems fine"

    # does it say "not found" or similar?
    no_info = ["does not contain", "not found", "no information",
               "cannot answer", "not mentioned", "not available"]
    if any(p in answer.lower() for p in no_info):
        notes["no_info"] = "Answer says info is missing - might need different retrieval"
        score -= 2
    else:
        notes["no_info"] = "Doesn't flag missing info"

    # grounding check - are the words in the answer actually in the context?
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "it", "this", "that", "with", "by"}
    
    ans_words = set(answer.lower().split()) - stop
    ctx_words = set(context.lower().split()) - stop
    
    if len(ans_words) > 0:
        overlap = len(ans_words & ctx_words) / len(ans_words)
    else:
        overlap = 0

    if overlap < 0.3:
        notes["grounding"] = f"Low grounding ({overlap:.0%}) - possible hallucination"
        score -= 3
    elif overlap < 0.5:
        notes["grounding"] = f"Moderate grounding ({overlap:.0%})"
        score -= 1
    else:
        notes["grounding"] = f"Good grounding ({overlap:.0%})"

    # does the answer actually address the question?
    q_words = set(question.lower().split()) - stop
    q_overlap = len(q_words & set(answer.lower().split())) / max(len(q_words), 1)
    if q_overlap < 0.3:
        notes["relevance"] = f"Low relevance ({q_overlap:.0%}) - might not be answering the right question"
        score -= 2
    else:
        notes["relevance"] = f"Relevance okay ({q_overlap:.0%})"

    score = max(1, min(10, score))
    return {"notes": notes, "score": score}


def llm_eval(answer, context, question, model_name=LLM_NAME):
    """
    Ask the LLM to evaluate the answer.
    Take this with a grain of salt - small models aren't great at this.
    """
    llm = get_llm(model_name)

    prompt = (
        "Evaluate this answer for completeness and accuracy. "
        "Point out anything missing or wrong.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context[:500]}\n\n"
        f"Answer: {answer}\n\n"
        "Evaluation:"
    )

    out = llm(prompt)
    return {"llm_feedback": out[0]["generated_text"].strip()}


def evaluate(answer, context, question, model_name=LLM_NAME):
    """
    Full evaluation combining heuristics and LLM feedback.
    Returns a score (1-10) and whether the answer needs revision.
    """
    # heuristics are the main signal
    h = heuristic_checks(answer, context, question)
    
    # llm adds some qualitative feedback
    l = llm_eval(answer, context, question, model_name)

    # build a readable summary
    feedback_lines = []
    for name, note in h["notes"].items():
        feedback_lines.append(f"  - {name}: {note}")
    feedback_lines.append(f"  - llm says: {l['llm_feedback']}")
    
    summary = "\n".join(feedback_lines)

    return {
        "score": h["score"],
        "feedback": summary,
        "notes": h["notes"],
        "llm_feedback": l["llm_feedback"],
        "needs_revision": h["score"] < 7,
    }


def show_eval(ev):
    print(f"\n{'='*50}")
    print("CRITIC AGENT")
    print(f"{'='*50}")
    print(f"Score: {ev['score']}/10")
    print(f"Needs revision: {ev['needs_revision']}")
    print(f"\nFeedback:\n{ev['feedback']}")


if __name__ == "__main__":
    test_ans = "The methodology uses transfer learning with a CNN."
    test_ctx = "The study employs transfer learning using a convolutional neural network."
    test_q = "What methodology was used?"

    result = evaluate(test_ans, test_ctx, test_q)
    show_eval(result)
