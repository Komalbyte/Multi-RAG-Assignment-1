"""
revision_agent.py
Takes the critic's feedback and tries to make the answer better.

The loop goes: answer -> critique -> revise -> re-critique
We cap it at 2 rounds to avoid wasting time since small models
don't always improve much with multiple passes anyway.

Honestly with flan-t5-base the revisions are sometimes marginal.
A bigger model would make this loop way more useful. But the
architecture is there and it does work.
"""

from src.answer_agent import get_llm, LLM_NAME

MAX_ROUNDS = 2


def revise(original_ans, feedback, context, question, model_name=LLM_NAME):
    """
    Generate a revised answer based on the critic's feedback.
    Still grounded in the same context - we don't retrieve new stuff here.
    """
    llm = get_llm(model_name)

    prompt = (
        "Improve this answer based on the feedback below. "
        "Stay grounded in the context, don't make stuff up.\n\n"
        f"Question: {question}\n\n"
        f"Context: {context[:500]}\n\n"
        f"Original answer: {original_ans}\n\n"
        f"Feedback: {feedback}\n\n"
        "Improved answer:"
    )

    out = llm(prompt)
    new_ans = out[0]["generated_text"].strip()

    return {
        "revised": new_ans,
        "original": original_ans,
        "feedback_used": feedback,
    }


def run_revision_loop(answer, context, question, eval_fn, model_name=LLM_NAME,
                      max_rounds=MAX_ROUNDS):
    """
    The full revision loop. Keeps trying until score >= 7 or we hit max rounds.
    
    eval_fn should be the critic's evaluate() function.
    """
    history = []
    current = answer

    # first evaluation
    ev = eval_fn(current, context, question, model_name)
    history.append({
        "round": 0,
        "answer": current,
        "score": ev["score"],
        "feedback": ev["feedback"],
    })

    rounds_done = 0
    while ev["needs_revision"] and rounds_done < max_rounds:
        rounds_done += 1

        # try to fix it
        rev = revise(current, ev["feedback"], context, question, model_name)
        current = rev["revised"]

        # check if it got better
        ev = eval_fn(current, context, question, model_name)
        history.append({
            "round": rounds_done,
            "answer": current,
            "score": ev["score"],
            "feedback": ev["feedback"],
        })

        print(f"  Revision round {rounds_done}: score = {ev['score']}/10")

    first_score = history[0]["score"]
    last_score = history[-1]["score"]

    return {
        "final_answer": current,
        "rounds": rounds_done,
        "history": history,
        "got_better": last_score > first_score,
        "score_before": first_score,
        "score_after": last_score,
    }


def show_revision(result):
    print(f"\n{'='*50}")
    print("REVISION AGENT")
    print(f"{'='*50}")
    print(f"Rounds: {result['rounds']}")
    print(f"Score: {result['score_before']}/10 -> {result['score_after']}/10")
    print(f"Improved: {result['got_better']}")
    print(f"\nFinal answer:\n{result['final_answer']}")


if __name__ == "__main__":
    from src.critic_agent import evaluate

    ctx = "The methodology uses transformers with attention mechanisms."
    q = "What is the methodology?"
    ans = "It uses something."  # intentionally bad

    result = run_revision_loop(ans, ctx, q, evaluate)
    show_revision(result)
