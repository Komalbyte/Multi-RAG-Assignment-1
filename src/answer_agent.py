"""
answer_agent.py
Takes the retrieved context and generates an answer using a local LLM.

Using google/flan-t5-base here because:
  - its free (no API key)
  - runs on CPU without issues
  - decent at Q&A tasks
  - only ~250M params so its not too slow

The downside is the small context window (~512 tokens). If the retrieved
context is too long, we truncate it. A bigger model would obviously do
better but this works for a demo/assignment.
"""

from transformers import pipeline

# keep model in memory so we dont reload it each time
_llm = None
LLM_NAME = "google/flan-t5-base"
MAX_OUT_TOKENS = 256


def get_llm(model_name=LLM_NAME):
    """Load the language model. First call downloads it, then its cached."""
    global _llm
    if _llm is not None:
        return _llm

    print(f"Loading LLM: {model_name}")
    _llm = pipeline(
        "text2text-generation",
        model=model_name,
        max_length=MAX_OUT_TOKENS,
        device=-1,  # -1 = CPU
    )
    print("LLM loaded.")
    return _llm


def make_prompt(question, context):
    """
    Build the prompt we send to the LLM.
    
    The key instruction is to only use the provided context.
    This reduces hallucination (though it doesn't eliminate it entirely).
    """
    prompt = (
        "Answer the following question based only on the provided context. "
        "If the answer is not in the context, say 'The context does not "
        "contain enough information to answer this.' Be specific.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


def build_answer(question, context, model_name=LLM_NAME):
    """
    Generate an answer for the given question using retrieved context.
    
    This is the main function other modules call.
    """
    llm = get_llm(model_name)
    prompt = make_prompt(question, context)

    # rough truncation guard - flan-t5 cant handle super long input
    if len(prompt) > 2000:
        short_ctx = context[:1800]
        prompt = make_prompt(question, short_ctx)

    out = llm(prompt)
    answer_text = out[0]["generated_text"].strip()

    return {
        "answer": answer_text,
        "prompt_used": prompt,
        "model": model_name,
    }


def combine_answers(question, partial_answers, model_name=LLM_NAME):
    """
    When the planner splits a query into subtasks, each one gets answered
    separately. This function merges those partial answers into one response.
    """
    llm = get_llm(model_name)

    parts = []
    for i, ans in enumerate(partial_answers):
        parts.append(f"Part {i+1}: {ans}")

    combined = "\n".join(parts)

    prompt = (
        "Combine these partial answers into one clear response. "
        "Don't add information that isn't already there.\n\n"
        f"{combined}\n\n"
        f"Original question: {question}\n\n"
        "Combined answer:"
    )

    out = llm(prompt)
    answer_text = out[0]["generated_text"].strip()

    return {
        "answer": answer_text,
        "prompt_used": prompt,
        "model": model_name,
        "partial_answers": partial_answers,
    }


def show_answer(result):
    print(f"\n{'='*50}")
    print("ANSWER AGENT")
    print(f"{'='*50}")
    print(f"Model: {result['model']}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    ctx = (
        "The study uses a CNN trained on ImageNet. "
        "They use transfer learning and fine-tune on 10k medical images."
    )
    result = build_answer("What methodology was used?", ctx)
    show_answer(result)
