"""
memory.py
Keeps track of previous queries and answers within a session.

This is useful for two things:
1. Follow-up questions can use context from earlier
2. We can look back at what happened for debugging

Its just an in-memory store, nothing fancy. Doesn't survive
a restart. For anything serious you'd want a database.
"""

from datetime import datetime


class Memory:
    """Simple session memory. Stores interactions as a list of dicts."""

    def __init__(self):
        self.log = []

    def save(self, query, subtasks=None, chunks_used=None, answer="",
             critic_score=0, feedback="", final_answer="", revisions=0):
        """Record what happened for this query."""
        entry = {
            "time": datetime.now().isoformat(),
            "query": query,
            "subtasks": subtasks or [],
            "chunks_used": chunks_used or [],
            "first_answer": answer,
            "critic_score": critic_score,
            "feedback": feedback,
            "final_answer": final_answer,
            "revisions": revisions,
        }
        self.log.append(entry)

    def get_recent(self, n=3):
        """Get summary of last n interactions for context."""
        recent = self.log[-n:] if len(self.log) >= n else self.log
        if not recent:
            return "No previous queries."

        lines = []
        for e in recent:
            ans = e["final_answer"] or e["first_answer"]
            lines.append(f"Q: {e['query']}\nA: {ans}")
        return "\n\n".join(lines)

    def get_all_chunks(self):
        """Get every chunk that was ever retrieved."""
        all_c = []
        for e in self.log:
            all_c.extend(e.get("chunks_used", []))
        return all_c

    def count(self):
        return len(self.log)

    def last(self):
        return self.log[-1] if self.log else None

    def clear(self):
        self.log = []

    def show(self):
        print(f"\n{'='*50}")
        print(f"MEMORY ({len(self.log)} interactions)")
        print(f"{'='*50}")
        for i, e in enumerate(self.log):
            print(f"\n--- #{i+1} [{e['time']}] ---")
            print(f"  Q: {e['query']}")
            print(f"  Score: {e['critic_score']}/10")
            print(f"  Revisions: {e['revisions']}")
            preview = (e['final_answer'] or e['first_answer'])[:100]
            print(f"  A: {preview}...")


if __name__ == "__main__":
    mem = Memory()

    mem.save(
        query="What is the methodology?",
        answer="Uses a CNN approach.",
        final_answer="Uses a CNN with transfer learning.",
        critic_score=8,
    )
    mem.save(
        query="What are the limitations?",
        answer="Dataset is small.",
        final_answer="Main limitation is the small dataset of 1000 samples.",
        critic_score=7,
    )

    mem.show()
    print(f"\nRecent context:\n{mem.get_recent()}")
