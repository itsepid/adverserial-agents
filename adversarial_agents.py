"""
Adversarial two-agent pattern: a Proposer drafts a solution,
a Critic's only job is to find flaws in it before it ships.

Built with agno + LiteLLM. Swap the model string for whatever
provider you're routing through LiteLLM.
"""

from agno.agent import Agent
from agno.models.litellm import LiteLLM

MODEL_ID = "gpt-4o-mini"  # swap for your LiteLLM-routed model of choice

proposer = Agent(
    name="Proposer",
    model=LiteLLM(id=MODEL_ID),
    instructions=[
        "You propose a concrete solution to the user's problem.",
        "Be decisive. State assumptions explicitly. Do not hedge.",
    ],
)

critic = Agent(
    name="Critic",
    model=LiteLLM(id=MODEL_ID),
    instructions=[
        "You are an adversarial reviewer. Your only job is to find flaws.",
        "Do not propose your own solution.",
        "Check for: unstated assumptions, edge cases, factual errors, "
        "and places where confidence exceeds evidence.",
        "If the proposal is genuinely solid, say so briefly — but look hard first.",
    ],
)


def adversarial_pass(task: str) -> dict:
    """Run one proposer -> critic round. Returns both outputs."""
    proposal = proposer.run(task)

    review_prompt = (
        f"Original task: {task}\n\n"
        f"Proposed solution:\n{proposal.content}\n\n"
        "Find flaws in this proposal."
    )
    critique = critic.run(review_prompt)

    return {
        "task": task,
        "proposal": proposal.content,
        "critique": critique.content,
    }


if __name__ == "__main__":
    result = adversarial_pass(
        "Design a caching strategy for a RAG pipeline serving Persian-language documents."
    )
    print("--- PROPOSAL ---")
    print(result["proposal"])
    print("\n--- CRITIQUE ---")
    print(result["critique"])

    # The gotcha worth mentioning in the post:
    # the critic can be just as confidently wrong as the proposer.
    # Log critiques the same way you log proposals — don't treat
    # "disagreement happened" as "the disagreement was correct."
