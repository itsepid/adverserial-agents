# Adversarial Agents: Proposer + Critic

A minimal two-agent pattern where one agent proposes a solution and a second agent's *only job* is to find flaws in it before anything ships.

Built with [agno](https://github.com/agno-agi/agno) + [LiteLLM](https://github.com/BerriAI/litellm), so you can point it at any model provider LiteLLM supports.

## Why

Single-agent pipelines tend toward false confidence — a model can be fluent and wrong at the same time, and nothing in a single pass forces that to surface. Adding a critic whose entire role is disagreement catches errors earlier than a second model that just agrees faster.

## How it works

1. **Proposer** drafts a concrete, decisive solution to the task — no hedging.
2. **Critic** reviews the proposal looking specifically for unstated assumptions, edge cases, factual errors, and confidence that exceeds the evidence.
3. Both outputs are returned together, so you can inspect the disagreement instead of only seeing the final answer.

## Usage

```bash
pip install agno litellm
python adversarial_agents.py
```

Edit `MODEL_ID` to match whatever model you're routing through LiteLLM, and change the task string in `__main__` to try your own.

## The gotcha

The critic can be just as confidently wrong as the proposer. Treat "disagreement happened" and "the disagreement was correct" as two separate claims — log critiques the same way you log proposals, don't just trust that friction equals correctness.

## License

MIT — use it, break it, extend it.
