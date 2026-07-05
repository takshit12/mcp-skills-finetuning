# Unsloth across the whole ladder — the "it all connects" finale

The three topics in this session aren't separate islands. **Unsloth** — the
fine-tuning engine from the notebook — happens to touch **all three rungs at
once**, which makes it the perfect way to end the session: one tool, MCP +
Skills + Fine-tune, live.

```
   5. FINE-TUNE   →  Unsloth trains the model (the notebook / Studio)
   4. SKILLS      →  an Unsloth "how to fine-tune" skill guides the agent
   3. MCP         →  the Unsloth MCP server lets an agent DRIVE training
   ── served back →  Unsloth Studio exposes your model as an OpenAI endpoint
                     that Claude Code / Cursor can point at
```

Below are four ways to show it — pick whichever fits your room and hardware.

---

## 1. Unsloth Studio — the no-code path (best for a non-technical room)

Studio is a local web UI for the whole fine-tune lifecycle — no code.

```bash
# macOS / Linux / WSL
curl -fsSL https://unsloth.ai/install.sh | sh
# Windows (PowerShell)
#   irm https://unsloth.ai/install.ps1 | iex

unsloth studio -H 0.0.0.0 -p 8888        # opens the UI at http://localhost:8888
```

What to demo in the UI (ties straight back to this repo):
- **Data Recipes** — drop in `docs/kestrel_home_faq.md` (or a PDF) and Studio
  auto-generates a training dataset. That's the *"data is 80% of the job"* slide,
  automated.
- **No-code training** — pick a base model (Llama-3.2-1B / Gemma 4), point it at
  `finetune/kestrel_support_dataset.jsonl`, click train. Live loss/GPU charts.
- **Model Arena** — load the **base** model and your **fine-tuned** model side by
  side and ask the same Kestrel question. This is the before/after payoff with
  zero code — the single most convincing thing for a non-technical audience.
- **Export** — one click to GGUF / Ollama / vLLM.

> Studio needs an NVIDIA GPU (or a Mac via MLX) for training. On a Mac with no
> NVIDIA GPU, use it for the chat / Data-Recipes / Model-Arena parts and do the
> actual training in Colab.

## 2. The Unsloth MCP server — MCP × fine-tuning (rung 3 meets rung 5)

There's a community MCP server that exposes Unsloth's fine-tuning pipeline **as
MCP tools**, so an agent (Claude Code, Cursor, …) can *drive* training by calling
tools — the literal intersection of this session's topics.

```bash
# add it to Claude Code (needs Node.js + an NVIDIA GPU + Python 3.10–3.12)
claude mcp add unsloth -- npx -y unsloth-mcp-server
claude mcp list        # unsloth → ✔ connected
```
Its tools: `check_installation`, `list_supported_models`, `load_model`,
`finetune_model` (LoRA/QLoRA on a dataset), `generate_text`, `export_model`.
Then just ask Claude: *"Fine-tune unsloth/Llama-3.2-1B on my Kestrel dataset with
LoRA rank 16"* and watch it call `finetune_model` — **MCP driving a fine-tune.**

Source: [github.com/OtotaO/unsloth-mcp-server](https://github.com/OtotaO/unsloth-mcp-server).

> No GPU? There's also a **docs** MCP server — `jdanas/unsloth-mcp` — that gives
> an agent Unsloth's documentation as a tool (no training, no GPU). Handy for
> "how do I set the LoRA rank?" answered from the real docs, live.

## 3. Serve your fine-tuned model back to Claude Code / Cursor

Unsloth Studio exposes your local model over an **OpenAI-compatible endpoint**,
so you can point Claude Code (or Cursor / Codex / Continue) at *your fine-tuned
Kestrel model* instead of a cloud model — closing the loop from rung 5 back to
the agent. See [unsloth.ai/docs/basics/claude-code](https://unsloth.ai/docs/basics/claude-code).

## 4. A skill that guides fine-tuning (rung 4)

You can package Unsloth know-how as an **Agent Skill** — a `SKILL.md` whose
`description` is "help me configure and run an Unsloth LoRA/QLoRA fine-tune",
with the recipe (r=16, α=16, batch 2 × grad-accum 4, lr 2e-4) and links in the
body. Same pattern as `skills/kestrel-support/` — the model pulls it in only when
you're actually setting up a fine-tune. (Community "Unsloth" skills exist in MCP
skill directories too.)

---

**The one line to land:** *the same Unsloth you used to change the model's
weights also ships as an MCP server, plugs into Studio, and can be wrapped in a
Skill — so a single tool literally demonstrates the whole ladder at once.*
