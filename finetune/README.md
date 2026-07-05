# Fine-tuning demo — teaching a small model the Kestrel Home support VOICE

This folder is the hands-on fine-tuning demo for the teaching session. It fine-tunes a
small LLM to **reply in the Kestrel Home support house format/voice** using
**QLoRA with Unsloth**, on a **free Google Colab T4 GPU**.

The whole point: **we teach the model a voice/format, not facts.** Every Kestrel support
reply follows the same five-part shape, and that consistency is exactly what the model
learns:

1. a short greeting line
2. `Answer:` — a direct answer
3. `Policy:` — the relevant policy / citation
4. `Next step:` — what to do or the escalation path
5. the sign-off `— Kestrel Home Support`

## Files

| File | What it is |
| --- | --- |
| `kestrel_support_dataset.jsonl` | ~190 chat-format training examples, every assistant reply in the fixed Kestrel five-part format. The format appears **only** in the replies (never in the prompt) and the system prompt varies (incl. empty) — see "Dataset design" below. Built by `build_dataset.py`. |
| `kestrel_support_dataset.curated.jsonl` | The 118 hand-written Q→reply pairs that seed the training set — the source of the natural voice. `build_dataset.py` re-prompts these and adds augmentation. |
| `kestrel_finetune_unsloth.ipynb` | The annotated Colab notebook: install → load base → LoRA → data → train → before/after → the "voice not facts" beat → export → a DPO section. |
| `kestrel_preference_dataset.jsonl` | 24 DPO preference pairs (`chosen` vs `rejected`) for the preference-tuning step beyond SFT. |
| `build_dataset.py` | Builds `kestrel_support_dataset.jsonl` from the curated core + canonical-fact augmentation, with format-free varied prompts (stdlib only, deterministic). |
| `check_format.py` | Evaluation: scores how many replies follow the five-part format (the curated set is 100%). |
| `unsloth-everywhere.md` | The finale: how Unsloth spans MCP + Skills + Fine-tune (Studio, the Unsloth MCP server, serving your model to Claude Code). |
| `README.md` | This file. |

> ### ⚠️ Which notebook should you actually train with?
> `kestrel_finetune_unsloth.ipynb` is written to **Unsloth's documented API** (the
> exact `FastLanguageModel` / `get_peft_model` / `SFTTrainer` calls and QLoRA
> hyperparameters from [unsloth.ai/docs](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)),
> and it's the **annotated teaching version** — but it has **not been run on a
> GPU here**, so treat it as API-correct, not battle-tested. **For a live session,
> the reliable path is Unsloth's own official, tested notebook:**
> 1. Open Unsloth's official **Llama-3.2 (1B) Conversational** notebook from
>    [unsloth.ai/docs/get-started/unsloth-notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
> 2. In its data cell, replace the sample dataset with **our** file:
>    `load_dataset("json", data_files="kestrel_support_dataset.jsonl", split="train")`
>    (upload `kestrel_support_dataset.jsonl` to the Colab session first).
> 3. Run all — same result, on a notebook Unsloth keeps current and tested.
>
> Use our notebook to *read and teach* the pipeline; use the official one to *run
> it in front of the room*.

## Two ways to run it

### (A) No-code — Unsloth Studio (best for non-coders / a live demo)

A local browser UI that does the same workflow with zero code.

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
unsloth studio
```

Then in the UI:
1. **Upload** `kestrel_support_dataset.jsonl`.
2. Pick the base model (`unsloth/Llama-3.2-1B-Instruct`).
3. Click **Train**.
4. Open **Model Arena** to compare the **base** model vs your **fine-tuned** model side-by-side.
5. **Export to Ollama** (GGUF) to run it locally afterward.

### (B) Notebook — the Colab (best for showing the mechanics)

1. Open `kestrel_finetune_unsloth.ipynb` in Google Colab.
2. Set the runtime to a GPU: **Runtime → Change runtime type → T4 GPU**.
3. Upload `kestrel_support_dataset.jsonl` into the Colab session (file panel on the left).
4. **Runtime → Run all.** The full run takes **~10 minutes** on a free T4.

## Before / after — what to watch for

- **Before** fine-tuning, the base model answers Kestrel questions in a generic, free-form,
  formatless way — no greeting, no `Answer:` / `Policy:` / `Next step:` markers, no sign-off.
- **After** ~190 examples, the model reliably emits the exact Kestrel five-part format —
  even though the format is **never mentioned in the prompt**, which is what makes it a
  *learned* behavior rather than instruction-following (see "Dataset design" below).

Also watch the **training loss** during the run: a healthy fine-tune settles around
**0.5–1.0**. If it dives toward **0**, the model is overfitting/memorizing.

## The punchline: voice, not facts

The notebook ends by asking the fine-tuned model about a **made-up new product** whose price
was never in the training data. It replies in flawless Kestrel voice — and gets the price
**wrong**, because it invents the fact it was never taught.

> **Fine-tuning taught the VOICE, not the FACTS. For facts, use RAG (last session).**
>
> **Production = fine-tune for voice + RAG for facts.**

## Dataset design: learned format, not instructed

The single most important thing about this dataset is **what is *not* in it**. Two rules:

1. **The five-part format appears only in the assistant replies — never in the system
   or user turns.** An earlier version of this demo put the format spec in the system
   prompt of every example. That's a trap: the model can then satisfy the training loss by
   just *following the instruction*, and the behavior collapses the moment you drop that
   instruction at inference (exactly what you'd see if you strip the format spec in Unsloth
   Studio or a bare API call). With the format kept out of the prompt, the *only* way to
   lower the loss is to **internalize** the format. That's a learned voice.
2. **The system prompt varies, including ~25% of examples with no system prompt at all.**
   We rotate a few neutral identity prompts ("You are Kestrel Home Support…") and leave many
   examples bare. This teaches the format as intrinsic to *being* Kestrel Home Support,
   robust to however the model is addressed — the **"train the way you'll infer"** principle.
   If at inference you'll use a bare prompt (or none), you must *train* on bare prompts.

Regenerate the set anytime (deterministic, stdlib only):

```bash
python build_dataset.py            # writes kestrel_support_dataset.jsonl (~190 examples)
python check_format.py             # should report 100% five-part adherence
```

`build_dataset.py` reads the hand-curated core (`kestrel_support_dataset.curated.jsonl`),
re-prompts it with the format-free pool, and adds fact-checked augmentation drawn from
`skills/kestrel-support/reference.md`, so every reply stays factually consistent with the
rest of the repo. Add seeds or prompts to the pool and re-run to grow it further.

> **Testing whether the tune "took":** query the fine-tuned model with the *bare* identity
> prompt (or an empty system prompt). If it still produces the five-part format, the tune
> worked. If it only formats when you paste the full format spec back into the prompt, it
> was leaning on the instruction — retrain with format-free prompts like these.

## Swapping the base model (one-line change)

In the notebook, change the model name in `FastLanguageModel.from_pretrained(...)`:

- `unsloth/Llama-3.2-1B-Instruct` — the default; Unsloth's recommended free-T4 starter.
- `unsloth/gemma-4-E4B-it` — newest small Gemma family, a step up in quality, still T4-friendly.
- `unsloth/gemma-3-270m` — the lightest option, fastest to train, great for a quick classroom demo.

## Troubleshooting

### `PicklingError` on `trainer.train()` at the end of training

If training runs all the way through (loss falls to ~0.5–1.0) but then dies at
checkpoint-save time with:

```
PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>:
it's not the same object as trl.trainer.sft_config.SFTConfig
```

the installed `trl` + `unsloth` ended up with **two copies of the `SFTConfig`
class** (newer TRL plus Unsloth's module patching). Pickle grabs the class off
your config instance, looks the name up again, and the identity check fails —
but only when it serializes the checkpoint, which is why training itself
finishes first. Your weights were trained fine; only the save blew up.

Fixes, in order:

1. **Pass a `SFTConfig`, not a `TrainingArguments`.** Section 5's training cell
   already does this (`args=SFTConfig(...)` with `dataset_text_field`,
   `max_seq_length`, `packing` moved inside it). `SFTConfig` subclasses
   `TrainingArguments`, so the hyperparameters are identical.
2. **Restart the runtime, then Run all.** The duplicate class is already loaded
   in the kernel — editing the cell won't evict it. In Colab: *Runtime →
   Restart runtime → Run all*. Also keep the import order: `from unsloth import
   FastLanguageModel` must run before any `from trl import ...`.
3. **Pin a known-good version pair** (fallback in the install cell), then
   restart: `pip install -U "unsloth==2025.1.1" "trl==0.12.2" "transformers<4.48"`
   (illustrative — bump to the latest compatible set if these have aged out).

## Night-before prep (do NOT train live in front of the room)

Fine-tuning takes ~10 minutes and the occasional Colab hiccup — don't gamble on training live.
The night before the session:

1. Run the notebook end-to-end on a T4 so all cells have real outputs saved.
2. If using Unsloth Studio, pre-train the model and confirm **Model Arena** shows the
   base-vs-tuned contrast, and that the Ollama export loads.
3. Have the finished notebook (with outputs) and, ideally, the exported GGUF ready so you
   can walk through results and re-run just the fun cells (the before/after and the
   "voice not facts" beat) live.
