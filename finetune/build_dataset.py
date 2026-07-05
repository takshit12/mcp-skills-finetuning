#!/usr/bin/env python3
"""Build the Kestrel Home support TRAINING set — the "learned format" way.

Teaching point: **"the data is 80% of the job"** — and *how you shape the data*
decides whether the model **learns** a behavior or merely **obeys** an
instruction. Two design rules make this a real fine-tuning demo:

1. **The format lives ONLY in the assistant replies.** Every assistant turn
   follows the fixed five-part Kestrel house format:

       <greeting line>
       Answer: ...
       Policy: ...
       Next step: ...
       — Kestrel Home Support

   ...but NOTHING in the system or user turns ever describes that format. If we
   told the model the format in the system prompt (as the first draft of this
   demo did), the model could pass by *following instructions* — and the moment
   you strip that instruction at inference time, the behavior collapses. By
   keeping the format out of the prompt, the only way to reduce training loss is
   to *internalize* the format. That is what "fine-tuning taught the voice" means.

2. **The system prompt VARIES — including empty.** We rotate a small pool of
   neutral identity prompts ("You are Kestrel Home Support...") and deliberately
   include examples with **no system message at all**. This teaches the format
   as intrinsic to *being* Kestrel Home Support, robust to however the model is
   addressed at inference (or not addressed). Principle: **train the way you'll
   infer.** If you'll query it with a bare prompt, train it on bare prompts.

The training file is assembled from two sources:

  * the hand-curated core (``kestrel_support_dataset.curated.jsonl``, 118
    natural (question -> five-part reply) pairs), re-prompted with the pool, and
  * programmatic **augmentation** built from the canonical facts in
    ``skills/kestrel-support/reference.md`` — extra phrasings of the same
    policies so the model sees many ways to ask and learns the format is
    invariant to wording.

All assistant replies stay factually consistent with reference.md.

Standard library only. Deterministic (fixed seed). Run:

    python build_dataset.py                    # writes kestrel_support_dataset.jsonl
    python build_dataset.py --n 200            # cap the number of examples
    python build_dataset.py --out mine.jsonl   # write somewhere else
    python build_dataset.py --no-curated       # augmentation only (pure synthetic)
"""

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CURATED_CORE = os.path.join(HERE, "kestrel_support_dataset.curated.jsonl")
DEFAULT_OUT = os.path.join(HERE, "kestrel_support_dataset.jsonl")
SEED = 3407

# ---------------------------------------------------------------------------
# System-prompt pool — NEUTRAL identity only. Crucially, NONE of these mention
# the greeting / Answer: / Policy: / Next step: / sign-off structure. The empty
# string means "emit no system message at all" for that example. Entries are
# intentionally repeated to weight them: the bare identity prompt (the likely
# inference default) and the no-prompt case get extra coverage.
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = [
    "",  # no system message at all — pure "learned format" case
    "You are Kestrel Home Support, the assistant for Kestrel Home smart thermostats.",
    "You are a customer support assistant for Kestrel Home smart thermostats.",
    "You are Kestrel Home Support, the assistant for Kestrel Home smart thermostats.",  # repeat: inference default
    "You are Kestrel Home Support.",
    "",  # repeat: weight the no-prompt case up
    "You are the support assistant for Kestrel Home. Help customers with questions about their smart thermostats.",
    "You are Kestrel Home Support. Be warm, concise, and accurate.",
]

# Greeting variants rotate so the greeting *slot* is clearly flexible while the
# rest of the structure stays fixed.
GREETINGS = [
    "Thanks for reaching out to Kestrel Home.",
    "Happy to help with that.",
    "Good to hear from you.",
    "Thanks for getting in touch.",
    "Appreciate you contacting Kestrel Home Support.",
    "Let's get this sorted for you.",
    "Hi there — thanks for reaching out!",
    "Glad you asked.",
]

# ---------------------------------------------------------------------------
# Augmentation seeds. Each seed carries several natural user phrasings plus one
# (answer, policy, next_step) drawn straight from reference.md, so the synthetic
# rows are factually correct AND format-consistent. None of the user phrasings
# hint at the output format.
# ---------------------------------------------------------------------------
SEEDS = [
    {
        "q": [
            "How long is the Kestrel warranty?",
            "What's the warranty period on my thermostat?",
            "How many months of coverage do I get?",
            "Remind me how long Kestrel warranties run?",
        ],
        "answer": "Every Kestrel Home device — Nest-7, Nest-7 Pro, and Aria — carries a 26-month warranty from the original purchase date.",
        "policy": "Kestrel's standard warranty is 26 months from purchase — a deliberate two-month buffer beyond the usual 24 — for indoor use only.",
        "next_step": "Keep your proof of purchase handy; if a defect appears in that window, reply here and we'll arrange a replacement.",
    },
    {
        "q": [
            "Does the warranty cover water damage?",
            "My thermostat got wet — is that covered?",
            "Will you replace a unit with water damage under warranty?",
        ],
        "answer": "Water damage isn't covered under the Kestrel warranty.",
        "policy": "The warranty excludes water damage, damage from unauthorized third-party firmware, and damage from outdoor mounting — all devices are indoor-rated only.",
        "next_step": "If you think the fault is a manufacturing defect rather than water, reply and we'll open a Tier 2 ticket to review it.",
    },
    {
        "q": [
            "I flashed custom firmware and bricked it — warranty?",
            "Is third-party firmware damage covered?",
            "I loaded unofficial firmware and now it won't boot. Can you replace it?",
        ],
        "answer": "Damage from unauthorized third-party firmware isn't covered under warranty.",
        "policy": "The warranty excludes third-party-firmware damage, water damage, and outdoor-mounting damage; only manufacturing defects on indoor-used devices qualify.",
        "next_step": "If you believe the hardware was already faulty before the firmware change, reply and we'll open a Tier 2 ticket to look into it.",
    },
    {
        "q": [
            "Can I mount my thermostat outside?",
            "Is outdoor installation covered?",
            "I put my Nest-7 on a covered porch — is that OK under warranty?",
        ],
        "answer": "No — Kestrel thermostats are indoor-rated only, and outdoor-mounting damage isn't covered.",
        "policy": "All models are rated for indoor use; damage from outdoor mounting is excluded from warranty, alongside water and third-party-firmware damage.",
        "next_step": "For an unheated or exposed space, reply with your setup and we'll suggest an indoor-safe placement.",
    },
    {
        "q": [
            "What's your return window for a refund?",
            "How long do I have to return for money back?",
            "Can I get a full refund if I send it back this week?",
        ],
        "answer": "Returns are accepted within 45 days of delivery for a full refund.",
        "policy": "Full refunds require the device to be in resalable condition with its original packaging, returned within 45 days of delivery — no restocking fee inside that window.",
        "next_step": "Reply with your order number and we'll email you a prepaid return label.",
    },
    {
        "q": [
            "Is there a restocking fee if I return an opened unit after 60 days?",
            "What's the fee for returning something 4 months in?",
            "Do you charge a restocking fee outside the return window?",
        ],
        "answer": "Yes — a 12% restocking fee applies to opened returns after the 45-day window but still within the 26-month warranty.",
        "policy": "The 12% fee only applies to opened returns outside 45 days and inside warranty; it's waived entirely when the return reason is a confirmed hardware defect.",
        "next_step": "Tell us the return reason — if it's a defect we'll waive the fee and cover shipping.",
    },
    {
        "q": [
            "Do I pay the restocking fee if my unit is defective?",
            "Is the fee waived for a hardware fault?",
            "My device is faulty — will I still be charged 12%?",
        ],
        "answer": "No — the 12% restocking fee is waived entirely for a confirmed hardware defect.",
        "policy": "Restocking applies only to normal opened returns outside 45 days; a confirmed manufacturing defect waives the fee and shipping.",
        "next_step": "Reply with your model and a short description of the fault and we'll start a no-fee defect return.",
    },
    {
        "q": [
            "How does a warranty replacement ship?",
            "What shipping speed do replacements go out at?",
            "Do I pay for shipping on a warranty replacement?",
        ],
        "answer": "Standard warranty replacements ship 2-day air at no cost.",
        "policy": "Warranty replacements ship 2-day air free; expedited overnight is an optional $35 upgrade.",
        "next_step": "Reply with your order number and we'll dispatch your replacement 2-day air.",
    },
    {
        "q": [
            "Can I get a replacement overnight?",
            "How much is overnight replacement shipping?",
            "Is expedited shipping available on a warranty swap?",
        ],
        "answer": "Yes — expedited overnight replacement is available for a $35 fee.",
        "policy": "Standard warranty replacements ship 2-day air at no cost; overnight is an optional $35 upgrade.",
        "next_step": "Reply 'overnight' with your order number and we'll add the $35 expedite and dispatch today.",
    },
    {
        "q": [
            "How fast is first response on the free plan?",
            "What's the response time if I don't pay for support?",
            "How long until someone replies on the standard plan?",
        ],
        "answer": "The included Standard plan targets an 18 business-hour first response by email.",
        "policy": "Standard support is email-only and included with every device, with an 18 business-hour first-response target.",
        "next_step": "If you need it faster, Comfort+ ($4.99/mo) adds live chat with a 2-hour target — reply and we can walk through upgrading.",
    },
    {
        "q": [
            "What do I get with Comfort+?",
            "How much is Comfort+ and how fast is it?",
            "What are the Comfort+ response times?",
        ],
        "answer": "Comfort+ is $4.99/month (or $49/year) and targets a 2-hour first response via live chat.",
        "policy": "Comfort+ includes live chat with a 2-hour target, 8 AM–8 PM Mountain Time, Monday–Saturday.",
        "next_step": "Start a live chat from your Kestrel app during those hours and we'll pick it up within the SLA.",
    },
    {
        "q": [
            "What's the fastest support tier?",
            "Which plan gets the quickest help?",
            "Is there a phone-support tier?",
        ],
        "answer": "Comfort+ Pro is the fastest — a 20-minute first-response target with phone support and a dedicated escalation line.",
        "policy": "Comfort+ Pro is $9.99/month (free for the first year with an Aria) and includes phone plus a dedicated escalation line at a 20-minute target.",
        "next_step": "If you own an Aria, reply and we'll check whether your first year of Comfort+ Pro is already included.",
    },
    {
        "q": [
            "Do you have a discount code I can use?",
            "Is there a promo code for my order?",
            "Can I get a goodwill discount?",
        ],
        "answer": "We can apply WARMTH2024 for $20 off a single order.",
        "policy": "WARMTH2024 is an agent-issued goodwill code worth $20 off any single order, valid through the current fiscal year, limited to one use per customer.",
        "next_step": "Reply to confirm and we'll apply WARMTH2024 to your next order.",
    },
    {
        "q": [
            "How does the referral program work?",
            "What's the referral discount?",
            "How many times can my household use referral codes?",
        ],
        "answer": "Referral codes (KESTREL-REF-XXXX) give 15% off, with up to 3 redemptions per household per calendar year.",
        "policy": "Referral codes are customer-generated (not agent-issued): 15% off, capped at 3 redemptions per household per year, and they don't stack with WARMTH2024.",
        "next_step": "Share your referral code with a friend to have them redeem it — reply if you'd like help finding yours in the app.",
    },
    {
        "q": [
            "My ticket has been open for days with no fix. What now?",
            "Nobody has resolved my issue in over 72 hours.",
            "My ticket is 4 days old and still open — what happens?",
        ],
        "answer": "Any ticket open more than 72 hours is automatically escalated to a Tier 2 specialist.",
        "policy": "The 72-hour auto-escalation applies on top of your plan's SLA so nothing stalls in the queue.",
        "next_step": "No action needed on your side; reply and we'll confirm your ticket has moved to Tier 2 and check its status.",
    },
    {
        "q": [
            "I smell smoke coming from my thermostat!",
            "My unit is sparking and smells like it's burning.",
            "I think I saw a small flame near the wiring.",
            "There's a burning smell when the device runs.",
        ],
        "answer": "Please unplug the device now if it's safe to do so — anything involving smoke, fire, sparking, or a burning smell is treated as a safety issue.",
        "policy": "Any report mentioning fire, smoke, burning smell, or sparking is immediately escalated to Tier 3 Safety Response, bypassing the normal SLA queue.",
        "next_step": "Safety Response is being paged now; keep the unit unplugged and reply with your model and order number — don't keep using it.",
    },
    {
        "q": [
            "Who do I contact about a support escalation?",
            "Who owns the escalation queue?",
            "What's the general escalation contact?",
        ],
        "answer": "Marcus Webb, our Support Operations Lead, owns the general escalation queue.",
        "policy": "General support escalations route to Marcus Webb (ext 4471); hardware defect and recall coordination goes to Sana Idrissi (ext 4488).",
        "next_step": "Reply with your ticket number and we'll route it to the escalation queue for you.",
    },
    {
        "q": [
            "Who handles hardware defects and recalls?",
            "What's the extension for hardware quality?",
            "Who should I talk to about a possible recall?",
        ],
        "answer": "Sana Idrissi, our Hardware Quality Lead, handles defects and recall coordination.",
        "policy": "Hardware defect and recall issues route to Sana Idrissi (ext 4488); general escalations go to Marcus Webb (ext 4471).",
        "next_step": "Reply with your model and serial number and we'll open a hardware-quality review.",
    },
    {
        "q": [
            "What thermostat models do you sell?",
            "Give me the Kestrel lineup and prices.",
            "What are the differences between your models?",
        ],
        "answer": "Three: the Nest-7 ($129, up to 2 zones), the Nest-7 Pro ($189, up to 6 zones with an air-quality sensor), and the flagship Aria ($279, up to 12 zones with air-quality plus humidity sensing and an e-ink display).",
        "policy": "All three run KestrelOS and carry the same 26-month indoor-use warranty; they differ in zones, sensors, battery backup, and price.",
        "next_step": "Tell us your home's size and zone count and we'll suggest the right model.",
    },
    {
        "q": [
            "How many zones does the Aria support?",
            "How long is the Aria's battery backup?",
            "What makes the Aria the flagship?",
        ],
        "answer": "The Aria supports up to 12 heating/cooling zones and has a 20-hour battery backup.",
        "policy": "The Aria (released June 2024, $279) adds air-quality and humidity sensing, an e-ink display, and 20-hour backup on top of 12-zone support.",
        "next_step": "If you need help mapping zones, reply and we'll walk through the Aria setup with you.",
    },
    {
        "q": [
            "Does the Nest-7 Pro have an air quality sensor?",
            "What does the Pro track for air quality?",
            "How long is the Nest-7 Pro's battery backup?",
        ],
        "answer": "Yes — the Nest-7 Pro has an air-quality sensor (VOC and PM2.5) and a 14-hour battery backup.",
        "policy": "The Nest-7 Pro ($189, up to 6 zones, released September 2022) adds VOC/PM2.5 air-quality sensing over the base Nest-7.",
        "next_step": "Reply if you'd like help reading the air-quality data in your Kestrel app.",
    },
    {
        "q": [
            "What firmware do Kestrel thermostats run?",
            "Which KestrelOS version is current?",
            "When do firmware updates install?",
        ],
        "answer": "They run KestrelOS, currently v4.3.1, which auto-updates overnight between 2:00 and 4:00 AM local time.",
        "policy": "KestrelOS updates push automatically overnight; you don't need to trigger them manually.",
        "next_step": "If an update didn't apply, reply and we'll help you confirm the device is online during the overnight window.",
    },
    {
        "q": [
            "Can you just tell me if my unusual setup will work?",
            "My configuration is weird — will it be supported?",
            "What if my question isn't in your standard info?",
        ],
        "answer": "For anything outside our standard policy I won't guess — I'll open a Tier 2 ticket so a specialist can confirm the right answer.",
        "policy": "House rule: a slow correct answer beats a fast wrong one, so anything not covered by policy goes to a Tier 2 ticket rather than a guess.",
        "next_step": "Reply with the details of your setup and we'll open that Tier 2 ticket to confirm.",
    },
    {
        "q": [
            "Where is Kestrel Home based?",
            "Who founded Kestrel Home?",
            "How many employees does Kestrel have?",
        ],
        "answer": "I don't have verified company background details on hand, so I won't guess at that.",
        "policy": "House rule: never invent facts — anything outside documented support policy goes to a Tier 2 ticket for a confirmed answer.",
        "next_step": "If this is for a support matter, reply and we'll open a Tier 2 ticket to get you a confirmed answer.",
    },
]


def build_reply(greeting, answer, policy, next_step):
    """Assemble the fixed five-part Kestrel reply from its parts."""
    return (
        f"{greeting}\n\n"
        f"Answer: {answer}\n\n"
        f"Policy: {policy}\n\n"
        f"Next step: {next_step}\n\n"
        "— Kestrel Home Support"
    )


def make_example(system_prompt, user, assistant):
    """Build one chat row. An empty system_prompt omits the system turn."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def load_curated_pairs(path):
    """Read (user, assistant) content pairs from the curated core file.

    We keep the hand-written assistant replies verbatim (natural voice) and
    drop the original system prompt — the pool re-assigns a format-free one.
    """
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            user = next((m["content"] for m in obj["messages"] if m["role"] == "user"), None)
            assistant = next((m["content"] for m in obj["messages"] if m["role"] == "assistant"), None)
            if user and assistant:
                pairs.append((user, assistant))
    return pairs


def build_augmentation():
    """Expand every seed phrasing into a (user, assistant) pair."""
    pairs = []
    g = 0
    for seed in SEEDS:
        for question in seed["q"]:
            greeting = GREETINGS[g % len(GREETINGS)]
            g += 1
            reply = build_reply(greeting, seed["answer"], seed["policy"], seed["next_step"])
            pairs.append((question, reply))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None,
                        help="cap the number of examples written")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output path (default: kestrel_support_dataset.jsonl)")
    parser.add_argument("--no-curated", action="store_true",
                        help="skip the curated core; write augmentation only")
    args = parser.parse_args()

    rng = random.Random(SEED)

    pairs = []
    if not args.no_curated and os.path.exists(CURATED_CORE):
        pairs.extend(load_curated_pairs(CURATED_CORE))
    pairs.extend(build_augmentation())

    # Assign system prompts round-robin BEFORE shuffling so the pool (including
    # the empty / no-prompt cases) is evenly represented across all examples.
    examples = [
        make_example(SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)], user, assistant)
        for i, (user, assistant) in enumerate(pairs)
    ]
    rng.shuffle(examples)

    if args.n is not None:
        examples = examples[: args.n]

    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n_empty = sum(1 for ex in examples if ex["messages"][0]["role"] != "system")
    print(f"Wrote {len(examples)} examples to {args.out}")
    print(f"  with a system prompt : {len(examples) - n_empty}")
    print(f"  with NO system prompt: {n_empty}")
    print("  format spec in prompts: 0  (the format is learned, not instructed)")
    print("\n===== sample row =====\n")
    print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    main()
