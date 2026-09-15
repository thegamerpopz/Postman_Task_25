# Task 1.7 — Attention Pattern Comparison: Writeup

## Setup

All three models share the same architecture and training budget — only the attention mask changes:

| | Dense | BigBird | Sliding Window |
|---|---|---|---|
| Params | 3.691841M | 3.691841M | 3.691841M |
| Attends to | every previous token (full causal) | local window (32) + 1 global block + 2 random blocks | local window only (32 previous tokens) |
| Final train loss | 1.2392 | 1.2328 | **1.1774** |
| Final val loss | 1.5293 | 1.5174 | **1.5049** |

(`window_size=32`, `attn_block_size=16`, `num_global=1`, `num_local_blocks=1`, `num_random=2`, `block_size=256`, 5000 iterations, single run each — no multiple seeds, so small gaps should be read as suggestive rather than conclusive.)

## What each pattern loses, in theory

- **Sliding window** loses everything outside the last 32 tokens. A token at position 200 has no direct route to information at position 50 — no global anchor, no random long-range links. Any dependency longer than the window can only be captured indirectly, through information "relayed" token-by-token as the window slides — and with only `n_layer=2`, there's very little depth for that relay to happen.
- **BigBird** loses most long-range connections too, but recovers a partial fallback: every token can always reach the first block (global) and 2 random earlier blocks. That doesn't guarantee coverage of any *specific* faraway dependency, but it means long-range information isn't structurally unreachable — it's reachable with some probability, and the global block gives every token at least one full-sequence-visible anchor point.
- **Dense** loses nothing — every token can attend to every earlier token directly.

So the theoretical ranking on "information available" is Dense > BigBird > Sliding Window.

## The result was the opposite of that ranking — and here's why

Sliding window had the *lowest* validation loss, BigBird was second, dense was last. So sparse was not worse here; if anything it was mildly better. A few things explain this:

**1. This is character-level Shakespeare, and the task is dominated by local structure.** Predicting the next character depends overwhelmingly on the last few words — spelling, word completion, common bigrams/trigrams, and short-range syntax. There's very little in this task that actually requires reasoning over something 200+ characters back. A 32-token window already covers several words of context, which is close to the effective horizon the task rewards.

**2. Full attention has to spend capacity learning to ignore irrelevant tokens.** Dense attention gives the model 256 candidate positions to weigh at every step, most of which are uninformative for character-level prediction. With a 2-layer, 4-head model, that's a harder optimization problem for comparatively little benefit — the model has to learn *not* to attend to most of the sequence rather than getting that for free. The sparse patterns hard-code a good prior (recent context matters most), which acts like a helpful inductive bias given the limited capacity and training steps here.

**3. Depth matters for when global tokens actually pay off.** Global/random long-range links are most valuable when a model needs multiple hops to route information across a long sequence (e.g., "the noun introduced 150 tokens ago") — that typically needs either a longer effective window or more layers to compose intermediate representations. With only `n_layer=2`, there isn't much depth for BigBird's global/random tokens to be exploited; they're available, but the model doesn't have to work as hard to make use of them the way a deeper model would. That's also consistent with BigBird landing *between* dense and sliding window rather than clearly beating both — the global/random connections add a bit of optimization noise (attending to a token that's occasionally irrelevant) without yet paying off the way they would in a deeper or longer-context setup.

**4. Structural learning (speaker tags) shows up in the sparse models' output.** Both the BigBird and sliding-window samples reproduce the play's `CHARACTER NAME:\ndialogue` formatting correctly (`MARCIUS:`, `CORIOLANUS:`, `BRUTUS:`, `CAMILLO:`, `MERCUTIO:`), while the dense sample's excerpt doesn't show this pattern as cleanly. That's a locally-repeating structural pattern (name, colon, newline) — exactly the kind of thing a local window is well-suited to pick up, and it didn't require dense attention's extra reach to learn it.

## If sparse *had* been worse, what we'd expect to see

Since it's worth stating explicitly per the prompt: sparse attention is expected to underperform when a task has genuine long-range dependencies — e.g., word-level or document-level modeling where a fact stated far earlier constrains a later token (coreference, matching a citation later in a document, long-form consistency). In that regime, a 32-token window would lose necessary information outright, and BigBird's global + 2 random blocks would only occasionally happen to preserve the needed link, causing a visible and growing loss gap. None of that shows up here because char-level Shakespeare over a 256-token window doesn't have dependencies that long — the local window already contains basically everything the model can usefully exploit at this scale.

## Bottom line

For this task and this model scale, sparse attention did not lose meaningful information relative to dense, and both sparse variants trained to a lower validation loss in the same number of steps. That result is task- and scale-specific: it reflects the fact that char-level Shakespeare prediction is a locally-dominated problem and the model is shallow enough that dense attention's extra reach isn't being exploited. It should not be read as "sparse attention is generally better than dense" — only that, here, the inductive bias of restricting attention to nearby (plus occasionally global/random) tokens matched the task well enough to make full attention's extra flexibility a net cost rather than a benefit.
