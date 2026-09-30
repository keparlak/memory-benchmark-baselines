# Schema-bump: does agent memory keep up when a tool changes mid-run?

A tool's schema changes partway through an agent's run (rename, remove, split,
add a required field, change a type). The change is announced once in the
history. At the end the agent must call that tool; the call either validates
against the new schema or it doesn't. No judge model.

Every arm uses the same agent model, prompt and history. Only the memory layer
differs. Idea from [@kartikb753](https://x.com/kartikb753); the length sweep
answers his follow-up question about whether an append-only store "grows until
retrieval drowns."

Code comments and console output are in Turkish; file names use *deney* (= experiment).

## Results

### Experiment 5 — 61-event runs

| memory arm | valid call | runs |
|---|---|---|
| "newest schema wins" policy (~20 lines) | **100%** | 40 |
| full history in context | **95%** | 40 |
| Mem0 OSS 2.1.0 | **20%** | 15 |
| TF-IDF + recency | 2.5% | 40 |

Mem0 by mutation type: type_change 2/2, add_required 1/3, rename 0/5, remove 0/4, split 0/1.

The non-Mem0 arms were repeated on three models (Nemotron 3 Super 120B,
Nemotron 3 Ultra 550B, DeepSeek v4.1 Flash) with the same outcome. The Mem0 arm
was not repeated across models.

### Experiment 6 — run length as the only variable

Same six mutations at every length. Two retrieval queries: a schema-describing
query (`"{tool} current schema arguments"`, used in Experiment 5) and the agent's
own task (`"call {tool} to {task}"`).

| events | store size | Mem0, task query | Mem0, schema query | full history |
|---|---|---|---|---|
| 20 | 16 | 83% | 83% | 100% |
| 60 | 47 | 67% | 67% | 100% |
| 120 | 103 | 17% | 50% | 100% |
| 200 | 182 | 33% | 67% | 100% |

Pooled, task query: 9/12 at ≤60 events vs 3/12 at ≥120 (Fisher exact, two-sided,
p = 0.039). Schema query: 9/12 vs 7/12 (p = 0.667). Full history: 24/24.

- **No garbage collection.** 2,840 Mem0 write decisions across both
  experiments, all `ADD`, zero `UPDATE`/`DELETE`. Store size tracks history
  (0.78–0.91 memories per event).
- **Read side.** Under the task query, the memory describing the schema change
  has median rank 5 / 28 / 47 at 20 / 60 / 120 events (noisy at 200). In 5 of 9
  runs of ≥60 events where it existed, it fell outside the top 20 the agent sees.
- **Write side.** The single-pass extraction does not always store the change as
  its own memory. Stored: 14/14 correct (schema query), 11/14 (task query).
  Not stored: 2/10 and 1/10.

## Context: this is Mem0's documented design

Mem0 v2.0.0 (2026-04-14) moved to ADD-only extraction:
*"no more UPDATE/DELETE events"* ([changelog](https://docs.mem0.ai/changelog/sdk)),
with supersession delegated to retrieval: *"Retrieval handles ranking: the most
relevant, current information surfaces first"*
([migration guide](https://docs.mem0.ai/migration/oss-v2-to-v3)).

In the open source package, ranking is `mem0/utils/scoring.py::score_and_rank`,
combining semantic similarity, BM25 and an entity boost. There is no timestamp
term. Time-aware retrieval (`reference_date`) and `decay` are Platform-only.
**The hosted Platform was not tested.**

## Pitfalls we hit (check these if you reproduce)

- **`top_k`, not `limit`.** In mem0 2.1.0, `get_all(limit=…)` and
  `search(limit=…)` silently ignore `limit` (it lands in `**kwargs`) and return
  the default 20. Experiment 5's `stored`/`loose` fields were measured this way
  and are **unreliable**; `ok` and `slots` are unaffected. Fixed in the code.
- **`OPENROUTER_API_KEY` overrides your config.** If it is set in the
  environment, Mem0's OpenAI client ignores the configured `base_url` and routes
  to OpenRouter. `deney5_llm.py` reads `.env` but never exports OpenRouter keys.
- **Rate limits are not model errors.** Service failures (429/503) are excluded
  from denominators; Mem0's own LLM calls go through the same limiter with
  retries.
- **The schema query favours the memory layer** — it lexically matches the
  change announcement. Use the task query as the realistic measurement.

## Reproduce

```bash
pip install mem0ai==2.1.0 fastembed qdrant-client
echo "NVIDIA_API_KEY=nvapi-..." > .env      # NVIDIA NIM free tier
echo "NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1" >> .env

# Experiment 6 (resumable; each call runs as many episodes as fit the budget)
python deney6_uzunluk.py 500     # repeat until it prints TAMAM
python deney6_analiz.py

# Experiment 5, Mem0 arm, in chunks of 3 episodes
python deney5_mem0_parca.py 0 3  # then 3 3, 6 3, 9 3, 12 3
python deney5_mem0_analiz.py

# Experiment 5, other arms; multi-model check
python deney5_asama2.py
python deney5_coklu_model.py 12
```

`results/` holds the raw output of every run reported above, including full
memory-store and ranking texts for Experiment 6. `results/deney5_asama2_sonuc.json`
has the 40-run non-Mem0 arms; its Mem0 row predates the fixes above — use
`results/deney5_mem0/` instead. `notes/` contains the lab notes (Turkish).

## Limits

- Open source Mem0 2.1.0 only; one agent model for the Mem0 runs.
- Six runs per length; the pooled effect is significant but only just.
- Synthetic tool histories; events written to Mem0 in batches of 10.
- One Experiment 6 run (200 events, `add_required`) lost 10 events to a Mem0
  parse error on the model's extraction output; it is kept and flagged.
- MemoryAgentBench (July 2025) evaluated Mem0's earlier two-pass pipeline; these
  experiments test the ADD-only pipeline. They are different algorithms.
