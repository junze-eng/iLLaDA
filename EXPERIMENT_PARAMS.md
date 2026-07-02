# Experiment parameter reference

Generated from `test_config.yaml` as of the last edit to this file. This is a snapshot for quick
reference -- if `test_config.yaml` changes, re-generate this rather than trusting stale numbers.

**Currently selected by default** (`run.tasks` in `test_config.yaml`): `arness` (Task 3).
This is only the config file's default -- it does not by itself prove what is actually running on
the Pod right now. Confirm live GPU state directly (`nvidia-smi`, or check which `outputs/.../`
folder is being written to) rather than trusting this file alone.

| Field | Meaning |
|---|---|
| experiments | how many distinct decoding conditions in the task |
| samples/exp | how many questions each condition runs on |
| total gens | experiments × samples -- the actual generation count, i.e. the "size" of the run |
| gen_length | generation token budget per sample |
| gen_steps | total diffusion steps (see block/steps coupling note below) |
| block_length | block size (== gen_length means single block, no semi-AR split) |

## Task 1 -- `parallel`: GSM8K/MBPP parallel-decoding sweep

| experiment | benchmark | arness label | gen_length | gen_steps | block_length | sample_limit |
|---|---|---|---|---|---|---|
| task1_gsm8k_ar_like | gsm8k | 1 | 256 | 256 | 256 | 50 |
| task1_gsm8k_mild_parallel | gsm8k | 4 | 256 | 64 | 256 | 50 |
| task1_gsm8k_strong_parallel | gsm8k | 8 | 256 | 32 | 256 | 50 |
| task1_mbpp_ar_like | mbpp | 1 | 512 | 512 | 512 | 50 |
| task1_mbpp_mild_parallel | mbpp | 4 | 512 | 128 | 512 | 50 |
| task1_mbpp_strong_parallel | mbpp | 8 | 512 | 64 | 512 | 50 |

**6 experiments × 50 samples = 300 generations total.** temperature=0.0, threshold=null for all
(from `defaults:`). Status: done, real GPU results in `outputs/task2_context_full/summary_all.csv`
(GSM8K 82/50/22%, MBPP 68/26/8% per ar_like/mild/strong -- verify against current data before citing).

## Task 2 -- `context`: RULER NIAH context-length sweep (colleague's)

| experiment | benchmark | context_length sweep | needle_position sweep | gen_length | num_samples |
|---|---|---|---|---|---|
| task2_ruler_niah_1_2_4_8k | ruler_niah_single_1 | 1024 / 2048 / 4096 / 8192 | front / middle / back | 128 | 20 |

**1 experiment × 4 context_lengths × 3 needle_positions × 20 samples = 240 generations total.**
gen_steps=128, block_length=128 (single block), max_seq_len=8192. Status: prep-only as of
2026-07-02 per direct user confirmation ("task2还没有") -- no completed GPU runs yet.

## Task 3 -- `arness`: two-sample detailed trace case study (colleague's, currently active default)

| experiment | benchmark | sample | gen_length | block_length | gen_steps sweep | threshold sweep |
|---|---|---|---|---|---|---|
| arness_trace_gsm8k_sample7 | gsm8k | idx 7 | 256 | 64 | 256/128/64/32/16 (5) | null/0.6/0.7/0.8/0.9 (5) |
| arness_trace_mbpp_sample6 | mbpp | idx 6 | 512 | 128 | 512/256/128/64/32/16 (6) | null/0.6/0.7/0.8/0.9 (5) |

**25 + 30 = 55 generations total** (1 fixed sample per experiment, swept across steps×threshold
combos -- not 55× multiple samples). Status per last check: not yet run by colleague.

Note: this task slot previously held a different sweep (`task3_gsm8k/mbpp_threshold_counterfactual`)
that already produced real results (no-threshold 22%, threshold=0.8 40%, threshold=0.9 10% at
arness=8) -- that data lives in `outputs/task2_context_full/summary_all.csv` even though the
config here was since overwritten by the colleague. Its exact block_length isn't independently
verifiable from the current file.

## Task 4 -- `temp_threshold_study`: temperature × threshold × steps ablation (this project's own)

| experiment | benchmark | gen_length | block_length | steps sweep | temperature sweep | threshold sweep | sample_limit |
|---|---|---|---|---|---|---|---|
| task4_gsm8k_temp_threshold_steps | gsm8k | 256 | 256 | 256/64/32 (3) | 0.0/0.3 (2) | null/0.8/0.9 (3) | 10 |

**1 experiment × 18 combos (3×2×3) × 10 samples = 180 generations total.** Status: was running as
of last update (~2hr estimate); check `outputs/arness_trace/summary_all.csv` for completion.

## Task 5 -- `custom_benchmark`: same Task 1 sweep, on the self-authored `custom_math` benchmark

| experiment | benchmark | arness label | gen_length | gen_steps | block_length | sample_limit |
|---|---|---|---|---|---|---|
| task5_custom_math_ar_like | custom_math | 1 | 256 | 256 | 256 | null (all 20) |
| task5_custom_math_mild_parallel | custom_math | 4 | 256 | 64 | 256 | null (all 20) |
| task5_custom_math_strong_parallel | custom_math | 8 | 256 | 32 | 256 | null (all 20) |

**3 experiments × 20 samples = 60 generations total.** Added to work around the "no paper-internal
benchmark" restriction (Task 1 uses GSM8K/MBPP, which the LLaDA/iLLaDA paper's own eval tables
cover -- `custom_math` is fully original content). ~40% the compute of Task 1's GSM8K half (60 vs
150 generations at the same gen_length), so noticeably cheaper. Status: not yet run.

## Scale comparison (total generations, roughly proportional to GPU cost at fixed gen_length)

| Task | Total generations | Status |
|---|---|---|
| Task 1 (parallel) | 300 | Done |
| Task 2 (context) | 240 | Prep only, no GPU runs yet |
| Task 3 (arness, current) | 55 | Not yet run |
| Task 4 (temp/threshold) | 180 | Was running |
| Task 5 (custom_math) | 60 | Not yet run |

## Comparison against the paper's own settings

The paper almost never states `steps` explicitly -- convention (stated in the LLaDA README FAQ) is
`steps == gen_length` for its main reported numbers. For the one block-diffusion GSM8K setting
(EVAL.md's optimized table), `steps == block_length` is *our own inference*, verified by working
through `generate.py`'s `steps_per_block = total_steps / num_blocks` math, not something the paper
states directly -- flagged below wherever it applies.

### GSM8K-comparable

| Our experiment | our gen_length | our gen_steps | our block_length | closest paper setting | paper gen_length | paper steps | paper block_length |
|---|---|---|---|---|---|---|---|
| Task1 task1_gsm8k_ar_like | 256 | 256 | 256 | LLaDA-Base Tab.1 simplicity setting | 256/512/1024 (interchangeable) | = gen_length (stated) | = gen_length |
| Task1 task1_gsm8k_mild_parallel | 256 | 64 | 256 | none -- paper never tests steps < gen_length with block_length = gen_length | — | — | — |
| Task1 task1_gsm8k_strong_parallel | 256 | 32 | 256 | none | — | — | — |
| Task3 arness_trace_gsm8k_sample7 | 256 | sweep [256,128,64,32,16] | 64 | LLaDA-Instruct GSM8K block-diffusion optimized setting | 256 | ≈ 8 (inferred, not stated) | **8** |
| Task4 task4_gsm8k_temp_threshold_steps | 256 | sweep [256,64,32] | 256 | LLaDA-1.5 GSM8K | 256 | not disclosed | **16** |
| Task5 task5_custom_math_* | 256 | 256/64/32 | 256 | no paper equivalent (original benchmark) -- GSM8K used only as a scale reference | 256 | — | — |

Key gap: every paper setting *tuned* for GSM8K uses `block_length << gen_length` (real semi-AR
splitting: 8 or 16 vs. 256). Our Task 1/4/5 all hold `block_length == gen_length` (never split into
blocks) so that `gen_steps` is the only thing being varied -- this is a deliberate choice to isolate
one variable, not a paper-matching setting. It's also exactly the gap the "does blocking reduce the
collapse" follow-up experiment (proposed, not yet run) would fill in.

### MBPP-comparable

| Our experiment | our gen_length | our gen_steps | our block_length | closest paper setting | paper gen_length | paper steps | paper block_length |
|---|---|---|---|---|---|---|---|
| Task1 task1_mbpp_ar_like | 512 | 512 | 512 | LLaDA-Instruct MBPP main result | **256** | = gen_length (stated) | = gen_length |
| Task1 task1_mbpp_mild_parallel | 512 | 128 | 512 | none | — | — | — |
| Task1 task1_mbpp_strong_parallel | 512 | 64 | 512 | none | — | — | — |
| Task3 arness_trace_mbpp_sample6 | 512 | sweep [512,...,16] | 128 | LLaDA-1.5 MBPP | 512 | not disclosed | **32** |

Flag: our Task 1 MBPP `gen_length=512` is double the paper's own MBPP main-result `gen_length=256`
-- our own experiment-design choice (presumably more room for generated code), not copied from the
paper. Worth being able to justify if asked.

### RULER / context (Task 2)

No paper equivalent at all -- RULER NIAH is outside the LLaDA/iLLaDA paper's own eval benchmarks,
which is exactly why it's unaffected by the "no paper-internal benchmark" restriction. Our setting:
`gen_length=128, gen_steps=128, block_length=128` (single block, no split).
