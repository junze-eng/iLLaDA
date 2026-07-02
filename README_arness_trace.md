# ARness Trace Case Study

This setup writes one folder per sample/condition under:

```text
outputs/arness_trace/sample_traces/<benchmark>/sample_XXXX/<condition>/
```

Each folder contains:

```text
block_timeline.csv
step_events.csv
block_metrics.csv
sample_metrics.json
plot_rows.csv
final_prediction.txt
```

`block_timeline.csv` is the main table. Rows follow the real autoregressive
generation order and include `active_block_idx` plus `block_local_round`.
Columns `block_00`, `block_01`, ... show each block's current visible state.
Unrevealed spans are compressed as `□xN`.

Run the detailed two-sample trace sweep from the unified config. It currently
plans 55 conditions:

- GSM8K sample 7: `gen_steps=[256,128,64,32,16]`
- MBPP sample 6: `gen_steps=[512,256,128,64,32,16]`
- both: `token_selection_confidence_threshold=[null,0.6,0.7,0.8,0.9]`

```bash
python run_test.py --config test_config.yaml
```

Create a compact plotting table after the run:

```bash
python tools/summarize_arness_trace.py
```

If you already have old `summary.jsonl` and `trace.jsonl` run directories, export
them into the same per-sample layout:

```bash
python tools/export_arness_trace.py --runs-root outputs/arness_trace --out outputs/arness_trace/sample_traces
```
