# ARness Trace Case Study

Current runs are organized by task, experiment, and condition:

```text
outputs/arness/<experiment>/<condition>/
  summary.jsonl
  trace.jsonl
  sample_traces/
```

Each exported sample trace contains:

```text
block_timeline.csv
step_events.csv
block_metrics.csv
sample_metrics.json
plot_rows.csv
final_prediction.txt
```

`block_timeline.csv` is the main table. Rows follow real generation order and
include `active_block_idx` plus `block_local_round`. Columns `block_00`,
`block_01`, ... show the current visible state of each autoregressive block.
Unrevealed spans are compressed as `[MASK]xN`.

Run the configured two-sample sweep:

```bash
python run_test.py --config test_config.yaml --only arness
```

Repair or export completed runs:

```bash
python trace.py --runs-root outputs/arness --overwrite --compress-masks --write-task-index
```

Repair old numbered runs into the new layout:

```bash
python trace.py --runs-root outputs/arness_trace --task-output-root outputs/arness --canonicalize --mode copy --overwrite --compress-masks --write-task-index
```
