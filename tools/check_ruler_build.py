#!/usr/bin/env python3
"""Check that Task2 RULER config builds before running the GPU evaluation."""
from pathlib import Path
import sys

repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
sys.path.insert(0, str(repo / 'opencompass'))
sys.path.insert(0, str(repo))

from opencompass.configs.datasets.ruler.ruler_niah_single_1 import ruler_niah_single_1_datasets
from opencompass.utils.build import build_dataset_from_cfg

cfg = ruler_niah_single_1_datasets[0].copy()
# Keep this tiny and fast. Use repeat haystack so this does not require Paul Graham data.
cfg.update(
    type_haystack='repeat',
    max_seq_length=512,
    tokens_to_generate=32,
    num_samples=1,
    depth_percents=[0],
)

ds = build_dataset_from_cfg(cfg)
print('[OK] RULER dataset build works; len =', len(ds))
print('[OK] keys =', list(ds[0].keys()))
print('[OK] answer =', ds[0]['answer'])
