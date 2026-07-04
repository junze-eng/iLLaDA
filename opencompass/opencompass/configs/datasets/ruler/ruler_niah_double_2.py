# Add this file to: opencompass/configs/datasets/ruler/ruler_niah_double_2.py
# It reuses the existing RULER NIAH single prepared-file dataset config for inference.
# The prepared_file_path/max_seq_length/tokens_to_generate/depth_percents fields are
# overwritten by run_test.py when rendering each OpenCompass config.

from copy import deepcopy

from opencompass.configs.datasets.ruler.ruler_niah_single_1 import ruler_niah_single_1_datasets

ruler_niah_double_2_datasets = deepcopy(ruler_niah_single_1_datasets)
for _dataset in ruler_niah_double_2_datasets:
    _dataset['abbr'] = 'ruler_niah_double_2'
