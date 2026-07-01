from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

from opencompass.models import LLaDAModel
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

datasets = gsm8k_datasets
_sample_test_range = '[:100]'
for _dataset in datasets:
    _dataset.setdefault('reader_cfg', {})['test_range'] = _sample_test_range
models = [
    dict(
        type=LLaDAModel,
        path='GSAI-ML/iLLaDA-8B-Instruct',
        abbr='illada_8b_instruct_parallel_gsm8k_steps_gsm8k_13',
        max_out_len=512,
        max_seq_len=4096,
        batch_size=1,
        batch_size_=1,
        mask_id=5,
        cfg=0.0,
        temperature=0.0,
        remasking='low_confidence',
        model_kwargs={'device_map': 'auto', 'torch_dtype': 'torch.bfloat16'},
        run_cfg={'num_gpus': 1},
        gen_length=256,
        gen_steps=256,
        gen_blocksize=128,
        diff_confidence_eos_eot_inf=True,
        diff_logits_eos_inf=False,
        token_selection_confidence_threshold=None,
        min_transfer_tokens=1,
        return_trace=False,
        trace_token_snapshots=False,
        trace_decode_snapshots=False,
        context_prefix_tokens=0,
        per_sample_output='C:\\Users\\whaletech002\\Desktop\\whaletech\\iLLaDA\\outputs\\illada_runs\\parallel_gsm8k_steps_gsm8k_13\\per_sample.jsonl',
        metrics_output='C:\\Users\\whaletech002\\Desktop\\whaletech\\iLLaDA\\outputs\\illada_runs\\parallel_gsm8k_steps_gsm8k_13\\per_sample.jsonl',
    )
]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        num_split=None,
        min_task_size=16,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(type=OpenICLInferTask),
        retry=1,
    ),
)
