from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets

from opencompass.models import LLaDAModel
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

datasets = humaneval_datasets
models = [
    dict(
        type=LLaDAModel,
        path='GSAI-ML/iLLaDA-8B-Instruct',
        abbr='illada_8b_instruct_mechanism_humaneval_humaneval_1',
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
        gen_length=512,
        gen_steps=128,
        gen_blocksize=64,
        diff_confidence_eos_eot_inf=False,
        diff_logits_eos_inf=True,
        token_selection_confidence_threshold=None,
        min_transfer_tokens=1,
        return_trace=True,
        trace_token_snapshots=False,
        trace_decode_snapshots=False,
        context_prefix_tokens=0,
        per_sample_output='C:\\Users\\whaletech002\\Desktop\\whaletech\\iLLaDA\\outputs\\illada_runs\\mechanism_humaneval_humaneval_1\\per_sample.jsonl',
        step_trace_output='C:\\Users\\whaletech002\\Desktop\\whaletech\\iLLaDA\\outputs\\illada_runs\\mechanism_humaneval_humaneval_1\\step_trace.jsonl',
        metrics_output='C:\\Users\\whaletech002\\Desktop\\whaletech\\iLLaDA\\outputs\\illada_runs\\mechanism_humaneval_humaneval_1\\per_sample.jsonl',
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
