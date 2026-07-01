from opencompass.datasets import RulerNiahDataset, RulerNiahEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


ruler_niah_single_1_reader_cfg = dict(input_columns=['prompt'], output_column='answer')

ruler_niah_single_1_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{prompt}'), dict(role='BOT', prompt='')]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=128),
)

ruler_niah_single_1_eval_cfg = dict(evaluator=dict(type=RulerNiahEvaluator), pred_role='BOT')

ruler_niah_single_1_datasets = [
    dict(
        type=RulerNiahDataset,
        abbr='ruler_niah_single_1',
        base_path='data/ruler',
        file_path='paul_graham_essay.jsonl',
        tokens_to_generate=128,
        max_seq_length=512,
        num_samples=20,
        tokenizer_model='gpt-4',
        type_haystack='essay',
        num_needle_k=1,
        num_needle_v=1,
        num_needle_q=1,
        reader_cfg=ruler_niah_single_1_reader_cfg,
        infer_cfg=ruler_niah_single_1_infer_cfg,
        eval_cfg=ruler_niah_single_1_eval_cfg,
    )
]
