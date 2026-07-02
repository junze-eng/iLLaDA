"""Self-authored grade-school arithmetic word problem set.

Not derived from GSM8K or any other published benchmark: every question and
few-shot demonstration below was written from scratch for this project,
specifically so it falls outside the set of benchmarks the LLaDA/iLLaDA paper
already reports results on. It reuses GSM8K's *scoring machinery* (extract the
final number, string-match against the reference) since that machinery is
generic and not benchmark content.
"""
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator

custom_math_reader_cfg = dict(input_columns=['question'], output_column='answer')

custom_math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="Question: A classroom has 24 students. The teacher forms groups of 4 for a project. How many groups are formed?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt="The teacher forms 24 / 4 = 6 groups.\nThe answer is 6\n"),
                dict(role='HUMAN', prompt="Question: Maya has 15 dollars. She buys a notebook for 6 dollars and a pen for 2 dollars. How much money does she have left?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt="Maya spends 6 + 2 = 8 dollars in total.\nShe has 15 - 8 = 7 dollars left.\nThe answer is 7\n"),
                dict(role='HUMAN', prompt="Question: A library receives 40 new books. It puts 25 books on the main shelf and splits the remaining books equally between 3 smaller shelves. How many books go on each smaller shelf?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt="The remaining books are 40 - 25 = 15.\nSplit equally between 3 shelves, each shelf gets 15 / 3 = 5 books.\nThe answer is 5\n"),
                dict(role='HUMAN', prompt="Question: A bus has 48 seats. On the first trip, 30 passengers board. On the second trip, 18 more passengers than the first trip board the bus. How many passengers are on the second trip?\nLet's think step by step\nAnswer:"),
                dict(role='BOT', prompt="The second trip has 30 + 18 = 48 passengers.\nThe answer is 48\n"),
                dict(role='HUMAN', prompt="Question: {question}\nLet's think step by step\nAnswer:"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

custom_math_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                             pred_postprocessor=dict(type=gsm8k_postprocess),
                             dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

custom_math_datasets = [
    dict(
        abbr='custom_math',
        type=GSM8KDataset,
        # placeholder; run_test.py rewrites this to an absolute path (repo_root/data/custom_math)
        # in the generated top-level config -- mmengine's read_base() lazy-parses this module, so
        # it cannot compute __file__-relative paths itself (method calls on LazyObject raise).
        path='data/custom_math',
        reader_cfg=custom_math_reader_cfg,
        infer_cfg=custom_math_infer_cfg,
        eval_cfg=custom_math_eval_cfg)
]
