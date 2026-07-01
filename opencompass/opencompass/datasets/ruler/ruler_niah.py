# flake8: noqa: F401, E501
"""RULER Needle-in-a-Haystack dataset for iLLaDA Task 2.

Direct replacement file.

Fixes:
1) Registers RulerNiahDataset under both short and fully-qualified names so
   MMEngine can build generated OpenCompass configs whose type is serialized as
   ``opencompass.datasets.RulerNiahDataset``.
2) Registers RulerNiahEvaluator aliases as well.
3) If a matching prepared jsonl exists in ``data/prepared/ruler_niah_single_1``,
   loads it directly to avoid regenerating samples during OpenCompass
   partitioning.
"""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import tiktoken
from datasets import Dataset
from transformers import AutoTokenizer

from opencompass.datasets.base import BaseDataset
from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

try:
    from opencompass.registry import ICL_EVALUATORS
except Exception:  # pragma: no cover
    ICL_EVALUATORS = None


def _safe_name(value: Any) -> str:
    chars = []
    for char in str(value).lower():
        chars.append(char if char.isalnum() else '_')
    return '_'.join(''.join(chars).split('_'))


def _depth_to_position(depth_percents: Any) -> Optional[str]:
    if depth_percents is None:
        return None
    if isinstance(depth_percents, (list, tuple)):
        if len(depth_percents) != 1:
            return None
        depth = int(depth_percents[0])
    else:
        depth = int(depth_percents)
    return {0: 'front', 50: 'middle', 100: 'back'}.get(depth)


def _candidate_repo_roots() -> List[Path]:
    here = Path(__file__).resolve()
    roots: List[Path] = []
    # /repo/opencompass/opencompass/datasets/ruler/ruler_niah.py -> /repo
    if len(here.parents) >= 5:
        roots.append(here.parents[4])
    roots.append(Path.cwd())
    roots.append(Path.cwd().parent)
    out: List[Path] = []
    for root in roots:
        root = root.resolve()
        if root not in out:
            out.append(root)
    return out


def _find_prepared_file(
    prepared_file_path: str = '',
    max_seq_length: int = 4096,
    tokens_to_generate: int = 128,
    num_samples: int = 500,
    random_seed: int = 42,
    depth_percents: Any = None,
) -> str:
    if prepared_file_path:
        return prepared_file_path

    pos = _depth_to_position(depth_percents)
    if pos is None:
        return ''

    condition_id = _safe_name(
        f'ruler_niah_single_1_ctx{max_seq_length}_pos{pos}_gen{tokens_to_generate}_samples{num_samples}_seed{random_seed}'
    )
    rel = Path('data') / 'prepared' / 'ruler_niah_single_1' / f'{condition_id}.jsonl'
    for root in _candidate_repo_roots():
        candidate = root / rel
        if candidate.exists():
            return str(candidate)
    return ''


def _load_prepared_jsonl(path: str) -> Dataset:
    prepared = get_data_path(path, local_mode=True)
    data = {'prompt': [], 'answer': []}
    with open(prepared, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get('prompt')
            answer = item.get('answer', item.get('answers'))
            if prompt is None or answer is None:
                continue
            data['prompt'].append(prompt)
            data['answer'].append(answer)
    print(f'[iLLaDA task2] loaded prepared RULER data: {prepared} n={len(data["prompt"])}', flush=True)
    return Dataset.from_dict(data)


@LOAD_DATASET.register_module()
class RulerNiahDataset(BaseDataset):
    @staticmethod
    def load(
        base_path: str,
        file_path: str,
        tokens_to_generate: int = 128,
        max_seq_length: int = 4096,
        tokenizer_model: str = 'gpt-4',
        num_samples: int = 500,
        random_seed: int = 42,
        template: str = 'Some special magic {type_needle_v} are hidden within the following text.\nMake sure to memorize it.\nI will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are',
        num_needle_k: int = 1,
        num_needle_v: int = 1,
        num_needle_q: int = 1,
        type_haystack: str = 'essay',
        type_needle_k: str = 'words',
        type_needle_v: str = 'numbers',
        remove_newline_tab: str = '',
        depth_percents=None,
        prepared_file_path: str = '',
    ) -> Dataset:
        prepared = _find_prepared_file(
            prepared_file_path=prepared_file_path,
            max_seq_length=max_seq_length,
            tokens_to_generate=tokens_to_generate,
            num_samples=num_samples,
            random_seed=random_seed,
            depth_percents=depth_percents,
        )
        if prepared:
            return _load_prepared_jsonl(prepared)

        data = {'prompt': [], 'answer': []}

        if tokenizer_model == 'gpt-4':
            tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

        random.seed(random_seed)
        np.random.seed(random_seed)
        num_needle_k = max(num_needle_k, num_needle_q)

        needle = 'One of the special magic {type_needle_v} for {key} is: {value}.'

        if type_haystack == 'essay':
            essay = os.path.join(base_path, file_path)
            essay = get_data_path(essay, local_mode=True)
            combined_essay = ''
            with open(essay, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    line_text = json.loads(line.strip()).get('text', '').strip()
                    combined_essay += line_text + ' '
            haystack = re.sub(r'\s+', ' ', combined_essay).split(' ')
        elif type_haystack == 'repeat':
            haystack = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'
        elif type_haystack == 'needle':
            haystack = needle
        else:
            raise NotImplementedError(f'{type_haystack} is not implemented.')

        try:
            import wonderwords
        except ImportError as exc:
            raise ImportError('Please install wonderwords by: pip install wonderwords') from exc

        nouns = wonderwords.random_word._get_words_from_text_file('nounlist.txt')
        adjs = wonderwords.random_word._get_words_from_text_file('adjectivelist.txt')
        words = sorted(list(set(f'{adj}-{noun}' for adj in adjs for noun in nouns)))

        depths = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))
        if depth_percents is not None:
            depths = [int(depth) for depth in depth_percents]

        def _generate_random_number(num_digits: int = 7) -> str:
            lower_bound = 10 ** (num_digits - 1)
            upper_bound = 10 ** num_digits - 1
            return str(random.randint(lower_bound, upper_bound))

        def _generate_random_word() -> str:
            return random.choice(words)

        def _generate_random_uuid() -> str:
            return str(uuid.UUID(int=random.getrandbits(128), version=4))

        def _generate_random(type_needle: str) -> str:
            if type_needle == 'numbers':
                return _generate_random_number()
            if type_needle == 'words':
                return _generate_random_word()
            if type_needle == 'uuids':
                return _generate_random_uuid()
            raise NotImplementedError(f'{type_needle} is not implemented.')

        def _generate_input_output(num_haystack: int, current_type_needle_v: str, current_template: str):
            keys, values, needles = [], [], []
            for _ in range(num_needle_k):
                keys.append(_generate_random(type_needle_k))
                value = []
                for _ in range(num_needle_v):
                    value.append(_generate_random(current_type_needle_v))
                    needles.append(
                        needle.format(
                            type_needle_v=current_type_needle_v,
                            key=keys[-1],
                            value=value[-1],
                        )
                    )
                values.append(value)

            random.Random(random_seed).shuffle(needles)

            if type_haystack == 'essay':
                text = ' '.join(haystack[:num_haystack])
                document_sents = text.split('. ')
                document_sents = [sentence.strip() for sentence in document_sents if sentence]

                if len(depths) >= len(needles):
                    sampled_depths = random.sample(depths, len(needles))
                else:
                    sampled_depths = [depths[i % len(depths)] for i in range(len(needles))]

                insertion_positions = (
                    [0]
                    + sorted([int(len(document_sents) * (depth / 100)) for depth in sampled_depths])
                    + [len(document_sents)]
                )
                document_sents_list = []
                for i in range(1, len(insertion_positions)):
                    last_pos = insertion_positions[i - 1]
                    next_pos = insertion_positions[i]
                    document_sents_list.append(' '.join(document_sents[last_pos:next_pos]))
                    if i - 1 < len(needles):
                        document_sents_list.append(needles[i - 1])
                context = ' '.join(document_sents_list)
            else:
                if type_haystack == 'repeat':
                    sentences = [haystack] * num_haystack
                elif type_haystack == 'needle':
                    sentences = [
                        haystack.format(
                            type_needle_v=current_type_needle_v,
                            key=_generate_random(type_needle_k),
                            value=_generate_random(current_type_needle_v),
                        )
                        for _ in range(num_haystack)
                    ]
                indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
                for index, element in zip(indexes, needles):
                    sentences.insert(index, element)
                context = '\n'.join(sentences)

            indices = random.sample(range(num_needle_k), num_needle_q)
            queries = [keys[i] for i in indices]
            answers = [a for i in indices for a in values[i]]
            query = ', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]

            final_template = current_template
            final_type_needle_v = current_type_needle_v
            if num_needle_q * num_needle_v == 1:
                final_template = final_template.replace('Some', 'A')
                final_template = final_template.replace('are all', 'is')
                final_template = final_template.replace('are', 'is')
                final_template = final_template.replace('answers', 'answer')
                final_type_needle_v = final_type_needle_v[:-1]

            input_text = final_template.format(
                type_needle_v=final_type_needle_v,
                context=context,
                query=query,
            )
            return input_text, answers

        if type_haystack == 'essay':
            incremental = 500
        else:
            incremental = 25
        if type_haystack != 'essay' and max_seq_length < 4096:
            incremental = 5

        num_haystack = incremental
        total_tokens = 0
        while total_tokens + tokens_to_generate < max_seq_length:
            input_text, answer = _generate_input_output(num_haystack, type_needle_v, template)
            total_tokens = len(tokenizer.encode(input_text + ' '.join(answer)))
            if total_tokens + tokens_to_generate > max_seq_length:
                num_haystack -= incremental
                break
            if type_haystack == 'essay' and num_haystack > len(haystack):
                num_haystack = len(haystack)
                break
            num_haystack += incremental

        for _ in range(num_samples):
            used_haystack = max(incremental, num_haystack)
            while True:
                input_text, answer = _generate_input_output(used_haystack, type_needle_v, template)
                length = len(tokenizer.encode(input_text)) + tokens_to_generate
                if length <= max_seq_length:
                    break
                if used_haystack <= incremental:
                    raise AssertionError(f'{length} exceeds max_seq_length and cannot reduce haystack further.')
                used_haystack -= incremental

            if remove_newline_tab:
                input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
            data['prompt'].append(input_text)
            data['answer'].append(answer)

        return Dataset.from_dict({'prompt': data['prompt'], 'answer': data['answer']})


class RulerNiahEvaluator(BaseEvaluator):
    def score(self, predictions, gold):
        score = (
            sum(
                [
                    sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
                    for pred, ref in zip(predictions, gold)
                ]
            )
            / len(predictions)
            * 100
        )
        return {'score': round(score, 2)}


def _register_aliases():
    for name in (
        'RulerNiahDataset',
        'opencompass.datasets.RulerNiahDataset',
        'opencompass.datasets.ruler.ruler_niah.RulerNiahDataset',
    ):
        try:
            LOAD_DATASET.register_module(name=name, module=RulerNiahDataset, force=True)
        except TypeError:
            if name not in getattr(LOAD_DATASET, 'module_dict', {}):
                LOAD_DATASET.register_module(name=name, module=RulerNiahDataset)

    if ICL_EVALUATORS is not None:
        for name in (
            'RulerNiahEvaluator',
            'opencompass.datasets.RulerNiahEvaluator',
            'opencompass.datasets.ruler.ruler_niah.RulerNiahEvaluator',
        ):
            try:
                ICL_EVALUATORS.register_module(name=name, module=RulerNiahEvaluator, force=True)
            except TypeError:
                if name not in getattr(ICL_EVALUATORS, 'module_dict', {}):
                    ICL_EVALUATORS.register_module(name=name, module=RulerNiahEvaluator)


_register_aliases()
